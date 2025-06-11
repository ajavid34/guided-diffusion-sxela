"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images with Entropy-Driven Sampling (EDS).
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def compute_entropy(probs, entropy_type='renyi', alpha=2.0):
    """Compute entropy of probability distribution."""
    # Add small epsilon for numerical stability
    probs = probs + 1e-8
    
    if entropy_type == 'renyi':
        if abs(alpha - 1.0) < 1e-6:
            # Shannon entropy (limit case)
            return -(probs * th.log(probs)).sum(dim=-1)
        else:
            # Rényi entropy
            return (1.0 / (1.0 - alpha)) * th.log((probs ** alpha).sum(dim=-1))
    
    elif entropy_type == 'tsallis':
        q = alpha  # Using alpha parameter as q for Tsallis
        if abs(q - 1.0) < 1e-6:
            # Shannon entropy (limit case)
            return -(probs * th.log(probs)).sum(dim=-1)
        else:
            # Tsallis entropy
            return (1.0 / (q - 1.0)) * (1.0 - (probs ** q).sum(dim=-1))
    
    elif entropy_type == 'min':
        # Min-entropy
        return -th.log(probs.max(dim=-1)[0])
    
    elif entropy_type == 'collision':
        # Collision entropy (Rényi entropy with α=2)
        return -th.log((probs ** 2).sum(dim=-1))

    elif entropy_type == 'js':
    # JS divergence from uniform
    num_classes = probs.shape[-1]
    uniform_prob = 1.0 / num_classes
    
    # Mixture distribution: m = (p + u) / 2
    mixture = 0.5 * (probs + uniform_prob)
    
    # JS divergence components
    # Note: For entropy regularization, we use negative JS divergence
    # so that maximizing entropy corresponds to minimizing distance from uniform
    kl_p_m = (probs * (th.log(probs) - th.log(mixture))).sum(dim=-1)
    kl_u_m = uniform_prob * num_classes * (th.log(th.tensor(uniform_prob)) - th.log(mixture).sum(dim=-1) / num_classes)
    
    # JS divergence
    js_div = 0.5 * (kl_p_m + kl_u_m)
    
    # Return negative JS divergence so it behaves like entropy
    # (higher value = more uniform = more entropy)
    return -js_div
    
    else:
        # Default to Shannon entropy
        return -(probs * th.log(probs)).sum(dim=-1)


def compute_scaling_factor(logits, num_classes, entropy_weight, entropy_type='renyi', entropy_alpha=2.0):
    """
    Compute the entropy-aware scaling factor s(x_t, φ).
    
    Args:
        logits: Classifier output logits
        num_classes: Number of classes
        entropy_weight: Weight parameter γ
        entropy_type: Type of entropy to use
        entropy_alpha: Alpha parameter for entropy computation
    
    Returns:
        Scaling factors for each sample in the batch
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy of predicted distribution
    H_pred = compute_entropy(probs, entropy_type=entropy_type, alpha=entropy_alpha)
    
    # Compute entropy of uniform distribution (theoretical maximum)
    uniform_probs = th.ones_like(probs) / num_classes
    H_uniform = compute_entropy(uniform_probs, entropy_type=entropy_type, alpha=entropy_alpha)
    
    # Compute scaling factor: s(x_t, φ) = γ * H(U(ỹ)) / H(p_φ(ỹ|x_t))
    # Add small epsilon to avoid division by zero
    scaling_factor = entropy_weight * H_uniform / (H_pred + 1e-8)
    
    return scaling_factor


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        """Conditional function with Entropy-Driven Sampling."""
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            
            # Compute gradient
            grad = th.autograd.grad(selected.sum(), x_in)[0]
            
            # Apply Entropy-Driven Sampling if enabled
            if args.use_eds:
                # Compute scaling factors for each sample in the batch
                scaling_factors = compute_scaling_factor(
                    logits, 
                    NUM_CLASSES, 
                    args.entropy_weight,
                    entropy_type=args.entropy_type,
                    entropy_alpha=args.entropy_alpha
                )
                
                # Reshape scaling factors to match gradient dimensions
                scaling_factors = scaling_factors.view(-1, 1, 1, 1)
                
                # Apply scaling to gradients
                grad = grad * scaling_factors
                
                # Log entropy information
                if args.log_entropy:
                    probs = F.softmax(logits, dim=-1)
                    H_pred = compute_entropy(probs, entropy_type=args.entropy_type, alpha=args.entropy_alpha)
                    logger.logkv(f"entropy_t{t[0].item()}", H_pred.mean().item())
                    logger.logkv(f"scaling_factor_t{t[0].item()}", scaling_factors.mean().item())
            
            return grad * args.classifier_scale

    def model_fn(x, t, y=None):
        # For unconditional diffusion models, don't pass y
        if args.class_cond and y is not None:
            return model(x, t, y)
        else:
            return model(x, t)

    logger.log("sampling with Entropy-Driven Sampling..." if args.use_eds else "sampling...")
    all_images = []
    all_labels = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        
        # Choose sampling method
        if args.use_ddim:
            sample_fn = diffusion.ddim_sample_loop
            logger.log("Using DDIM sampling")
        else:
            sample_fn = diffusion.p_sample_loop
            logger.log("Using DDPM sampling")
        
        # Run sampling with conditional guidance
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            progress=args.show_progress,
        )
        
        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        
        if args.log_entropy:
            logger.dumpkvs()

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        if args.use_eds:
            # Add EDS info to filename
            out_path = out_path.replace(".npz", f"_eds_{args.entropy_type}_alpha{args.entropy_alpha}_gamma{args.entropy_weight}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        # Entropy-Driven Sampling parameters
        use_eds=True,
        entropy_type="renyi",  # Options: renyi, tsallis, min, collision
        entropy_alpha=2.0,     # Alpha parameter for Rényi/Tsallis entropy
        entropy_weight=1.0,    # Gamma parameter in the paper
        log_entropy=False,     # Whether to log entropy values during sampling
        show_progress=True,    # Show sampling progress
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
