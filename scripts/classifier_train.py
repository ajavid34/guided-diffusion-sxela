
"""
Train a noised image classifier with improved ECT implementation
"""

import argparse
import os
import sys

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

# Import losses module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from losses import get_loss
except:
    print("Warning: losses module not found, using basic ECT")

def compute_entropy(probs, entropy_type='renyi', alpha=2.0):
    """Compute entropy with numerical stability"""
    # Ensure numerical stability
    probs = probs.clamp(min=1e-8, max=1.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
    
    if entropy_type == 'renyi':
        if abs(alpha - 1.0) < 1e-6:
            return -(probs * th.log(probs)).sum(dim=-1)
        else:
            return (1.0 / (1.0 - alpha)) * th.log((probs ** alpha).sum(dim=-1))
    elif entropy_type == 'tsallis':
        q = alpha
        if abs(q - 1.0) < 1e-6:
            return -(probs * th.log(probs)).sum(dim=-1)
        else:
            return (1.0 / (q - 1.0)) * (1.0 - (probs ** q).sum(dim=-1))
    elif entropy_type == 'min':
        return -th.log(probs.max(dim=-1)[0])
    elif entropy_type == 'collision':
        return -th.log((probs ** 2).sum(dim=-1))
    else:
        return -(probs * th.log(probs)).sum(dim=-1)

def compute_ect_loss(probs, target_dist, divergence_type='JS'):
    """Compute ECT loss with various f-divergences"""
    # Ensure numerical stability
    probs = probs.clamp(min=1e-8, max=1.0)
    target_dist = target_dist.clamp(min=1e-8, max=1.0)
    
    if divergence_type == 'KL':
        return (target_dist * (th.log(target_dist) - th.log(probs))).sum(dim=-1).mean()
    elif divergence_type == 'JS':
        m = 0.5 * (probs + target_dist)
        kl1 = (probs * (th.log(probs) - th.log(m))).sum(dim=-1)
        kl2 = (target_dist * (th.log(target_dist) - th.log(m))).sum(dim=-1)
        return 0.5 * (kl1 + kl2).mean()
    elif divergence_type == 'Hellinger':
        return 0.5 * ((th.sqrt(probs) - th.sqrt(target_dist)) ** 2).sum(dim=-1).mean()
    else:
        # Default to JS
        return compute_ect_loss(probs, target_dist, 'JS')

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {args.resume_checkpoint}...")
            model.load_state_dict(
                dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
            )

    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
    )
    
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    # Get number of classes
    num_classes = model.module.out_channels if hasattr(model, 'module') else model.out_channels
    
    logger.log("training classifier model with improved ECT...")
    
    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        
        # Add noise to images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits, sub_labels, reduction="none")
            
            # Compute probabilities
            probs = F.softmax(logits, dim=-1)
            
            # ECT loss - encourage uncertainty on noisy data
            # Use a mixture of uniform and true label distribution
            uniform_dist = th.ones_like(probs) / num_classes
            
            # Create target distribution that's between uniform and one-hot
            # More noise -> more uniform, less noise -> more one-hot
            noise_level = sub_t.float() / 1000.0  # Normalize timesteps
            target_dist = noise_level.unsqueeze(1) * uniform_dist +                          (1 - noise_level.unsqueeze(1)) * F.one_hot(sub_labels, num_classes).float()
            
            ect_loss = compute_ect_loss(probs, target_dist, args.ect_divergence)
            
            # Mutual information regularization
            mi_loss = 0.0
            if args.mi_weight > 0:
                # Simplified MI: encourage diversity in predictions
                batch_probs_mean = probs.mean(dim=0)
                mi_loss = -compute_entropy(batch_probs_mean, args.entropy_type, args.entropy_alpha)
            
            # Total loss with adaptive weighting
            # Increase ECT weight for high-noise samples
            adaptive_ect_weight = args.ect_weight * (1 + noise_level.mean())
            total_loss = ce_loss + adaptive_ect_weight * ect_loss + args.mi_weight * mi_loss
            
            # Logging
            losses = {}
            losses[f"{prefix}_loss"] = ce_loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(logits, sub_labels, k=1, reduction="none")
            losses[f"{prefix}_acc@5"] = compute_top_k(logits, sub_labels, k=5, reduction="none")
            
            log_loss_dict(diffusion, sub_t, losses)
            
            # Log scalar metrics
            logger.logkv(f"{prefix}_ect_loss", ect_loss.detach().mean().item())
            logger.logkv(f"{prefix}_mi_loss", mi_loss if isinstance(mi_loss, float) else mi_loss.detach().mean().item())
            logger.logkv(f"{prefix}_total_loss", total_loss.detach().mean().item())
            
            # Log entropy
            entropy = compute_entropy(probs, entropy_type=args.entropy_type, alpha=args.entropy_alpha)
            logger.logkv(f"{prefix}_entropy", entropy.detach().mean().item())
            
            # Log noise-specific metrics
            for noise_quartile in [0, 250, 500, 750]:
                mask = (sub_t >= noise_quartile) & (sub_t < noise_quartile + 250)
                if mask.any():
                    logger.logkv(f"{prefix}_entropy_t{noise_quartile}", entropy[mask].mean().item())
            
            loss = total_loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 0
    max_patience = 5000
    
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        
        # Validation
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        
        # Logging
        if not step % args.log_interval:
            logger.dumpkvs()
        
        # Saving
        if step and dist.get_rank() == 0 and not (step + resume_step) % args.save_interval:
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)
            
            # Early stopping check
            current_loss = logger.get_current().get('train_total_loss', float('inf'))
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                patience = 0
            else:
                patience += args.save_interval
                
            if patience > max_patience:
                logger.log(f"Early stopping at step {step + resume_step}")
                break

    if dist.get_rank() == 0:
        logger.log("saving final model...")
        save_model(mp_trainer, opt, step + resume_step)
    
    dist.barrier()

def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        save_dir = logger.get_dir()
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"model{step:06d}.pt")
        opt_path = os.path.join(save_dir, f"opt{step:06d}.pt")
        
        logger.log(f"Saving model to {model_path}")
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            model_path
        )
        th.save(opt.state_dict(), opt_path)
        logger.log("Model saved successfully!")

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        # ECT parameters
        ect_weight=0.5,  # Increased default
        ect_divergence="JS",
        # MI regularization parameters
        mi_weight=0.05,  # Increased default
        mi_divergence="JS",
        # Entropy parameters
        entropy_type="renyi",
        entropy_alpha=2.0,
        # Data augmentation
        random_crop=True,
        random_flip=True,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add custom divergence params
    parser.add_argument('--divergence_params', nargs='+', type=float, default=None,
                       help='Parameters for divergence measures')
    
    return parser

if __name__ == "__main__":
    main()
