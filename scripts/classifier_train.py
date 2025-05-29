"""
Train a noised image classifier on ImageNet with Entropy-Constraint Training (ECT)
and Mutual Information regularization.
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

# Import the losses module - adjust path as needed
# Try multiple possible locations for the losses module
try:
    from losses import get_loss
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from losses import get_loss
    except ImportError:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from losses import get_loss


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
    
    else:
        # Default to Shannon entropy
        return -(probs * th.log(probs)).sum(dim=-1)


def compute_mi_regularization(logits, targets, divergence_name='JS', divergence_params=None):
    """
    Compute mutual information regularization using f-divergence.
    
    This approximates the MI regularization term D_f[p(x,z)||p(x)p(z)]
    by treating the classifier's hidden representations as latent variables.
    """
    # For simplicity, we use the logits themselves as a proxy for latent representations
    # In practice, you might want to extract actual hidden layer features
    
    batch_size = logits.shape[0]
    n_classes = logits.shape[1]
    
    # Compute marginal distributions
    # p(z) ≈ average over batch
    marginal_logits = logits.mean(dim=0, keepdim=True).expand(batch_size, -1)
    
    # Get the divergence loss function
    div_loss = get_loss(divergence_name, param=divergence_params, nclass=n_classes,
                       softmax_logits=True, softmax_gt=False)
    
    # Compute divergence between joint and product of marginals
    # This encourages the model to use the latent information
    mi_loss = div_loss(logits, marginal_logits.detach())
    
    return mi_loss


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
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
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
        random_crop=True,
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

    # Initialize ECT loss
    num_classes = model.module.out_channels if hasattr(model, 'module') else model.out_channels
    ect_loss_fn = get_loss(
        args.ect_divergence, 
        param=args.divergence_params,
        nclass=num_classes,
        softmax_logits=True,
        softmax_gt=False
    )

    logger.log("training classifier model with ECT...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
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
            
            # Compute ECT loss
            probs = F.softmax(logits, dim=-1)
            uniform_dist = th.ones_like(probs) / num_classes
            ect_loss = ect_loss_fn(probs, uniform_dist)
            
            # Compute MI regularization if enabled
            mi_loss = th.tensor(0.0).to(dist_util.dev())
            if args.mi_weight > 0:
                mi_loss = compute_mi_regularization(
                    logits, sub_labels, 
                    divergence_name=args.mi_divergence,
                    divergence_params=args.divergence_params
                )
            
            # Total loss
            total_loss = ce_loss + args.ect_weight * ect_loss + args.mi_weight * mi_loss

            losses = {}
            losses[f"{prefix}_loss"] = ce_loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            
            # Log timestep-dependent losses for log_loss_dict
            log_loss_dict(diffusion, sub_t, losses)
            
            # Log scalar losses separately
            logger.logkv(f"{prefix}_ect_loss", ect_loss.detach().mean().item())
            logger.logkv(f"{prefix}_mi_loss", mi_loss.detach().mean().item())
            logger.logkv(f"{prefix}_total_loss", total_loss.detach().mean().item())
            
            # Log entropy of predictions
            entropy = compute_entropy(probs, entropy_type=args.entropy_type, alpha=args.entropy_alpha)
            logger.logkv(f"{prefix}_entropy", entropy.detach().mean().item())
            del losses
            
            loss = total_loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

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
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


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
        ect_weight=0.1,
        ect_divergence="JS",  # Options: KL, JS, Hellinger, etc.
        # MI regularization parameters
        mi_weight=0.01,
        mi_divergence="JS",
        # Entropy parameters
        entropy_type="renyi",  # Options: renyi, tsallis, min, collision
        entropy_alpha=2.0,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Custom parsing for divergence_params (not in defaults to avoid conflict)
    parser.add_argument('--divergence_params', nargs='+', type=float, default=None,
                       help='Parameters for divergence measures')
    
    return parser


if __name__ == "__main__":
    main()
