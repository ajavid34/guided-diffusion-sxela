"""
Fine-tune a pre-trained noised image classifier with Entropy-Constraint Training (ECT) 
and Mutual Information regularization for improved conditional diffusion generation.
"""

import argparse
import os
import sys
import math

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

# Import your loss functions - make sure losses.py is in the path
try:
    from losses import get_loss
except ImportError:
    print("Warning: losses.py not found. Using basic KL divergence only.")
    get_loss = None


def compute_uniform_distribution(batch_size, num_classes, device):
    """Create uniform distribution for ECT loss computation."""
    uniform_dist = th.ones(batch_size, num_classes, device=device) / num_classes
    return uniform_dist


def compute_ect_loss(logits, num_classes, divergence_type="KL", divergence_params=None):
    """
    Compute Entropy-Constraint Training (ECT) loss.
    
    Args:
        logits: Classifier predictions [batch_size, num_classes]
        num_classes: Number of classes
        divergence_type: Type of f-divergence to use
        divergence_params: Parameters for the divergence measure
    
    Returns:
        ECT loss value
    """
    batch_size = logits.shape[0]
    device = logits.device
    
    # Convert logits to probabilities
    pred_probs = F.softmax(logits, dim=1)
    
    # Create uniform distribution (target for ECT)
    uniform_dist = compute_uniform_distribution(batch_size, num_classes, device)
    
    if get_loss is not None:
        # Use the comprehensive loss library
        try:
            divergence_fn = get_loss(
                name=divergence_type,
                param=divergence_params,
                nclass=num_classes,
                softmax_logits=False,  # Already converted to probabilities
                softmax_gt=False       # Uniform distribution doesn't need softmax
            )
            # Note: Most f-divergences compute D_f(P||Q), we want D_f(pred||uniform)
            ect_loss = divergence_fn(pred_probs, uniform_dist)
        except Exception as e:
            logger.log(f"Error with {divergence_type} divergence: {e}. Falling back to KL.")
            # Fallback to KL divergence
            log_uniform = th.log(uniform_dist + 1e-8)
            ect_loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_dist, reduction='batchmean')
    else:
        # Fallback implementation using KL divergence
        log_uniform = th.log(uniform_dist + 1e-8)
        ect_loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_dist, reduction='batchmean')
    
    return ect_loss


def compute_mi_regularization(batch, reconstructed, latent_vars=None, mi_type="KL"):
    """
    Compute mutual information regularization term.
    
    Args:
        batch: Original clean images [batch_size, channels, height, width]
        reconstructed: Reconstructed/denoised images  
        latent_vars: Latent variables (if available)
        mi_type: Type of divergence for MI computation
    
    Returns:
        MI regularization loss
    """
    # Simplified MI regularization using reconstruction error
    if reconstructed is not None:
        # L2 reconstruction loss as a proxy for MI
        mi_loss = F.mse_loss(batch, reconstructed, reduction='mean')
    else:
        # If no reconstruction available, return zero loss
        mi_loss = th.tensor(0.0, device=batch.device)
    
    return mi_loss


def freeze_layers(model, freeze_strategy="none"):
    """
    Freeze specific layers during fine-tuning.
    
    Args:
        model: The classifier model
        freeze_strategy: Strategy for freezing layers
            - "none": Don't freeze anything
            - "early": Freeze early layers (first half)
            - "backbone": Freeze everything except final classifier
            - "gradual": Gradually unfreeze layers during training
    """
    if freeze_strategy == "none":
        return
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    total_layers = len(named_params)
    
    if freeze_strategy == "early":
        # Freeze first half of layers
        freeze_count = total_layers // 2
        for i, (name, param) in enumerate(named_params):
            if i < freeze_count:
                param.requires_grad = False
                logger.log(f"Frozen layer: {name}")
    
    elif freeze_strategy == "backbone":
        # Freeze everything except final classification layers
        for name, param in named_params:
            if "final" in name.lower() or "classifier" in name.lower() or "head" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
                logger.log(f"Frozen layer: {name}")
    
    logger.log(f"Applied freeze strategy: {freeze_strategy}")


def unfreeze_gradually(model, current_step, total_steps, unfreeze_schedule="linear"):
    """
    Gradually unfreeze layers during training.
    
    Args:
        model: The classifier model
        current_step: Current training step
        total_steps: Total training steps
        unfreeze_schedule: Schedule for unfreezing ("linear", "exponential")
    """
    if unfreeze_schedule == "none":
        return
    
    named_params = list(model.named_parameters())
    total_layers = len(named_params)
    
    if unfreeze_schedule == "linear":
        # Linearly unfreeze layers based on training progress
        progress = current_step / total_steps
        layers_to_unfreeze = int(progress * total_layers)
        
        for i, (name, param) in enumerate(named_params):
            if i < layers_to_unfreeze:
                param.requires_grad = True


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    
    # Load pre-trained classifier for fine-tuning
    if args.pretrained_classifier_path:
        logger.log(f"Loading pre-trained classifier from: {args.pretrained_classifier_path}")
        pretrained_state = dist_util.load_state_dict(args.pretrained_classifier_path, map_location="cpu")
        
        # Handle potential mismatches in state dict keys
        model_state = model.state_dict()
        filtered_state = {}
        
        for key, value in pretrained_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                logger.log(f"Skipping parameter {key} due to shape mismatch or absence")
        
        model.load_state_dict(filtered_state, strict=False)
        logger.log(f"Loaded {len(filtered_state)} parameters from pre-trained model")
    else:
        logger.log("No pre-trained classifier specified. Training from scratch.")
    
    model.to(dist_util.dev())
    
    # Apply layer freezing strategy
    if args.freeze_strategy != "none":
        freeze_layers(model, args.freeze_strategy)
    
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
    # Use different learning rates for different parts if doing partial fine-tuning
    if args.freeze_strategy == "backbone":
        # Lower learning rate for fine-tuning
        opt = AdamW(
            filter(lambda p: p.requires_grad, mp_trainer.master_params), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    logger.log("fine-tuning classifier model with ECT and MI regularization...")
    
    # Log training configuration
    logger.log(f"Fine-tuning mode: {'Yes' if args.pretrained_classifier_path else 'No'}")
    logger.log(f"Freeze strategy: {args.freeze_strategy}")
    logger.log(f"ECT enabled: {args.use_ect}")
    logger.log(f"ECT weight (η): {args.ect_weight}")
    logger.log(f"ECT divergence: {args.ect_divergence}")
    logger.log(f"MI regularization enabled: {args.use_mi_reg}")
    logger.log(f"MI weight (ζ): {args.mi_weight}")
    logger.log(f"Learning rate: {args.lr}")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        
        # Store clean batch for MI regularization
        clean_batch = batch.clone()
        
        # Add noise for noisy classifier training
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            # Forward pass through classifier
            logits = model(sub_batch, timesteps=sub_t)
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits, sub_labels, reduction="none")
            
            # Initialize total loss with CE loss
            total_loss = ce_loss.mean()
            
            # Entropy-Constraint Training (ECT) loss
            ect_loss = th.tensor(0.0, device=sub_batch.device)
            if args.use_ect:
                try:
                    # Prepare divergence parameters
                    divergence_params = []
                    if args.ect_divergence in ['PowD', 'GenKL', 'Exp', 'LeCam', 'AlphaRenyi', 'BetaSkew', 'Tsallis', 'RenyiDivergence']:
                        divergence_params = [args.ect_alpha]
                    
                    ect_loss = compute_ect_loss(
                        logits, 
                        args.num_classes,
                        divergence_type=args.ect_divergence,
                        divergence_params=divergence_params if divergence_params else None
                    )
                    total_loss = total_loss + args.ect_weight * ect_loss
                except Exception as e:
                    logger.log(f"ECT computation error: {e}")
                    ect_loss = th.tensor(0.0, device=sub_batch.device)
            
            # Mutual Information regularization
            mi_loss = th.tensor(0.0, device=sub_batch.device)
            if args.use_mi_reg:
                try:
                    # For MI regularization, we can use the difference between noisy and clean predictions
                    if args.noised:
                        # Get clean batch corresponding to sub_batch
                        start_idx = i * args.microbatch if args.microbatch > 0 else 0
                        end_idx = start_idx + len(sub_batch)
                        sub_clean_batch = clean_batch[start_idx:end_idx]
                        
                        # Compute MI regularization
                        mi_loss = compute_mi_regularization(
                            sub_clean_batch, 
                            sub_batch,  # This could be replaced with reconstructed images if available
                            mi_type=args.mi_divergence
                        )
                        total_loss = total_loss + args.mi_weight * mi_loss
                except Exception as e:
                    logger.log(f"MI regularization computation error: {e}")
                    mi_loss = th.tensor(0.0, device=sub_batch.device)

            # Compute accuracy metrics
            losses = {}
            losses[f"{prefix}_ce_loss"] = ce_loss.detach()
            losses[f"{prefix}_ect_loss"] = ect_loss.detach()
            losses[f"{prefix}_mi_loss"] = mi_loss.detach()
            losses[f"{prefix}_total_loss"] = total_loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            
            # Backward pass
            if total_loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(total_loss * len(sub_batch) / len(batch))

    # Training loop
    for step in range(args.iterations - resume_step):
        current_step = step + resume_step
        
        logger.logkv("step", current_step)
        logger.logkv(
            "samples",
            (current_step + 1) * args.batch_size * dist.get_world_size(),
        )
        
        # Gradual unfreezing
        if args.freeze_strategy == "gradual":
            unfreeze_gradually(model.module, current_step, args.iterations, "linear")
        
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, current_step / args.iterations)
        
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
            and not current_step % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, current_step)

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
        iterations=10000,  # Fewer iterations for fine-tuning
        lr=1e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=50,  # More frequent logging
        eval_interval=200,
        save_interval=2000,  # More frequent saves
        
        # Fine-tuning specific parameters
        pretrained_classifier_path="",  # Path to pre-trained classifier
        freeze_strategy="none",  # Options: "none", "early", "backbone", "gradual"
        
        # ECT (Entropy-Constraint Training) parameters
        use_ect=True,
        ect_weight=0.1,  # η parameter in the paper
        ect_divergence="KL",  # Type of f-divergence for ECT
        ect_alpha=2.0,  # Alpha parameter for divergences that need it
        
        # MI (Mutual Information) regularization parameters  
        use_mi_reg=False,  # Set to True when you have proper latent variables
        mi_weight=0.01,  # ζ parameter in the paper
        mi_divergence="KL",  # Type of divergence for MI regularization
        
        # Number of classes (adjust based on your dataset)
        num_classes=1000,  # ImageNet default, change for CIFAR-10 (10) or other datasets
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
