"""
Enhanced classifier training with Entropy-Constraint Training (ECT) and f-divergences
Based on your research proposal for information-theoretic conditional diffusion
"""

import argparse
import os
import math
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

# Import your existing modules
from entropy_driven_guided_diffusion import dist_util, logger
from entropy_driven_guided_diffusion.fp16_util import MixedPrecisionTrainer
from entropy_driven_guided_diffusion.image_datasets import load_data
from entropy_driven_guided_diffusion.resample import create_named_schedule_sampler
from entropy_driven_guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from entropy_driven_guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

# Import the losses module you provided
from entropy_driven_guided_diffusion.losses import get_loss


class EntropyConstraintLoss:
    """
    Implements various entropy measures and f-divergences for ECT
    """
    def __init__(self, num_classes, divergence_type='KL', entropy_type='renyi', 
                 alpha=2.0, divergence_params=None):
        self.num_classes = num_classes
        self.divergence_type = divergence_type
        self.entropy_type = entropy_type
        self.alpha = alpha
        
        # Create uniform distribution for ECT
        self.uniform_dist = th.ones(num_classes) / num_classes
        
        # Initialize f-divergence loss
        self.divergence_loss = get_loss(
            name=divergence_type,
            param=divergence_params or [1.0, alpha],
            nclass=num_classes,
            softmax_logits=True,
            softmax_gt=False
        )
    
    def compute_renyi_entropy(self, probs, alpha=None):
        """Compute Rényi entropy H_α(p)"""
        if alpha is None:
            alpha = self.alpha
            
        if abs(alpha - 1.0) < 1e-6:
            # Shannon entropy limit case
            return -(probs * th.log(th.clamp(probs, min=1e-8))).sum(dim=-1)
        else:
            # Rényi entropy
            return (1.0 / (1.0 - alpha)) * th.log(
                th.clamp((probs ** alpha).sum(dim=-1), min=1e-8)
            )
    
    def compute_tsallis_entropy(self, probs, q=None):
        """Compute Tsallis entropy S_q(p)"""
        if q is None:
            q = self.alpha
        return (1.0 / (q - 1.0)) * (1.0 - (probs ** q).sum(dim=-1))
    
    def compute_min_entropy(self, probs):
        """Compute min-entropy H_∞(p)"""
        return -th.log(th.clamp(th.max(probs, dim=-1)[0], min=1e-8))
    
    def compute_collision_entropy(self, probs):
        """Compute collision entropy H_2(p)"""
        return -th.log(th.clamp((probs ** 2).sum(dim=-1), min=1e-8))
    
    def compute_entropy(self, probs):
        """Compute entropy based on specified type"""
        if self.entropy_type == 'shannon':
            return -(probs * th.log(th.clamp(probs, min=1e-8))).sum(dim=-1)
        elif self.entropy_type == 'renyi':
            return self.compute_renyi_entropy(probs)
        elif self.entropy_type == 'tsallis':
            return self.compute_tsallis_entropy(probs)
        elif self.entropy_type == 'min':
            return self.compute_min_entropy(probs)
        elif self.entropy_type == 'collision':
            return self.compute_collision_entropy(probs)
        else:
            raise ValueError(f"Unknown entropy type: {self.entropy_type}")
    
    def compute_ect_loss(self, logits, device):
        """
        Compute Entropy-Constraint Training loss using f-divergence
        L_ECT = D_f(p_φ(ỹ|x_t, σ(t)) || U(ỹ))
        """
        # Get predicted probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Create uniform distribution target
        uniform_target = self.uniform_dist.to(device).unsqueeze(0).expand(probs.size(0), -1)
        
        # Compute f-divergence loss
        ect_loss = self.divergence_loss(logits, uniform_target)
        
        return ect_loss
    
    def compute_mutual_info_regularization(self, z_logits, x0_features):
        """
        Compute mutual information regularization term
        D_{x_0,z} = D_f[p_φ(x_0, z) || p_data(x_0)p_φ(z)]
        
        This is a simplified implementation - you may need to adapt based on your model architecture
        """
        if z_logits is None or x0_features is None:
            return th.tensor(0.0, device=x0_features.device if x0_features is not None else z_logits.device)
        
        # Compute joint distribution approximation (simplified)
        joint_probs = F.softmax(z_logits, dim=-1)
        marginal_z = joint_probs.mean(dim=0, keepdim=True).expand_as(joint_probs)
        
        # Use same divergence for MI regularization
        mi_reg = self.divergence_loss(z_logits, marginal_z.detach())
        
        return mi_reg


def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist(local_rank=args.local_rank)
    logger.configure(dir=args.log_dir)
    logger.log('current rank == {}, total_num = {}'.format(dist.get_rank(), dist.get_world_size()))
    logger.log(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    # Initialize ECT loss with specified divergence
    ect_loss_fn = EntropyConstraintLoss(
        num_classes=NUM_CLASSES,
        divergence_type=args.divergence_type,
        entropy_type=args.entropy_type,
        alpha=args.alpha,
        divergence_params=[1.0, args.alpha] if args.alpha != 1.0 else None
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
        dataset_type=args.dataset_type
    )
    
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            dataset_type=args.dataset_type
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.resume_checkpoint:
        opt_checkpoint = os.path.join(
            os.path.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    logger.log("training classifier model...")

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
            # Forward pass
            model_output = model(sub_batch, timesteps=sub_t)
            if isinstance(model_output, tuple):
                logits = model_output[0]
                # Extract additional features if available for MI regularization
                features = model_output[1] if len(model_output) > 1 else None
            else:
                logits = model_output
                features = None

            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(logits, sub_labels, reduction="none")

            # Compute ECT loss using f-divergence
            ect_loss = ect_loss_fn.compute_ect_loss(logits, dist_util.dev())
            if isinstance(ect_loss, th.Tensor) and ect_loss.dim() == 0:
                ect_loss = ect_loss.unsqueeze(0).expand(ce_loss.size(0))
            elif isinstance(ect_loss, th.Tensor) and ect_loss.size(0) != ce_loss.size(0):
                ect_loss = ect_loss.mean().unsqueeze(0).expand(ce_loss.size(0))

            # Compute mutual information regularization (if enabled)
            mi_reg = th.tensor(0.0, device=dist_util.dev())
            if args.mi_reg_weight > 0 and features is not None:
                mi_reg = ect_loss_fn.compute_mutual_info_regularization(logits, features)
                if isinstance(mi_reg, th.Tensor) and mi_reg.dim() == 0:
                    mi_reg = mi_reg.unsqueeze(0).expand(ce_loss.size(0))

            # Total loss
            total_loss = (ce_loss + 
                         args.ect_weight * ect_loss + 
                         args.mi_reg_weight * mi_reg)

            # Logging
            losses = {}
            losses[f"{prefix}_ce_loss"] = ce_loss.detach()
            losses[f"{prefix}_ect_loss"] = ect_loss.detach() if isinstance(ect_loss, th.Tensor) else th.tensor(ect_loss)
            losses[f"{prefix}_mi_reg"] = mi_reg.detach() if isinstance(mi_reg, th.Tensor) else th.tensor(mi_reg)
            losses[f"{prefix}_total_loss"] = total_loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(logits, sub_labels, k=1, reduction="none")
            losses[f"{prefix}_acc@5"] = compute_top_k(logits, sub_labels, k=5, reduction="none")
            
            # Compute entropy for monitoring
            probs = F.softmax(logits, dim=-1)
            entropy = ect_loss_fn.compute_entropy(probs)
            losses[f"{prefix}_entropy"] = entropy.detach()

            log_loss_dict(diffusion, sub_t, losses)
            del losses

            total_loss = total_loss.mean()
            if total_loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(total_loss * len(sub_batch) / len(batch))

    # Training loop
    for step in tqdm(range(args.iterations - resume_step)):
        logger.logkv("step", step + resume_step)
        logger.logkv("samples", (step + resume_step + 1) * args.batch_size * dist.get_world_size())
        
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
        
        if (step and dist.get_rank() == 0 and not (step + resume_step) % args.save_interval):
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
            yield tuple(x[i: i + microbatch] if x is not None else None for x in args)


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
        log_dir="",
        dataset_type='imagenet1000',
        
        # ECT and f-divergence parameters
        divergence_type='KL',  # Options: KL, JS, Hellinger, TV, X2, etc.
        entropy_type='renyi',  # Options: shannon, renyi, tsallis, min, collision
        alpha=2.0,  # Parameter for Rényi/Tsallis entropy and alpha-divergences
        ect_weight=0.1,  # Weight for ECT loss (η in the paper)
        mi_reg_weight=0.01,  # Weight for MI regularization (ζ in the paper)
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
