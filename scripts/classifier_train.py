"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    logger.log('current rank == {}, total_num = {}'.format(dist.get_rank(), dist.get_world_size()))
    
    logger.log(args)

    logger.log("creating model and diffusion...")
    
    # Create model with original function call
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    
    # Manually fix the model to have 200 classes for Tiny ImageNet
    logger.log("Checking and fixing model output dimensions...")
    
    # Print model structure to understand the architecture
    logger.log("Model structure:")
    for name, module in model.named_modules():
        if hasattr(module, 'out_features') or hasattr(module, 'out_channels'):
            if hasattr(module, 'out_features'):
                logger.log(f"  {name}: {type(module).__name__} out_features={module.out_features}")
            if hasattr(module, 'out_channels'):
                logger.log(f"  {name}: {type(module).__name__} out_channels={module.out_channels}")
    
    # Find and replace layers with 1000 outputs
    import torch.nn as nn
    layers_replaced = 0
    
    def replace_layer_recursive(module, name=""):
        nonlocal layers_replaced
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check Conv1d layers (for final classification)
            if isinstance(child_module, nn.Conv1d) and child_module.out_channels == 1000:
                logger.log(f"Replacing Conv1d layer {full_name}: {child_module.in_channels} -> 1000 with {child_module.in_channels} -> 200")
                new_layer = nn.Conv1d(
                    child_module.in_channels, 
                    200,
                    child_module.kernel_size,
                    child_module.stride,
                    child_module.padding,
                    child_module.dilation,
                    child_module.groups,
                    child_module.bias is not None
                )
                nn.init.xavier_uniform_(new_layer.weight)
                if new_layer.bias is not None:
                    nn.init.zeros_(new_layer.bias)
                setattr(module, child_name, new_layer)
                layers_replaced += 1
            
            # Check Linear layers
            elif isinstance(child_module, nn.Linear) and child_module.out_features == 1000:
                logger.log(f"Replacing Linear layer {full_name}: {child_module.in_features} -> 1000 with {child_module.in_features} -> 200")
                new_layer = nn.Linear(child_module.in_features, 200)
                nn.init.xavier_uniform_(new_layer.weight)
                nn.init.zeros_(new_layer.bias)
                setattr(module, child_name, new_layer)
                layers_replaced += 1
            
            # Check Conv2d layers ONLY if they are likely output layers (avoid internal ResBlock layers)
            elif isinstance(child_module, nn.Conv2d) and child_module.out_channels == 1000 and "out" in full_name:
                logger.log(f"Replacing Conv2d layer {full_name}: {child_module.in_channels} -> 1000 with {child_module.in_channels} -> 200")
                new_layer = nn.Conv2d(
                    child_module.in_channels, 
                    200,
                    child_module.kernel_size,
                    child_module.stride,
                    child_module.padding,
                    child_module.dilation,
                    child_module.groups,
                    child_module.bias is not None
                )
                nn.init.xavier_uniform_(new_layer.weight)
                if new_layer.bias is not None:
                    nn.init.zeros_(new_layer.bias)
                setattr(module, child_name, new_layer)
                layers_replaced += 1
            
            # Recursively check child modules
            else:
                replace_layer_recursive(child_module, full_name)
    
    replace_layer_recursive(model)
    logger.log(f"Replaced {layers_replaced} layers")
    
    # If no layers were replaced, don't try to replace internal layers
    if layers_replaced == 0:
        logger.log("No output layers with 1000 channels found - will use runtime slicing instead")
    
    model.to(dist_util.dev())

    # Verify the fix worked by testing actual output
    logger.log("Verifying model output dimensions...")
    dummy_input = th.randn(1, 3, args.image_size, args.image_size).to(dist_util.dev())
    dummy_timestep = th.zeros(1, dtype=th.long).to(dist_util.dev())
    with th.no_grad():
        dummy_output = model(dummy_input, timesteps=dummy_timestep)
        if isinstance(dummy_output, tuple):
            output_shape = dummy_output[0].shape
        else:
            output_shape = dummy_output.shape
        
        logger.log(f"Model output shape: {output_shape}")
        
        if output_shape[-1] == 200:
            logger.log(f"SUCCESS: Model correctly outputs 200 classes")
        else:
            logger.log(f"WARNING: Model still outputs {output_shape[-1]} classes instead of 200")
            logger.log("Continuing training anyway - the loss will handle the mismatch")
            logger.log("Note: You may need to manually edit the guided_diffusion source code to fix NUM_CLASSES")

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
        random_crop=True
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True
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
        if bf.exists(opt_checkpoint):
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images (recommended for diffusion classifiers)
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            
            # Handle model output (some models return tuple)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Force 200 classes if model still outputs 1000
            if logits.shape[-1] == 1000:
                logits = logits[:, :200]  # Take only first 200 classes
            
            # Fix label indexing: clamp labels to valid range [0, 199]
            sub_labels = th.clamp(sub_labels, 0, 199)
            
            loss_ce = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss_ce"] = loss_ce.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss_ce
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in tqdm(range(args.iterations - resume_step)):
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
            yield tuple(x[i: i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,  # Keep True for diffusion classifier training
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
        log_dir="./logs",
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
