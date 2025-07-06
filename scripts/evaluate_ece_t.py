#!/usr/bin/env python3
"""
ECE_t Evaluation Script for Classifier-Guided Diffusion
Evaluates Expected Calibration Error at different time steps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import json

def compute_ece_t(predictions, labels, n_bins=10):
    """
    Compute Expected Calibration Error for given predictions
    
    Args:
        predictions: softmax probabilities [N, num_classes]
        labels: ground truth labels [N]
        n_bins: number of confidence bins
    
    Returns:
        ece: Expected Calibration Error
        bin_boundaries: confidence bin boundaries
        bin_lowers: lower bounds of bins
        bin_uppers: upper bounds of bins
        bin_accuracies: accuracy within each bin
        bin_confidences: average confidence within each bin
    """
    confidences = torch.max(predictions, dim=1)[0]
    accuracies = predictions.argmax(dim=1).eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = torch.zeros(1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_counts.append(in_bin.sum().item())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    return ece.item(), bin_accuracies, bin_confidences, bin_counts

def add_noise_to_images(images, t, noise_schedule):
    """
    Add noise to images according to diffusion schedule at time t
    
    Args:
        images: clean images [B, C, H, W]
        t: time step (0 to 1000)
        noise_schedule: noise schedule parameters
    
    Returns:
        noisy_images: images with added noise
    """
    # Linear noise schedule (as used in your training)
    beta_start = 0.0001
    beta_end = 0.02
    num_timesteps = 1000
    
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    alpha_t = alphas_cumprod[t]
    
    noise = torch.randn_like(images)
    noisy_images = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise
    
    return noisy_images

def evaluate_classifier_at_timestep(classifier, dataloader, device, timestep):
    """
    Evaluate classifier on noisy images at specific timestep
    
    Args:
        classifier: trained classifier model
        dataloader: ImageNette validation dataloader
        device: torch device
        timestep: diffusion timestep (0-1000)
    
    Returns:
        ece_t: ECE at timestep t
        accuracy: top-1 accuracy at timestep t
        all_predictions: all model predictions
        all_labels: all ground truth labels
    """
    classifier.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating t={timestep}"):
            images, labels = images.to(device), labels.to(device)
            
            # Add noise according to timestep
            if timestep > 0:
                noisy_images = add_noise_to_images(images, timestep, None)
            else:
                noisy_images = images
            
            # Get classifier predictions
            t_tensor = torch.full((images.shape[0],), timestep, device=device)
            logits = classifier(noisy_images, t_tensor)
            predictions = torch.softmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute ECE_t
    ece_t, _, _, _ = compute_ece_t(all_predictions, all_labels)
    
    # Compute accuracy
    accuracy = (all_predictions.argmax(dim=1) == all_labels).float().mean().item()
    
    return ece_t, accuracy, all_predictions, all_labels

def load_imagenette_dataloader(data_dir, batch_size=32):
    """
    Load ImageNette validation dataset
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Adjust this based on your dataset structure
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline classifier (without ECT)')
    parser.add_argument('--ect_checkpoint', type=str, required=True,
                       help='Path to ECT-trained classifier')
    parser.add_argument('--data_dir', type=str, default='./imagenet_subset',
                       help='Path to ImageNette dataset')
    parser.add_argument('--output_dir', type=str, default='./ece_results',
                       help='Directory to save results')
    parser.add_argument('--timesteps', type=str, default='0,10,20,30,40,50,100,150,200,250',
                       help='Comma-separated timesteps to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse timesteps
    timesteps = [int(t) for t in args.timesteps.split(',')]
    
    # Load classifiers
    print("Loading classifiers...")
    
    # Load baseline classifier (you'll need to adapt this to your classifier architecture)
    from scripts.classifier_train_multientry import create_classifier  # Adjust import
    baseline_classifier = create_classifier(
        image_size=256,
        classifier_attention_resolutions="32,16,8",
        classifier_depth=2,
        classifier_width=128,
        classifier_pool="attention",
        classifier_resblock_updown=True,
        classifier_use_scale_shift_norm=True,
        classifier_use_fp16=True,
        num_classes=10  # ImageNette has 10 classes
    )
    baseline_classifier.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
    baseline_classifier.to(device)
    
    # Load ECT-trained classifier
    ect_classifier = create_classifier(
        image_size=256,
        classifier_attention_resolutions="32,16,8",
        classifier_depth=2,
        classifier_width=128,
        classifier_pool="attention",
        classifier_resblock_updown=True,
        classifier_use_scale_shift_norm=True,
        classifier_use_fp16=True,
        num_classes=10
    )
    ect_classifier.load_state_dict(torch.load(args.ect_checkpoint, map_location=device))
    ect_classifier.to(device)
    
    # Load dataset
    print("Loading ImageNette dataset...")
    dataloader = load_imagenette_dataloader(args.data_dir, args.batch_size)
    
    # Evaluate ECE_t for both classifiers
    baseline_results = {'timesteps': [], 'ece_t': [], 'accuracy': []}
    ect_results = {'timesteps': [], 'ece_t': [], 'accuracy': []}
    
    print("Evaluating baseline classifier...")
    for t in timesteps:
        ece_t, acc, _, _ = evaluate_classifier_at_timestep(
            baseline_classifier, dataloader, device, t
        )
        baseline_results['timesteps'].append(t)
        baseline_results['ece_t'].append(ece_t)
        baseline_results['accuracy'].append(acc)
        print(f"Baseline - t={t}: ECE_t={ece_t:.4f}, Accuracy={acc:.4f}")
    
    print("Evaluating ECT-trained classifier...")
    for t in timesteps:
        ece_t, acc, _, _ = evaluate_classifier_at_timestep(
            ect_classifier, dataloader, device, t
        )
        ect_results['timesteps'].append(t)
        ect_results['ece_t'].append(ece_t)
        ect_results['accuracy'].append(acc)
        print(f"ECT - t={t}: ECE_t={ece_t:.4f}, Accuracy={acc:.4f}")
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/baseline_results.json", 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    with open(f"{args.output_dir}/ect_results.json", 'w') as f:
        json.dump(ect_results, f, indent=2)
    
    # Plot ECE_t curves (similar to Figure 1)
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_results['timesteps'], baseline_results['ece_t'], 
             'b-o', label='Baseline (without ECT)', linewidth=2, markersize=6)
    plt.plot(ect_results['timesteps'], ect_results['ece_t'], 
             'r-s', label='With ECT (Tsallis α=1.5)', linewidth=2, markersize=6)
    
    plt.xlabel('Diffusion Timestep t', fontsize=12)
    plt.ylabel('ECE_t (Expected Calibration Error)', fontsize=12)
    plt.title('ECE_t Performance: With vs Without Entropy Constraint Training', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/ece_t_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_results['timesteps'], baseline_results['accuracy'], 
             'b-o', label='Baseline (without ECT)', linewidth=2, markersize=6)
    plt.plot(ect_results['timesteps'], ect_results['accuracy'], 
             'r-s', label='With ECT (Tsallis α=1.5)', linewidth=2, markersize=6)
    
    plt.xlabel('Diffusion Timestep t', fontsize=12)
    plt.ylabel('Top-1 Accuracy', fontsize=12)
    plt.title('Accuracy Performance: With vs Without Entropy Constraint Training', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to {args.output_dir}/")
    print("ECE_t evaluation completed!")

if __name__ == "__main__":
    main()
