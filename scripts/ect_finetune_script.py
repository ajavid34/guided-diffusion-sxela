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
import os
import sys

# Add the guided-diffusion directory to the path
sys.path.append('/content/guided-diffusion-sxela')

from guided_diffusion.script_util import create_classifier_and_diffusion, classifier_and_diffusion_defaults

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

def add_noise_to_images(images, t, diffusion):
    """
    Add noise to images according to diffusion schedule at time t
    
    Args:
        images: clean images [B, C, H, W]
        t: time step (0 to 1000)
        diffusion: diffusion model for noise scheduling
    
    Returns:
        noisy_images: images with added noise
    """
    if t == 0:
        return images
    
    t_tensor = torch.full((images.shape[0],), t, device=images.device, dtype=torch.long)
    noisy_images = diffusion.q_sample(images, t_tensor)
    
    return noisy_images

def evaluate_classifier_at_timestep(classifier, diffusion, dataloader, device, timestep, debug=False):
    """
    Evaluate classifier on noisy images at specific timestep
    
    Args:
        classifier: trained classifier model (outputs 1000 ImageNet classes)
        diffusion: diffusion model for noise scheduling
        dataloader: ImageNette validation dataloader
        device: torch device
        timestep: diffusion timestep (0-1000)
        debug: whether to print debugging information
    
    Returns:
        ece_t: ECE at timestep t
        accuracy: top-1 accuracy at timestep t
        all_predictions: all model predictions
        all_labels: all ground truth labels
    """
    classifier.eval()
    all_predictions = []
    all_labels = []
    
    # Get ImageNet indices from dataset class names
    dataset_classes = dataloader.dataset.classes
    imagenet_indices = [int(cls) for cls in dataset_classes]
    
    batch_count = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating t={timestep}", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Debug: Check original image statistics
            if debug and batch_count == 0:
                print(f"\n=== DEBUG INFO for t={timestep} ===")
                print(f"Original images: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
            
            # Add noise according to timestep
            if timestep > 0:
                noisy_images = add_noise_to_images(images, timestep, diffusion)
            else:
                noisy_images = images
            
            # Debug: Check noisy image statistics
            if debug and batch_count == 0:
                print(f"Noisy images (t={timestep}): min={noisy_images.min():.3f}, max={noisy_images.max():.3f}, mean={noisy_images.mean():.3f}")
                if timestep > 0:
                    noise_diff = torch.abs(images - noisy_images).mean()
                    print(f"Noise difference: {noise_diff:.3f}")
            
            # Get classifier predictions (1000 classes)
            t_tensor = torch.full((images.shape[0],), timestep, device=device, dtype=torch.long)
            logits = classifier(noisy_images, timesteps=t_tensor)
            
            # Debug: Check logits statistics
            if debug and batch_count == 0:
                print(f"Logits: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
                print(f"Logits shape: {logits.shape}")
                # Check if classifier is actually using timestep
                logits_clean = classifier(images, timesteps=torch.zeros_like(t_tensor))
                logits_diff = torch.abs(logits - logits_clean).mean()
                print(f"Logits difference (noisy vs clean): {logits_diff:.6f}")
                print(f"Max confidence (clean): {torch.softmax(logits_clean, dim=1).max(dim=1)[0].mean():.3f}")
                print(f"Max confidence (noisy): {torch.softmax(logits, dim=1).max(dim=1)[0].mean():.3f}")
            
            # Extract logits for the ImageNet classes present in ImageNette
            relevant_logits = logits[:, imagenet_indices]  # Shape: [batch_size, 10]
            
            # Convert to probabilities
            predictions = torch.softmax(relevant_logits, dim=1)
            
            # Debug: Check predictions
            if debug and batch_count == 0:
                print(f"Predictions shape: {predictions.shape}")
                print(f"Max prediction confidence: {predictions.max(dim=1)[0].mean():.3f}")
                print(f"Predicted classes (first 10): {predictions.argmax(dim=1)[:10].cpu().numpy()}")
                print(f"True labels (first 10): {labels[:10].cpu().numpy()}")
                print("=== END DEBUG ===\n")
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            batch_count += 1
    
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
    ImageNette uses specific ImageNet class indices for 10 classes
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Try different possible dataset structures
    try:
        # First try val subdirectory
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)
    except:
        try:
            # Try test subdirectory  
            dataset = ImageFolder(root=f"{data_dir}/test", transform=transform)
        except:
            # Try root directory directly
            dataset = ImageFolder(root=data_dir, transform=transform)
    
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
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
    
    # Load baseline classifier using guided_diffusion.script_util
    print("Loading classifiers and diffusion model...")
    
    # Use the default arguments from classifier_and_diffusion_defaults
    from guided_diffusion.script_util import classifier_and_diffusion_defaults
    classifier_args = classifier_and_diffusion_defaults()
    
    # Override with your specific training configuration
    classifier_args.update({
        'image_size': 256,
        'classifier_attention_resolutions': '32,16,8',
        'classifier_depth': 2,
        'classifier_width': 128,
        'classifier_pool': 'attention',
        'classifier_resblock_updown': True,
        'classifier_use_scale_shift_norm': True,
        'classifier_use_fp16': True,
    })
    
    # Load baseline classifier and diffusion model
    print("Creating baseline classifier...")
    baseline_classifier, diffusion = create_classifier_and_diffusion(**classifier_args)
    print(f"Loading baseline checkpoint: {args.baseline_checkpoint}")
    baseline_state = torch.load(args.baseline_checkpoint, map_location=device)
    
    # Debug baseline checkpoint
    print(f"✅ Baseline checkpoint keys: {list(baseline_state.keys())[:10]}...")  # Show first 10 keys
    print(f"✅ Baseline checkpoint size: {len(baseline_state)} parameters")
    
    baseline_classifier.load_state_dict(baseline_state)
    baseline_classifier.to(device)
    
    # Convert to FP16 if needed (this is important for mixed precision models)
    if classifier_args['classifier_use_fp16']:
        baseline_classifier.convert_to_fp16()
    
    # Load ECT-trained classifier (same diffusion model)
    print("Creating ECT classifier...")
    ect_classifier, _ = create_classifier_and_diffusion(**classifier_args)
    print(f"Loading ECT checkpoint: {args.ect_checkpoint}")
    ect_state = torch.load(args.ect_checkpoint, map_location=device)
    
    # Debug ECT checkpoint
    print(f"✅ ECT checkpoint keys: {list(ect_state.keys())[:10]}...")  # Show first 10 keys
    print(f"✅ ECT checkpoint size: {len(ect_state)} parameters")
    
    ect_classifier.load_state_dict(ect_state)
    ect_classifier.to(device)
    
    # Convert to FP16 if needed
    if classifier_args['classifier_use_fp16']:
        ect_classifier.convert_to_fp16()
    
    # Load dataset
    print("Loading ImageNette dataset...")
    dataloader = load_imagenette_dataloader(args.data_dir, args.batch_size)
    
    # Get ImageNet mapping info
    dataset_classes = dataloader.dataset.classes
    imagenet_indices = [int(cls) for cls in dataset_classes]
    print(f"✅ ImageNette classes (local 0-9) map to ImageNet indices: {imagenet_indices}")
    
    # Quick test to verify setup
    print("\nVerifying dataset and classifier setup...")
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_images, sample_labels = sample_batch
        sample_images = sample_images.to(device)
        t_tensor = torch.zeros(sample_images.shape[0], device=device, dtype=torch.long)
        
        baseline_logits = baseline_classifier(sample_images, timesteps=t_tensor)
        print(f"✅ Baseline classifier output shape: {baseline_logits.shape} (expects [batch_size, 1000])")
        print(f"✅ Sample labels range: {sample_labels.min().item()} to {sample_labels.max().item()} (expects 0-9)")
        print(f"✅ Dataset classes: {len(dataloader.dataset.classes)} classes")
        
        ect_logits = ect_classifier(sample_images, timesteps=t_tensor)
        print(f"✅ ECT classifier output shape: {ect_logits.shape} (expects [batch_size, 1000])")
    
    print("\n" + "="*60)
    
    # Evaluate ECE_t for both classifiers
    baseline_results = {'timesteps': [], 'ece_t': [], 'accuracy': []}
    ect_results = {'timesteps': [], 'ece_t': [], 'accuracy': []}
    
    # Test with one timestep first to ensure everything works
    print(f"\nTesting evaluation with timestep 0...")
    try:
        test_ece, test_acc, _, _ = evaluate_classifier_at_timestep(
            baseline_classifier, diffusion, dataloader, device, 0, debug=True
        )
        print(f"✅ Test successful: ECE_t={test_ece:.4f}, Accuracy={test_acc:.4f}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("EVALUATING BASELINE CLASSIFIER (without ECT)")
    print("="*60)
    
    for i, t in enumerate(timesteps):
        try:
            # Add debug for a few key timesteps
            debug = (t in [0, 50, 150, 250])
            ece_t, acc, _, _ = evaluate_classifier_at_timestep(
                baseline_classifier, diffusion, dataloader, device, t, debug=debug
            )
            baseline_results['timesteps'].append(t)
            baseline_results['ece_t'].append(ece_t)
            baseline_results['accuracy'].append(acc)
            print(f"[{i+1:2}/{len(timesteps)}] t={t:3}: ECE_t={ece_t:.4f}, Accuracy={acc:.4f}")
        except Exception as e:
            print(f"❌ Error at timestep {t}: {e}")
            continue
    
    print("\n" + "="*60)
    print("EVALUATING ECT-TRAINED CLASSIFIER (with Tsallis α=1.5)")
    print("="*60)
    
    for i, t in enumerate(timesteps):
        try:
            # Add debug for a few key timesteps
            debug = (t in [0, 50, 150, 250])
            ece_t, acc, _, _ = evaluate_classifier_at_timestep(
                ect_classifier, diffusion, dataloader, device, t, debug=debug
            )
            ect_results['timesteps'].append(t)
            ect_results['ece_t'].append(ece_t)
            ect_results['accuracy'].append(acc)
            print(f"[{i+1:2}/{len(timesteps)}] t={t:3}: ECE_t={ece_t:.4f}, Accuracy={acc:.4f}")
        except Exception as e:
            print(f"❌ Error at timestep {t}: {e}")
            continue
    
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
