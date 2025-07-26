"""
Image Fusion Evaluation Script

This script provides functions to evaluate fusion results and save fused images
from different models to their respective folders.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def save_fusion_result(fused_img, img1_path, img2_path, model_name):
    """
    Save a fusion result to the appropriate folder
    
    Args:
        fused_img: The fused image as a numpy array (normalized to [0,1])
        img1_path: Path to the first source image
        img2_path: Path to the second source image
        model_name: Name of the fusion model (LRD, NSST_PAPCNN, etc.)
    
    Returns:
        Path to the saved image
    """
    # Create output directory if it doesn't exist
    output_dir = f"fused_images/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract image names for the output filename
    img1_name = os.path.basename(img1_path).split('.')[0]
    img2_name = os.path.basename(img2_path).split('.')[0]
    output_name = f"{img1_name}_{img2_name}_fused.png"
    output_path = os.path.join(output_dir, output_name)
    
    # Convert to uint8 and save
    fused_img_uint8 = (fused_img * 255).astype(np.uint8)
    cv2.imwrite(output_path, fused_img_uint8)
    
    return output_path

def evaluate_fusion(fused_img, img1, img2):
    """
    Evaluate fusion quality using common metrics
    
    Args:
        fused_img: The fused image
        img1: First source image
        img2: Second source image
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate PSNR
    psnr1 = psnr(img1, fused_img)
    psnr2 = psnr(img2, fused_img)
    avg_psnr = (psnr1 + psnr2) / 2
    
    # Calculate SSIM
    ssim1 = ssim(img1, fused_img, data_range=1.0)
    ssim2 = ssim(img2, fused_img, data_range=1.0)
    avg_ssim = (ssim1 + ssim2) / 2
    
    # Return metrics
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'psnr1': psnr1,
        'psnr2': psnr2,
        'ssim1': ssim1,
        'ssim2': ssim2
    }

def compare_models(img1_path, img2_path, model_results):
    """
    Compare fusion results from different models
    
    Args:
        img1_path: Path to first source image
        img2_path: Path to second source image
        model_results: Dictionary mapping model names to fused images
    """
    # Load source images for display
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) / 255.0
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) / 255.0
    
    # Calculate metrics for each model
    metrics = {}
    for model_name, fused_img in model_results.items():
        metrics[model_name] = evaluate_fusion(fused_img, img1, img2)
    
    # Create a visualization
    n_models = len(model_results)
    plt.figure(figsize=(12, 4 + 3*((n_models+2)//3)))
    
    # Display source images
    plt.subplot(1 + n_models//3, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(os.path.basename(img1_path))
    plt.axis('off')
    
    plt.subplot(1 + n_models//3, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(os.path.basename(img2_path))
    plt.axis('off')
    
    # Display fusion results
    for i, (model_name, fused_img) in enumerate(model_results.items()):
        plt.subplot(1 + n_models//3, 3, i+3)
        plt.imshow(fused_img, cmap='gray')
        met = metrics[model_name]
        plt.title(f"{model_name}\nPSNR: {met['psnr']:.2f}dB, SSIM: {met['ssim']:.4f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fused_images/comparison.png', dpi=300)
    plt.show()
    
    # Print metrics table
    print(f"{'Model':<15} {'PSNR (dB)':<10} {'SSIM':<10}")
    print("-" * 40)
    for model_name, met in metrics.items():
        print(f"{model_name:<15} {met['psnr']:<10.2f} {met['ssim']:<10.4f}")
    
    return metrics

def process_all_pairs(image_pairs, fusion_functions, max_pairs=None):
    """
    Process multiple image pairs with multiple fusion methods
    
    Args:
        image_pairs: List of tuples containing image pair paths
        fusion_functions: Dictionary mapping model names to fusion functions
        max_pairs: Maximum number of pairs to process (None for all)
    """
    # Limit number of pairs if specified
    if max_pairs is not None:
        pairs_to_process = image_pairs[:min(max_pairs, len(image_pairs))]
    else:
        pairs_to_process = image_pairs
    
    # Initialize results storage
    all_results = {model: [] for model in fusion_functions.keys()}
    
    # Process each pair
    for idx, (img1_path, img2_path) in enumerate(tqdm(pairs_to_process, desc="Processing")):
        # Load images once
        img1, img2 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) / 255.0, cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # Process with each fusion method
        for model_name, fusion_func in fusion_functions.items():
            # Apply fusion
            start_time = time.time()
            fused_img = fusion_func(img1, img2)
            execution_time = time.time() - start_time
            
            # Evaluate fusion
            metrics = evaluate_fusion(fused_img, img1, img2)
            metrics['time'] = execution_time
            
            # Save result
            output_path = save_fusion_result(fused_img, img1_path, img2_path, model_name)
            
            # Store metrics
            all_results[model_name].append({
                'img1': os.path.basename(img1_path),
                'img2': os.path.basename(img2_path),
                'output': output_path,
                **metrics
            })
    
    # Calculate average metrics for each model
    summary = {}
    for model_name, results in all_results.items():
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        
        summary[model_name] = {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_time': avg_time,
            'num_images': len(results)
        }
        
        print(f"\n{model_name} Summary:")
        print(f"Processed {len(results)} image pairs")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average execution time: {avg_time:.3f} seconds per image pair")
    
    return all_results, summary

if __name__ == "__main__":
    print("Run this module from your Jupyter notebooks to use these functions.")
