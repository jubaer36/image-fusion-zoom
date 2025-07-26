"""
This module provides common model definitions and utility functions for image fusion.

It allows importing models from each notebook to make comparisons easier.
"""

import os
import numpy as np
import cv2
from scipy import ndimage
import pywt


# ------------ LRD Model Functions ------------

def gaussian_pyramid(img, levels=4):
    """
    Build a Gaussian pyramid
    
    Args:
        img: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of Gaussian pyramid levels
    """
    # Initialize pyramid with original image
    g_pyr = [img.copy()]
    
    # Build Gaussian pyramid
    for i in range(1, levels):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(g_pyr[i-1], (5, 5), 0)
        # Downsample
        downsampled = blurred[::2, ::2]
        g_pyr.append(downsampled)
    
    return g_pyr

def laplacian_pyramid(img, levels=4):
    """
    Build a Laplacian pyramid
    
    Args:
        img: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of Laplacian pyramid levels
    """
    # Build Gaussian pyramid
    g_pyr = gaussian_pyramid(img, levels)
    
    # Initialize Laplacian pyramid
    l_pyr = []
    
    # Build Laplacian pyramid
    for i in range(levels - 1):
        # Get current and next Gaussian pyramid level
        curr_level = g_pyr[i]
        next_level = g_pyr[i + 1]
        
        # Upsample next level
        next_level_upsampled = np.zeros((curr_level.shape[0], curr_level.shape[1]), dtype=np.float32)
        next_level_upsampled[::2, ::2] = next_level
        
        # Apply Gaussian blur to upsampled image to smooth it
        next_level_upsampled = cv2.GaussianBlur(next_level_upsampled, (5, 5), 0)
        
        # Calculate Laplacian as difference
        laplacian = curr_level - next_level_upsampled
        l_pyr.append(laplacian)
    
    # Append the smallest Gaussian level as the last element
    l_pyr.append(g_pyr[-1])
    
    return l_pyr

def reconstruct_from_laplacian(l_pyr):
    """
    Reconstruct image from Laplacian pyramid
    
    Args:
        l_pyr: Laplacian pyramid
        
    Returns:
        Reconstructed image
    """
    # Start with the smallest level
    reconstructed = l_pyr[-1]
    
    # Iterate from second-last to first level
    for i in range(len(l_pyr) - 2, -1, -1):
        # Get current Laplacian level
        curr_laplacian = l_pyr[i]
        
        # Upsample reconstructed image
        reconstructed_upsampled = np.zeros(curr_laplacian.shape, dtype=np.float32)
        reconstructed_upsampled[::2, ::2] = reconstructed
        
        # Apply Gaussian blur to upsampled image
        reconstructed_upsampled = cv2.GaussianBlur(reconstructed_upsampled, (5, 5), 0)
        
        # Add Laplacian level
        reconstructed = reconstructed_upsampled + curr_laplacian
    
    return reconstructed

def calculate_saliency(img, kernel_size=3):
    """
    Calculate visual saliency map
    
    Args:
        img: Input image
        kernel_size: Size of kernel for filtering
        
    Returns:
        Saliency map
    """
    # Calculate gradient in x and y directions
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)
    
    # Calculate gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    saliency = (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag) + 1e-10)
    
    return saliency

def laplacian_re_decomposition(l_pyr, levels=4):
    """
    Perform Laplacian Re-Decomposition (LRD)
    
    Args:
        l_pyr: Laplacian pyramid
        levels: Number of decomposition levels
        
    Returns:
        Re-decomposed Laplacian pyramid
    """
    # Initialize re-decomposed pyramid
    re_l_pyr = []
    
    # Process each level of the Laplacian pyramid
    for i, level in enumerate(l_pyr):
        if i < levels - 1:
            # Calculate saliency for intermediate levels
            saliency = calculate_saliency(level)
            
            # Apply saliency weighting to enhance details
            re_level = level * (1 + saliency)
            re_l_pyr.append(re_level)
        else:
            # Keep the smallest level unchanged
            re_l_pyr.append(level)
    
    return re_l_pyr

def fuse_laplacian_levels(l_pyr1, l_pyr2):
    """
    Fuse two Laplacian pyramids
    
    Args:
        l_pyr1: First Laplacian pyramid
        l_pyr2: Second Laplacian pyramid
        
    Returns:
        Fused Laplacian pyramid
    """
    # Initialize fused pyramid
    fused_l_pyr = []
    
    # Process each level
    for i in range(len(l_pyr1)):
        level1 = l_pyr1[i]
        level2 = l_pyr2[i]
        
        if i == len(l_pyr1) - 1:
            # For the smallest level (residual), use average
            fused_level = (level1 + level2) / 2
        else:
            # For detail levels, choose maximum absolute values
            abs_level1 = np.abs(level1)
            abs_level2 = np.abs(level2)
            
            # Create selection mask
            mask = (abs_level1 >= abs_level2).astype(np.float32)
            
            # Apply mask to select coefficients with higher absolute values
            fused_level = mask * level1 + (1 - mask) * level2
        
        fused_l_pyr.append(fused_level)
    
    return fused_l_pyr

def lrd_fusion(img1, img2, levels=4):
    """
    Complete Laplacian Re-Decomposition (LRD) fusion method
    
    Args:
        img1: First input image
        img2: Second input image
        levels: Number of pyramid levels
        
    Returns:
        Fused image
    """
    # Convert images to float32 if they're not already
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Step 1: Build Laplacian pyramids
    l_pyr1 = laplacian_pyramid(img1, levels)
    l_pyr2 = laplacian_pyramid(img2, levels)
    
    # Step 2: Laplacian Re-Decomposition
    re_l_pyr1 = laplacian_re_decomposition(l_pyr1, levels)
    re_l_pyr2 = laplacian_re_decomposition(l_pyr2, levels)
    
    # Step 3: Fusion of Laplacian pyramids
    fused_l_pyr = fuse_laplacian_levels(re_l_pyr1, re_l_pyr2)
    
    # Step 4: Reconstruct from fused Laplacian pyramid
    fused_img = reconstruct_from_laplacian(fused_l_pyr)
    
    # Clip values to valid range [0, 1]
    fused_img = np.clip(fused_img, 0, 1)
    
    return fused_img


# ------------ NSST-PAPCNN Model Functions ------------

def nsst_decomposition(img, levels=4, wavelet='db1'):
    """
    Decompose image using wavelet transform as an approximation of NSST
    
    Args:
        img: Input image
        levels: Number of decomposition levels
        wavelet: Wavelet type
        
    Returns:
        Dictionary containing coefficients at each level and orientation
    """
    # Initialize result dictionary
    decomposition = {}
    
    # Perform stationary wavelet transform (undecimated - similar to non-subsampled transform)
    coeffs = pywt.swt2(img, wavelet, level=levels)
    
    # Extract coefficients
    decomposition['lowpass'] = coeffs[0][0]  # Approximation coefficients from the highest level
    
    # Store highpass coefficients for each level
    decomposition['highpass'] = []
    
    for i in range(levels):
        # Get horizontal, vertical and diagonal details
        level_coeffs = {
            'horizontal': coeffs[i][1][0],
            'vertical': coeffs[i][1][1],
            'diagonal': coeffs[i][1][2]
        }
        decomposition['highpass'].append(level_coeffs)
    
    return decomposition

def nsst_reconstruction(decomposition, wavelet='db1'):
    """
    Reconstruct image from wavelet coefficients
    
    Args:
        decomposition: Dictionary containing wavelet coefficients
        wavelet: Wavelet type
        
    Returns:
        Reconstructed image
    """
    # Get number of levels
    levels = len(decomposition['highpass'])
    
    # Prepare coefficients for reconstruction
    coeffs = []
    for i in range(levels):
        level_idx = levels - 1 - i
        if i == 0:
            # For the first level, use the lowpass coefficients
            cA = decomposition['lowpass']
        else:
            # For subsequent levels, approximate coefficients are already included
            cA = None
            
        # Get highpass coefficients
        level_highpass = decomposition['highpass'][level_idx]
        cH = level_highpass['horizontal']
        cV = level_highpass['vertical']
        cD = level_highpass['diagonal']
        
        # Combine coefficients
        if i == 0:
            coeffs.append((cA, (cH, cV, cD)))
        else:
            coeffs.append((None, (cH, cV, cD)))
    
    # Perform inverse stationary wavelet transform
    reconstructed = pywt.iswt2(coeffs, wavelet)
    
    return reconstructed

def calculate_spatial_frequency(img):
    """
    Calculate spatial frequency of an image
    
    Args:
        img: Input image
        
    Returns:
        Spatial frequency value
    """
    # Calculate row frequency
    rf = np.sqrt(np.sum(np.diff(img, axis=1) ** 2) / (img.shape[0] * img.shape[1]))
    
    # Calculate column frequency
    cf = np.sqrt(np.sum(np.diff(img, axis=0) ** 2) / (img.shape[0] * img.shape[1]))
    
    # Calculate spatial frequency
    sf = np.sqrt(rf ** 2 + cf ** 2)
    
    return sf

def calculate_weight_exponent(img):
    """
    Calculate weight exponent based on spatial frequency
    
    Args:
        img: Input image
        
    Returns:
        Weight exponent for PAPCNN
    """
    sf = calculate_spatial_frequency(img)
    # Based on the paper, weight exponent is inversely proportional to spatial frequency
    weight_exp = np.exp(-sf)
    return weight_exp

def papcnn_fusion(img1, img2, iterations=10):
    """
    Parameter-Adaptive Pulse Coupled Neural Network (PAPCNN) for image fusion
    
    Args:
        img1: First input image
        img2: Second input image
        iterations: Number of PAPCNN iterations
        
    Returns:
        Fusion weights for both images
    """
    # Calculate weight exponents based on spatial frequency
    beta1 = calculate_weight_exponent(img1)
    beta2 = calculate_weight_exponent(img2)
    
    # Get image dimensions
    h, w = img1.shape
    
    # Initialize PAPCNN variables
    # F: Feeding input
    F1 = img1.copy()
    F2 = img2.copy()
    
    # L: Linking
    L1 = np.zeros((h, w))
    L2 = np.zeros((h, w))
    
    # U: Internal activity
    U1 = F1
    U2 = F2
    
    # Y: Pulse output
    Y1 = np.zeros((h, w))
    Y2 = np.zeros((h, w))
    
    # T: Dynamic threshold
    T1 = np.ones((h, w))
    T2 = np.ones((h, w))
    
    # Linking strength and decay parameters
    alpha_L = 0.1
    alpha_T = 0.2
    V_T = 20.0
    
    # Kernel for linking
    kernel = np.array([[0.1, 0.1, 0.1], 
                      [0.1, 0.0, 0.1], 
                      [0.1, 0.1, 0.1]])
    
    # Output accumulation
    Y1_sum = np.zeros((h, w))
    Y2_sum = np.zeros((h, w))
    
    # Iterate PAPCNN
    for n in range(iterations):
        # Update linking
        L1 = alpha_L * L1 + ndimage.convolve(Y1, kernel, mode='constant')
        L2 = alpha_L * L2 + ndimage.convolve(Y2, kernel, mode='constant')
        
        # Update internal activity
        U1 = F1 * (1 + beta1 * L1)
        U2 = F2 * (1 + beta2 * L2)
        
        # Update pulse output
        Y1 = (U1 > T1).astype(np.float64)
        Y2 = (U2 > T2).astype(np.float64)
        
        # Update threshold
        T1 = alpha_T * T1 + V_T * Y1
        T2 = alpha_T * T2 + V_T * Y2
        
        # Accumulate output
        Y1_sum += Y1 * (n + 1)
        Y2_sum += Y2 * (n + 1)
    
    # Calculate fusion weights
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    weight1 = Y1_sum / (Y1_sum + Y2_sum + epsilon)
    weight2 = Y2_sum / (Y1_sum + Y2_sum + epsilon)
    
    return weight1, weight2

def lowpass_fusion(lowpass1, lowpass2):
    """
    Fusion rule for low-frequency coefficients using average method
    
    Args:
        lowpass1: Low-frequency coefficients of first image
        lowpass2: Low-frequency coefficients of second image
        
    Returns:
        Fused low-frequency coefficients
    """
    # Simple average fusion for lowpass components
    return (lowpass1 + lowpass2) / 2

def highpass_fusion(highpass1, highpass2):
    """
    Fusion rule for high-frequency coefficients using PAPCNN
    
    Args:
        highpass1: High-frequency coefficients of first image
        highpass2: High-frequency coefficients of second image
        
    Returns:
        Fused high-frequency coefficients
    """
    # Apply PAPCNN to determine weights
    weight1, weight2 = papcnn_fusion(np.abs(highpass1), np.abs(highpass2))
    
    # Apply weighted fusion
    fused_highpass = weight1 * highpass1 + weight2 * highpass2
    
    return fused_highpass

def nsst_papcnn_fusion(img1, img2, levels=3, wavelet='db1'):
    """
    Complete NSST-PAPCNN fusion method
    
    Args:
        img1: First input image
        img2: Second input image
        levels: Number of decomposition levels
        wavelet: Wavelet type
        
    Returns:
        Fused image
    """
    # Decompose images
    decomp1 = nsst_decomposition(img1, levels, wavelet)
    decomp2 = nsst_decomposition(img2, levels, wavelet)
    
    # Initialize fused decomposition
    fused_decomp = {'lowpass': None, 'highpass': []}
    
    # Fuse lowpass coefficients
    fused_decomp['lowpass'] = lowpass_fusion(decomp1['lowpass'], decomp2['lowpass'])
    
    # Fuse highpass coefficients for each level
    for level in range(levels):
        level_fused = {}
        
        # Fuse each orientation (horizontal, vertical, diagonal)
        for orientation in ['horizontal', 'vertical', 'diagonal']:
            highpass1 = decomp1['highpass'][level][orientation]
            highpass2 = decomp2['highpass'][level][orientation]
            level_fused[orientation] = highpass_fusion(highpass1, highpass2)
        
        fused_decomp['highpass'].append(level_fused)
    
    # Reconstruct fused image
    fused_img = nsst_reconstruction(fused_decomp, wavelet)
    
    # Ensure pixel values are in valid range [0, 1]
    fused_img = np.clip(fused_img, 0, 1)
    
    return fused_img


# ------------ U2Fusion Model Functions ------------

def guided_filter(p, I, r=5, eps=0.1):
    """
    Edge-preserving smoothing filter used in U2Fusion
    
    Args:
        p: Input image to be filtered
        I: Guidance image (can be the same as p)
        r: Filter radius
        eps: Regularization parameter
        
    Returns:
        Filtered image
    """
    # Convert inputs to float32
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    # Get dimensions
    h, w = I.shape
    
    # Step 1: Mean filter
    mean_I = cv2.boxFilter(I, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(p, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Correlation of I and p
    corr_Ip = cv2.boxFilter(I * p, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Auto-correlation of I
    corr_II = cv2.boxFilter(I * I, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Step 2: Linear coefficients
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    # Compute a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Step 3: Mean filter for a and b
    mean_a = cv2.boxFilter(a, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Step 4: Output
    q = mean_a * I + mean_b
    
    return q

def calculate_saliency_u2(img, kernel_size=3):
    """
    Calculate visual saliency map for U2Fusion
    
    Args:
        img: Input image
        kernel_size: Size of kernel for filtering
        
    Returns:
        Saliency map
    """
    # Convert to float32
    img = img.astype(np.float32)
    
    # Apply Gaussian filter to get global mean
    global_mean = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Calculate saliency as squared difference between image and its mean
    saliency = (img - global_mean) ** 2
    
    # Apply guided filter to smooth saliency map while preserving edges
    saliency = guided_filter(saliency, img, r=5, eps=0.1)
    
    # Normalize to [0, 1]
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-10)
    
    return saliency

def decompose_image(img, r=5, eps=0.1):
    """
    Decompose image into base layer and detail layer using guided filter
    
    Args:
        img: Input image
        r: Filter radius
        eps: Regularization parameter
        
    Returns:
        Base layer and detail layer
    """
    # Base layer (structure) via guided filter
    base_layer = guided_filter(img, img, r, eps)
    
    # Detail layer (texture) via subtraction
    detail_layer = img - base_layer
    
    return base_layer, detail_layer

def soft_threshold(x, T):
    """
    Soft thresholding function
    
    Args:
        x: Input value
        T: Threshold value
        
    Returns:
        Soft thresholded value
    """
    return np.sign(x) * np.maximum(np.abs(x) - T, 0)

def fusion_unified_framework(base1, base2, detail1, detail2, saliency1, saliency2):
    """
    Fusion using the unified framework (U2Fusion)
    
    Args:
        base1: Base layer of first image
        base2: Base layer of second image
        detail1: Detail layer of first image
        detail2: Detail layer of second image
        saliency1: Saliency map of first image
        saliency2: Saliency map of second image
        
    Returns:
        Fused base layer and fused detail layer
    """
    # Normalize saliency maps to create weights
    weight_sum = saliency1 + saliency2 + 1e-10
    weight1 = saliency1 / weight_sum
    weight2 = saliency2 / weight_sum
    
    # Fuse base layers using weighted average
    fused_base = weight1 * base1 + weight2 * base2
    
    # L1-norm for detail layers
    abs_detail1 = np.abs(detail1)
    abs_detail2 = np.abs(detail2)
    
    # Adaptive threshold
    T = 0.1 * np.mean(abs_detail1 + abs_detail2)
    
    # Soft thresholding
    soft_detail1 = soft_threshold(detail1, T)
    soft_detail2 = soft_threshold(detail2, T)
    
    # Choose maximum absolute value for detail layers (L1-norm)
    detail_mask = (abs_detail1 >= abs_detail2).astype(np.float32)
    fused_detail = detail_mask * soft_detail1 + (1 - detail_mask) * soft_detail2
    
    return fused_base, fused_detail

def u2fusion(img1, img2, r=5, eps=0.1):
    """
    Complete U2Fusion method
    
    Args:
        img1: First input image
        img2: Second input image
        r: Filter radius for guided filter
        eps: Regularization parameter for guided filter
        
    Returns:
        Fused image
    """
    # Calculate saliency maps
    saliency1 = calculate_saliency_u2(img1)
    saliency2 = calculate_saliency_u2(img2)
    
    # Decompose images into base and detail layers
    base1, detail1 = decompose_image(img1, r, eps)
    base2, detail2 = decompose_image(img2, r, eps)
    
    # Fusion using unified framework
    fused_base, fused_detail = fusion_unified_framework(
        base1, base2, detail1, detail2, saliency1, saliency2
    )
    
    # Reconstruct fused image
    fused_img = fused_base + fused_detail
    
    # Ensure pixel values are in valid range [0, 1]
    fused_img = np.clip(fused_img, 0, 1)
    
    return fused_img
