"""
AgriVision-AI: Advanced 8-Stage Preprocessing Pipeline
=====================================================

This module implements the state-of-the-art 8-stage preprocessing pipeline 
designed specifically for agricultural disease detection in maize and sugarcane crops.

Stages:
1. Super-resolution enhancement
2. AI-powered denoising
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. Leaf segmentation 
5. Texture analysis and mapping
6. Disease hotspot detection
7. Edge enhancement
8. Color space normalization

Authors: IIT Fellowship Team
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAgriPreprocessor:
    """
    Advanced image preprocessing pipeline for agricultural disease detection.
    
    This class implements an 8-stage preprocessing pipeline specifically designed
    to enhance subtle disease symptoms in agricultural crops (maize and sugarcane).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config or self._get_default_config()
        logger.info("🔥 INITIALIZING ADVANCED AGRICULTURAL AI PREPROCESSING PIPELINE!")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'img_size': 512,
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid_size': (8, 8),
            'denoise_h': 10,
            'denoise_template_window': 7,
            'denoise_search_window': 21,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'lbp_radius': 8,
            'lbp_n_points': 24,
            'kmeans_clusters': 4,
            'edge_canny_low': 50,
            'edge_canny_high': 150,
            'green_hsv_lower1': np.array([35, 40, 40]),
            'green_hsv_upper1': np.array([85, 255, 255]),
            'green_hsv_lower2': np.array([25, 40, 40]),
            'green_hsv_upper2': np.array([35, 255, 255])
        }
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and resize image to standard format.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed RGB image array (0-1 normalized)
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        h, w = bgr.shape[:2]
        size = self.config['img_size']
        scale_factor = size / max(h, w)
        
        # Resize maintaining aspect ratio
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to square
        pad_h = size - new_h
        pad_w = size - new_w
        bgr_padded = cv2.copyMakeBorder(
            bgr_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def stage1_super_resolution_upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 1: Super-resolution enhancement using cubic interpolation and sharpening.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Super-resolution enhanced image
        """
        h, w = image.shape[:2]
        
        # Upscale with cubic interpolation
        upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]], dtype=np.float32)
        
        sharpened = np.zeros_like(upscaled)
        for i in range(3):  # Apply to each channel
            sharpened[:, :, i] = cv2.filter2D(upscaled[:, :, i], -1, sharpen_kernel)
        
        # Resize back to original dimensions
        enhanced = cv2.resize(sharpened, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return np.clip(enhanced, 0, 1)
    
    def stage2_ai_powered_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 2: Advanced denoising using Non-Local Means and Bilateral filtering.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Denoised image
        """
        # Convert to uint8 for OpenCV functions
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Non-Local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8, None,
            self.config['denoise_h'],
            self.config['denoise_h'],
            self.config['denoise_template_window'],
            self.config['denoise_search_window']
        )
        
        # Bilateral filtering for edge preservation
        bilateral_filtered = cv2.bilateralFilter(
            denoised,
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
        
        return bilateral_filtered.astype(np.float32) / 255.0
    
    def stage3_enhance_with_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 3: CLAHE enhancement in LAB color space.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            CLAHE enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_tile_grid_size']
        )
        L_enhanced = clahe.apply(L)
        
        # Merge channels and convert back to RGB
        lab_enhanced = cv2.merge([L_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return rgb_enhanced.astype(np.float32) / 255.0
    
    def stage4_segment_leaf_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 4: Intelligent leaf segmentation using HSV color space.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Tuple of (segmented_image, leaf_mask)
        """
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Create masks for green regions
        mask1 = cv2.inRange(hsv, self.config['green_hsv_lower1'], self.config['green_hsv_upper1'])
        mask2 = cv2.inRange(hsv, self.config['green_hsv_lower2'], self.config['green_hsv_upper2'])
        
        # Combine masks
        plant_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations for noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        plant_mask_3d = np.stack([plant_mask] * 3, axis=-1) / 255.0
        segmented = image * plant_mask_3d
        
        return segmented, plant_mask.astype(np.float32) / 255.0
    
    def stage5_texture_analysis_features(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 5: Advanced texture analysis using Local Binary Patterns.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Texture-enhanced image
        """
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            P=self.config['lbp_n_points'],
            R=self.config['lbp_radius'],
            method='uniform'
        )
        
        # Normalize LBP
        if lbp.max() > lbp.min():
            lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())
        else:
            lbp_norm = np.zeros_like(lbp, dtype=np.float32)
        
        # Enhance texture in green channel
        enhanced = image.copy()
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * (1.0 + 0.2 * lbp_norm), 0, 1)
        
        return enhanced
    
    def stage6_disease_hotspot_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 6: AI-powered disease hotspot detection using K-means clustering in LAB space.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Tuple of (hotspot_enhanced_image, disease_probability_map)
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        h, w = lab.shape[:2]
        
        # Reshape for K-means
        lab_reshaped = lab.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.config['kmeans_clusters'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_reshaped).reshape(h, w)
        centers = kmeans.cluster_centers_  # Shape: (clusters, 3) for L,a,b
        
        # Identify disease clusters based on color characteristics
        # Heuristic: darker (low L) or reddish (high a) or bluish (low b) regions
        disease_clusters = []
        for i in range(centers.shape[0]):
            L_val, a_val, b_val = centers[i]
            if (L_val < 120) or (a_val > 135) or (b_val < 120):
                disease_clusters.append(i)
        
        # Create disease probability map
        disease_map = np.zeros((h, w), dtype=np.float32)
        for cluster_id in disease_clusters:
            disease_map[labels == cluster_id] = 1.0
        
        # Smooth the disease map
        disease_map = cv2.GaussianBlur(disease_map, (7, 7), 0)
        
        # Enhance disease regions
        enhanced = image.copy()
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] + 0.2 * disease_map, 0, 1)  # More red
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] - 0.1 * disease_map, 0, 1)  # Less green
        
        return enhanced, disease_map
    
    def stage7_edge_enhancement_ai(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 7: Advanced edge enhancement using Canny and Sobel operators.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Edge-enhanced image
        """
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges_canny = cv2.Canny(
            gray,
            self.config['edge_canny_low'],
            self.config['edge_canny_high']
        ).astype(np.float32) / 255.0
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = (sobel_magnitude / (sobel_magnitude.max() + 1e-6)).astype(np.float32)
        
        # Combine edge information
        combined_edges = np.clip(0.5 * edges_canny + 0.5 * sobel_magnitude, 0, 1)
        
        # Apply to image
        edges_3d = np.stack([combined_edges] * 3, axis=-1)
        enhanced = np.clip(image + 0.3 * edges_3d, 0, 1)
        
        return enhanced
    
    def stage8_color_space_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 8: Intelligent color space normalization and enhancement.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Color-normalized image
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Enhance saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        
        # Adaptive brightness adjustment
        v_mean = float(hsv[:, :, 2].mean())
        if v_mean < 100:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        elif v_mean > 200:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
    def apply_complete_pipeline(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply the complete 8-stage preprocessing pipeline.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Dictionary containing processed images and intermediate results
        """
        logger.info("🔥 Processing with ADVANCED AGRICULTURAL AI PIPELINE...")
        
        results = {'original': image}
        
        # Stage 1: Super-resolution
        stage1_result = self.stage1_super_resolution_upscale(image)
        results['stage1_super_resolution'] = stage1_result
        
        # Stage 2: Denoising
        stage2_result = self.stage2_ai_powered_denoising(stage1_result)
        results['stage2_denoising'] = stage2_result
        
        # Stage 3: CLAHE
        stage3_result = self.stage3_enhance_with_clahe(stage2_result)
        results['stage3_clahe'] = stage3_result
        
        # Stage 4: Segmentation
        stage4_result, leaf_mask = self.stage4_segment_leaf_regions(stage3_result)
        results['stage4_segmentation'] = stage4_result
        results['leaf_mask'] = leaf_mask
        
        # Stage 5: Texture Analysis
        stage5_result = self.stage5_texture_analysis_features(stage4_result)
        results['stage5_texture'] = stage5_result
        
        # Stage 6: Disease Hotspot Detection
        stage6_result, disease_map = self.stage6_disease_hotspot_detection(stage5_result)
        results['stage6_hotspots'] = stage6_result
        results['disease_probability_map'] = disease_map
        
        # Stage 7: Edge Enhancement
        stage7_result = self.stage7_edge_enhancement_ai(stage6_result)
        results['stage7_edges'] = stage7_result
        
        # Stage 8: Color Normalization
        final_result = self.stage8_color_space_analysis(stage7_result)
        results['final_processed'] = final_result
        
        logger.info("✅ Pipeline processing completed successfully!")
        
        return results
    
    def process_image_from_path(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Process an image from file path through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing all processing results
        """
        # Load and preprocess
        image = self.load_and_preprocess_image(image_path)
        
        # Apply pipeline
        results = self.apply_complete_pipeline(image)
        
        return results


def visualize_pipeline_results(results: Dict[str, np.ndarray], save_path: Optional[str] = None):
    """
    Create a comprehensive visualization of pipeline results.
    
    Args:
        results: Dictionary containing processing results
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('AgriVision-AI: Advanced 8-Stage Preprocessing Pipeline Results', 
                 fontsize=16, fontweight='bold')
    
    # Define visualization order
    viz_order = [
        ('original', 'Original Image'),
        ('stage1_super_resolution', 'Stage 1: Super-Resolution'),
        ('stage2_denoising', 'Stage 2: AI Denoising'),
        ('stage3_clahe', 'Stage 3: CLAHE Enhancement'),
        ('stage4_segmentation', 'Stage 4: Leaf Segmentation'),
        ('leaf_mask', 'Leaf Mask'),
        ('stage5_texture', 'Stage 5: Texture Analysis'),
        ('stage6_hotspots', 'Stage 6: Disease Hotspots'),
        ('disease_probability_map', 'Disease Probability Map'),
        ('stage7_edges', 'Stage 7: Edge Enhancement'),
        ('final_processed', 'Stage 8: Final Result'),
        ('final_processed', 'Pipeline Complete')
    ]
    
    for idx, (key, title) in enumerate(viz_order):
        row = idx // 4
        col = idx % 4
        
        if key in results:
            image = results[key]
            
            # Handle grayscale images
            if len(image.shape) == 2:
                axes[row, col].imshow(image, cmap='gray')
            else:
                axes[row, col].imshow(np.clip(image, 0, 1))
            
            axes[row, col].set_title(title, fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {save_path}")
    
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = AdvancedAgriPreprocessor()
    
    # Example configuration for different crops
    maize_config = {
        'green_hsv_lower1': np.array([30, 40, 40]),  # Adjusted for maize-specific greens
        'green_hsv_upper1': np.array([80, 255, 255]),
        'clahe_clip_limit': 2.5,  # Softer enhancement for maize
    }
    
    sugarcane_config = {
        'green_hsv_lower1': np.array([35, 50, 50]),  # Adjusted for sugarcane
        'green_hsv_upper1': np.array([85, 255, 255]),
        'clahe_clip_limit': 3.5,  # Stronger enhancement for sugarcane
    }
    
    logger.info("🌾 AgriVision-AI Preprocessing Pipeline Ready!")
    logger.info("📊 Configured for Maize and Sugarcane Disease Detection")
    logger.info("🔬 8-Stage Advanced Processing Available")
