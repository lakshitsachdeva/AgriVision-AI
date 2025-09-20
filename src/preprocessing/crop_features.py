"""
AgriVision-AI: Crop-Specific Feature Extraction
==============================================

This module implements specialized feature extraction techniques tailored for
maize and sugarcane disease detection, focusing on:

1. HSV color space analysis for disease-specific color patterns
2. LAB color space for perceptual uniformity  
3. Texture analysis using advanced filters
4. Disease-specific color ranges and thresholds
5. Morphological operations for noise reduction

Authors: IIT Fellowship Team
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional, Any
import logging
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CropSpecificAnalyzer:
    """
    Advanced crop-specific feature analyzer for maize and sugarcane diseases.
    
    This class implements sophisticated color space analysis and texture feature
    extraction tailored to the specific visual characteristics of crop diseases.
    """
    
    def __init__(self, crop_type: str = 'maize'):
        """
        Initialize the crop-specific analyzer.
        
        Args:
            crop_type: Either 'maize' or 'sugarcane'
        """
        self.crop_type = crop_type.lower()
        
        # Disease-specific color configurations
        self.disease_configs = self._get_disease_configurations()
        
        logger.info(f"🌾 Initialized CropSpecificAnalyzer for {crop_type.upper()}")
    
    def _get_disease_configurations(self) -> Dict[str, Any]:
        """Get crop and disease-specific configurations."""
        
        if self.crop_type == 'maize':
            return {
                'healthy_hsv_ranges': [
                    {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},
                    {'lower': np.array([25, 40, 40]), 'upper': np.array([35, 255, 255])}
                ],
                'disease_patterns': {
                    'northern_leaf_blight': {
                        'hsv_range': {'lower': np.array([15, 50, 30]), 'upper': np.array([35, 200, 150])},
                        'lab_characteristics': {'L_range': (20, 80), 'a_range': (120, 140), 'b_range': (130, 150)},
                        'texture_frequency': 'high',
                        'lesion_shape': 'elongated'
                    },
                    'common_rust': {
                        'hsv_range': {'lower': np.array([5, 100, 50]), 'upper': np.array([20, 255, 200])},
                        'lab_characteristics': {'L_range': (30, 90), 'a_range': (135, 155), 'b_range': (140, 160)},
                        'texture_frequency': 'medium',
                        'lesion_shape': 'circular'
                    },
                    'gray_leaf_spot': {
                        'hsv_range': {'lower': np.array([0, 0, 40]), 'upper': np.array([180, 50, 120])},
                        'lab_characteristics': {'L_range': (40, 100), 'a_range': (110, 130), 'b_range': (110, 130)},
                        'texture_frequency': 'low',
                        'lesion_shape': 'rectangular'
                    }
                }
            }
        
        elif self.crop_type == 'sugarcane':
            return {
                'healthy_hsv_ranges': [
                    {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
                    {'lower': np.array([30, 40, 40]), 'upper': np.array([40, 255, 255])}
                ],
                'disease_patterns': {
                    'red_rot': {
                        'hsv_range': {'lower': np.array([0, 120, 50]), 'upper': np.array([10, 255, 255])},
                        'lab_characteristics': {'L_range': (20, 70), 'a_range': (140, 170), 'b_range': (120, 150)},
                        'texture_frequency': 'high',
                        'lesion_shape': 'irregular'
                    },
                    'smut': {
                        'hsv_range': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 80])},
                        'lab_characteristics': {'L_range': (10, 50), 'a_range': (120, 140), 'b_range': (120, 140)},
                        'texture_frequency': 'very_high',
                        'lesion_shape': 'powdery'
                    },
                    'yellow_leaf_disease': {
                        'hsv_range': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
                        'lab_characteristics': {'L_range': (60, 120), 'a_range': (100, 120), 'b_range': (140, 180)},
                        'texture_frequency': 'low',
                        'lesion_shape': 'linear'
                    },
                    'wilt': {
                        'hsv_range': {'lower': np.array([10, 50, 30]), 'upper': np.array([25, 180, 120])},
                        'lab_characteristics': {'L_range': (30, 80), 'a_range': (120, 140), 'b_range': (125, 145)},
                        'texture_frequency': 'medium',
                        'lesion_shape': 'diffuse'
                    }
                }
            }
    
    def extract_hsv_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract HSV-based features optimized for crop diseases.
        
        Args:
            image: RGB image (0-1 normalized)
            
        Returns:
            Dictionary containing HSV-based features and analysis
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        results = {'hsv_image': hsv_image}
        
        # Extract healthy tissue mask
        healthy_mask = self._extract_healthy_regions(hsv_image)
        results['healthy_mask'] = healthy_mask
        
        # Extract disease-specific features for each known disease
        disease_maps = {}
        for disease_name, config in self.disease_configs['disease_patterns'].items():
            disease_mask = cv2.inRange(
                hsv_image,
                config['hsv_range']['lower'],
                config['hsv_range']['upper']
            )
            
            # Apply morphological operations based on lesion characteristics
            disease_mask = self._refine_disease_mask(disease_mask, config['lesion_shape'])
            disease_maps[disease_name] = disease_mask.astype(np.float32) / 255.0
        
        results['disease_maps'] = disease_maps
        
        # Calculate HSV statistics
        results['hsv_statistics'] = self._calculate_hsv_statistics(hsv_image, healthy_mask)
        
        return results
    
    def extract_lab_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract LAB color space features for perceptually uniform analysis.
        
        Args:
            image: RGB image (0-1 normalized)
            
        Returns:
            Dictionary containing LAB-based features
        """
        # Convert to LAB
        lab_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        results = {'lab_image': lab_image}
        
        # Extract L*a*b* channels
        L, a, b = cv2.split(lab_image)
        results['L_channel'] = L.astype(np.float32) / 255.0
        results['a_channel'] = a.astype(np.float32) / 255.0  
        results['b_channel'] = b.astype(np.float32) / 255.0
        
        # Advanced disease detection using LAB clustering
        disease_probability_map = self._lab_disease_clustering(lab_image)
        results['disease_probability_map'] = disease_probability_map
        
        # Analyze color distribution in LAB space
        lab_statistics = self._calculate_lab_statistics(lab_image)
        results['lab_statistics'] = lab_statistics
        
        # Disease-specific LAB analysis
        disease_lab_features = {}
        for disease_name, config in self.disease_configs['disease_patterns'].items():
            features = self._extract_disease_lab_features(lab_image, config['lab_characteristics'])
            disease_lab_features[disease_name] = features
        
        results['disease_lab_features'] = disease_lab_features
        
        return results
    
    def extract_advanced_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract advanced texture features using multiple approaches.
        
        Args:
            image: RGB image (0-1 normalized)
            
        Returns:
            Dictionary containing texture features
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        results = {}
        
        # 1. Local Binary Pattern (LBP) with multiple scales
        lbp_features = self._extract_multiscale_lbp(gray)
        results['lbp_features'] = lbp_features
        
        # 2. Gray-Level Co-occurrence Matrix (GLCM) features
        glcm_features = self._extract_glcm_features(gray)
        results['glcm_features'] = glcm_features
        
        # 3. Gabor filter responses
        gabor_features = self._extract_gabor_features(gray)
        results['gabor_features'] = gabor_features
        
        # 4. Crop-specific texture patterns
        crop_texture_features = self._extract_crop_texture_patterns(gray)
        results['crop_texture_features'] = crop_texture_features
        
        return results
    
    def _extract_healthy_regions(self, hsv_image: np.ndarray) -> np.ndarray:
        """Extract mask for healthy plant regions."""
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for hsv_range in self.disease_configs['healthy_hsv_ranges']:
            mask = cv2.inRange(hsv_image, hsv_range['lower'], hsv_range['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def _refine_disease_mask(self, mask: np.ndarray, lesion_shape: str) -> np.ndarray:
        """Refine disease mask based on lesion characteristics."""
        if lesion_shape == 'elongated':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        elif lesion_shape == 'circular':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        elif lesion_shape == 'rectangular':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        elif lesion_shape == 'irregular':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        elif lesion_shape == 'powdery':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        elif lesion_shape == 'linear':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        else:  # diffuse
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # Apply appropriate morphological operations
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        return refined_mask
    
    def _calculate_hsv_statistics(self, hsv_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features from HSV channels."""
        stats = {}
        
        for i, channel_name in enumerate(['H', 'S', 'V']):
            channel = hsv_image[:, :, i]
            masked_channel = channel[mask > 0]
            
            if len(masked_channel) > 0:
                stats[f'{channel_name}_mean'] = float(np.mean(masked_channel))
                stats[f'{channel_name}_std'] = float(np.std(masked_channel))
                stats[f'{channel_name}_median'] = float(np.median(masked_channel))
                stats[f'{channel_name}_min'] = float(np.min(masked_channel))
                stats[f'{channel_name}_max'] = float(np.max(masked_channel))
            else:
                stats.update({f'{channel_name}_{stat}': 0.0 for stat in ['mean', 'std', 'median', 'min', 'max']})
        
        return stats
    
    def _lab_disease_clustering(self, lab_image: np.ndarray) -> np.ndarray:
        """Advanced disease detection using K-means clustering in LAB space."""
        h, w = lab_image.shape[:2]
        lab_reshaped = lab_image.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        n_clusters = 6 if self.crop_type == 'sugarcane' else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_reshaped).reshape(h, w)
        centers = kmeans.cluster_centers_
        
        # Identify disease clusters using crop-specific criteria
        disease_clusters = []
        for i, center in enumerate(centers):
            L_val, a_val, b_val = center
            
            if self.crop_type == 'maize':
                # Maize-specific disease detection criteria
                if (L_val < 100 and a_val > 130) or (L_val < 80) or (a_val < 115 and b_val < 115):
                    disease_clusters.append(i)
            elif self.crop_type == 'sugarcane':
                # Sugarcane-specific disease detection criteria  
                if (L_val < 90 and a_val > 135) or (L_val < 70) or (a_val > 145):
                    disease_clusters.append(i)
        
        # Create disease probability map
        disease_map = np.zeros((h, w), dtype=np.float32)
        for cluster_id in disease_clusters:
            disease_map[labels == cluster_id] = 1.0
        
        # Apply Gaussian smoothing
        disease_map = cv2.GaussianBlur(disease_map, (5, 5), 0)
        
        return disease_map
    
    def _calculate_lab_statistics(self, lab_image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive LAB color space statistics."""
        L, a, b = cv2.split(lab_image)
        
        stats = {}
        for channel, name in zip([L, a, b], ['L', 'a', 'b']):
            stats[f'{name}_mean'] = float(np.mean(channel))
            stats[f'{name}_std'] = float(np.std(channel))
            stats[f'{name}_skewness'] = float(self._calculate_skewness(channel))
            stats[f'{name}_kurtosis'] = float(self._calculate_kurtosis(channel))
        
        # Color diversity measures
        stats['color_diversity'] = float(np.std(lab_image.reshape(-1, 3), axis=0).mean())
        stats['chroma'] = float(np.sqrt(np.mean(a**2 + b**2)))
        
        return stats
    
    def _extract_disease_lab_features(self, lab_image: np.ndarray, characteristics: Dict) -> Dict[str, float]:
        """Extract disease-specific features from LAB image."""
        L, a, b = cv2.split(lab_image)
        
        features = {}
        
        # Check for characteristic LAB ranges
        L_range = characteristics['L_range']
        a_range = characteristics['a_range'] 
        b_range = characteristics['b_range']
        
        L_mask = (L >= L_range[0]) & (L <= L_range[1])
        a_mask = (a >= a_range[0]) & (a <= a_range[1])
        b_mask = (b >= b_range[0]) & (b <= b_range[1])
        
        combined_mask = L_mask & a_mask & b_mask
        
        features['disease_pixel_ratio'] = float(np.sum(combined_mask) / combined_mask.size)
        features['disease_area_coverage'] = float(np.sum(combined_mask))
        
        if np.sum(combined_mask) > 0:
            features['disease_L_mean'] = float(np.mean(L[combined_mask]))
            features['disease_a_mean'] = float(np.mean(a[combined_mask]))
            features['disease_b_mean'] = float(np.mean(b[combined_mask]))
        else:
            features.update({'disease_L_mean': 0.0, 'disease_a_mean': 0.0, 'disease_b_mean': 0.0})
        
        return features
    
    def _extract_multiscale_lbp(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract LBP features at multiple scales."""
        lbp_features = {}
        
        # Multiple radius and points combinations
        lbp_params = [
            (1, 8),   # Fine texture
            (2, 16),  # Medium texture  
            (3, 24),  # Coarse texture
        ]
        
        for radius, n_points in lbp_params:
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            lbp_features[f'lbp_r{radius}_p{n_points}'] = hist
            lbp_features[f'lbp_image_r{radius}_p{n_points}'] = lbp
        
        return lbp_features
    
    def _extract_glcm_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features."""
        # Reduce gray levels for computational efficiency
        gray_scaled = (gray_image / 64).astype(np.uint8) * 64
        
        features = {}
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for distance in distances:
            glcm = greycomatrix(gray_scaled, distances=[distance], angles=angles, 
                             levels=256, symmetric=True, normed=True)
            
            # Calculate texture properties
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                prop_values = greycoprops(glcm, prop).flatten()
                features[f'glcm_{prop}_d{distance}'] = float(np.mean(prop_values))
        
        return features
    
    def _extract_gabor_features(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Gabor filter responses for texture analysis."""
        gabor_features = {}
        
        # Gabor filter parameters optimized for agricultural textures
        frequencies = [0.1, 0.3, 0.5]  # Spatial frequencies
        angles = [0, 45, 90, 135]       # Orientations in degrees
        
        for freq in frequencies:
            for angle in angles:
                angle_rad = np.deg2rad(angle)
                
                # Apply Gabor filter
                real_response, _ = gabor(gray_image, frequency=freq, theta=angle_rad)
                
                # Calculate response statistics
                gabor_features[f'gabor_f{freq}_a{angle}_mean'] = np.mean(real_response)
                gabor_features[f'gabor_f{freq}_a{angle}_std'] = np.std(real_response)
                gabor_features[f'gabor_f{freq}_a{angle}_energy'] = np.sum(real_response**2)
        
        return gabor_features
    
    def _extract_crop_texture_patterns(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract crop-specific texture patterns."""
        features = {}
        
        # Crop-specific texture analysis
        if self.crop_type == 'maize':
            # Maize leaf texture characteristics
            features.update(self._analyze_maize_textures(gray_image))
        elif self.crop_type == 'sugarcane':
            # Sugarcane-specific texture patterns
            features.update(self._analyze_sugarcane_textures(gray_image))
        
        return features
    
    def _analyze_maize_textures(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze maize-specific texture patterns."""
        features = {}
        
        # Vertical stripe patterns (leaf veins)
        vertical_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        vertical_response = cv2.filter2D(gray_image, -1, vertical_kernel)
        features['maize_vertical_pattern'] = float(np.std(vertical_response))
        
        # Edge density (disease lesion boundaries)  
        edges = cv2.Canny(gray_image, 50, 150)
        features['maize_edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        return features
    
    def _analyze_sugarcane_textures(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze sugarcane-specific texture patterns."""
        features = {}
        
        # Horizontal stripe patterns (growth patterns)
        horizontal_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
        horizontal_response = cv2.filter2D(gray_image, -1, horizontal_kernel)
        features['sugarcane_horizontal_pattern'] = float(np.std(horizontal_response))
        
        # Surface roughness (fungal growth)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        features['sugarcane_surface_roughness'] = float(np.var(laplacian))
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)
    
    def analyze_complete_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete crop-specific feature analysis.
        
        Args:
            image: RGB image (0-1 normalized)
            
        Returns:
            Comprehensive feature analysis results
        """
        logger.info(f"🔬 Performing complete feature analysis for {self.crop_type.upper()}")
        
        results = {
            'crop_type': self.crop_type,
            'image_shape': image.shape
        }
        
        # HSV analysis
        hsv_features = self.extract_hsv_features(image)
        results['hsv_analysis'] = hsv_features
        
        # LAB analysis  
        lab_features = self.extract_lab_features(image)
        results['lab_analysis'] = lab_features
        
        # Texture analysis
        texture_features = self.extract_advanced_texture_features(image)
        results['texture_analysis'] = texture_features
        
        # Overall disease assessment
        disease_score = self._calculate_overall_disease_score(hsv_features, lab_features, texture_features)
        results['disease_assessment'] = disease_score
        
        logger.info("✅ Complete feature analysis finished")
        
        return results
    
    def _calculate_overall_disease_score(self, hsv_features: Dict, lab_features: Dict, 
                                       texture_features: Dict) -> Dict[str, float]:
        """Calculate overall disease assessment scores."""
        scores = {}
        
        # Combine evidence from different feature spaces
        disease_maps = hsv_features['disease_maps']
        lab_disease_map = lab_features['disease_probability_map']
        
        for disease_name, hsv_map in disease_maps.items():
            # Combine HSV and LAB evidence
            combined_score = 0.6 * np.mean(hsv_map) + 0.4 * np.mean(lab_disease_map)
            
            # Adjust based on texture characteristics
            disease_config = self.disease_configs['disease_patterns'][disease_name]
            texture_weight = self._get_texture_weight(disease_config['texture_frequency'])
            
            # Simple texture score (could be more sophisticated)
            texture_score = texture_features['glcm_features'].get('glcm_contrast_d1', 0)
            
            final_score = combined_score * (1 + texture_weight * texture_score)
            scores[disease_name] = float(np.clip(final_score, 0, 1))
        
        # Overall health score (inverse of max disease score)
        max_disease_score = max(scores.values()) if scores else 0
        scores['health_score'] = float(1 - max_disease_score)
        
        return scores
    
    def _get_texture_weight(self, texture_frequency: str) -> float:
        """Get texture weighting based on disease characteristics."""
        weights = {
            'very_high': 0.5,
            'high': 0.3,
            'medium': 0.2,
            'low': 0.1
        }
        return weights.get(texture_frequency, 0.2)


def visualize_crop_features(analyzer: CropSpecificAnalyzer, image: np.ndarray, 
                           save_path: Optional[str] = None):
    """
    Create comprehensive visualization of crop-specific feature analysis.
    
    Args:
        analyzer: CropSpecificAnalyzer instance
        image: RGB image to analyze
        save_path: Optional path to save visualization
    """
    # Perform analysis
    results = analyzer.analyze_complete_features(image)
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'AgriVision-AI: {analyzer.crop_type.upper()} Feature Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # HSV components
    hsv_image = results['hsv_analysis']['hsv_image']
    for i, (ax, title) in enumerate(zip(axes[0, 1:4], ['H Channel', 'S Channel', 'V Channel'])):
        ax.imshow(hsv_image[:, :, i], cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
    
    # Disease maps
    disease_maps = results['hsv_analysis']['disease_maps']
    for i, (disease, disease_map) in enumerate(list(disease_maps.items())[:4]):
        row, col = (i + 4) // 4, (i + 4) % 4
        if row < 4 and col < 4:
            axes[row, col].imshow(disease_map, cmap='hot')
            axes[row, col].set_title(f'{disease.replace("_", " ").title()}')
            axes[row, col].axis('off')
    
    # LAB disease probability
    lab_disease_map = results['lab_analysis']['disease_probability_map']
    axes[2, 0].imshow(lab_disease_map, cmap='hot')
    axes[2, 0].set_title('LAB Disease Probability')
    axes[2, 0].axis('off')
    
    # Texture features (LBP example)
    texture_features = results['texture_analysis']['lbp_features']
    if 'lbp_image_r2_p16' in texture_features:
        axes[2, 1].imshow(texture_features['lbp_image_r2_p16'], cmap='gray')
        axes[2, 1].set_title('LBP Texture (R=2, P=16)')
        axes[2, 1].axis('off')
    
    # Disease assessment scores
    disease_scores = results['disease_assessment']
    disease_names = [name for name in disease_scores.keys() if name != 'health_score']
    scores = [disease_scores[name] for name in disease_names]
    
    axes[3, 0].bar(range(len(disease_names)), scores)
    axes[3, 0].set_xticks(range(len(disease_names)))
    axes[3, 0].set_xticklabels([name.replace('_', '\n').title() for name in disease_names], 
                              rotation=45, fontsize=8)
    axes[3, 0].set_title('Disease Assessment Scores')
    axes[3, 0].set_ylabel('Score')
    
    # Health score
    health_score = disease_scores.get('health_score', 0)
    axes[3, 1].pie([health_score, 1-health_score], labels=['Healthy', 'Disease'], 
                  colors=['green', 'red'], autopct='%1.1f%%')
    axes[3, 1].set_title(f'Overall Health Assessment\n{health_score:.2f}')
    
    # Remove unused subplots
    for i in range(2, 4):
        for j in range(2, 4):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("🌾 AgriVision-AI Crop-Specific Feature Analysis")
    print("="*60)
    
    # Test both crop types
    for crop_type in ['maize', 'sugarcane']:
        print(f"\n🔬 Testing {crop_type.upper()} feature analysis")
        
        analyzer = CropSpecificAnalyzer(crop_type=crop_type)
        
        # Create a dummy image for testing
        dummy_image = np.random.rand(224, 224, 3)
        
        # Test individual feature extraction methods
        hsv_features = analyzer.extract_hsv_features(dummy_image)
        lab_features = analyzer.extract_lab_features(dummy_image)
        texture_features = analyzer.extract_advanced_texture_features(dummy_image)
        
        print(f"✅ HSV features extracted: {len(hsv_features)} components")
        print(f"✅ LAB features extracted: {len(lab_features)} components") 
        print(f"✅ Texture features extracted: {len(texture_features)} components")
        
        # Test complete analysis
        complete_results = analyzer.analyze_complete_features(dummy_image)
        disease_scores = complete_results['disease_assessment']
        
        print(f"📊 Disease assessment completed:")
        for disease, score in disease_scores.items():
            print(f"   - {disease.replace('_', ' ').title()}: {score:.3f}")
    
    print("\n🚀 All crop-specific feature analyzers tested successfully!")
