"""
AgriVision-AI: Advanced Model Architectures
==========================================

This module implements state-of-the-art model architectures for agricultural disease detection:
1. Lightweight CNNs (MobileNet, EfficientNet-Lite) for mobile deployment
2. Transfer Learning models (ResNet, EfficientNet) for high accuracy
3. Vision Transformers (ViT, Swin) for robust contextual understanding

Designed for maize and sugarcane disease classification with focus on:
- Real-time inference (<2s on mobile devices)
- High recall for disease detection
- Interpretable predictions with attention mechanisms

Authors: IIT Fellowship Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, Any, Optional, List, Tuple
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AgriDiseaseConfig:
    """Configuration class for agricultural disease detection models."""
    
    # Disease classes for different crops
    MAIZE_DISEASES = [
        'Healthy',
        'Northern_Leaf_Blight',
        'Common_Rust', 
        'Gray_Leaf_Spot',
        'Blight'
    ]
    
    SUGARCANE_DISEASES = [
        'Healthy',
        'Red_Rot',
        'Smut',
        'Wilt',
        'Yellow_Leaf_Disease',
        'Mosaic'
    ]
    
    # Model configurations
    MODEL_CONFIGS = {
        'mobilenet_v3_small': {'input_size': 224, 'inference_time': '<1s'},
        'mobilenet_v3_large': {'input_size': 224, 'inference_time': '<1.5s'},
        'efficientnet_lite0': {'input_size': 224, 'inference_time': '<1.2s'},
        'efficientnet_b0': {'input_size': 224, 'inference_time': '<2s'},
        'resnet50': {'input_size': 224, 'inference_time': '<2.5s'},
        'vit_tiny': {'input_size': 224, 'inference_time': '<3s'},
        'swin_tiny': {'input_size': 224, 'inference_time': '<3.5s'}
    }


class LightweightCNN(nn.Module):
    """
    Lightweight CNN architecture optimized for mobile deployment.
    Target: <2s inference time on mobile devices with high accuracy.
    """
    
    def __init__(self, model_name: str = 'mobilenet_v3_small', 
                 num_classes: int = 5, pretrained: bool = True):
        super(LightweightCNN, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif model_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Linear(960, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'efficientnet_lite0':
            self.backbone = timm.create_model('tf_efficientnet_lite0', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported lightweight model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps for visualization."""
        if 'mobilenet' in self.model_name:
            features = self.backbone.features(x)
            return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        else:
            return self.backbone.forward_features(x)


class TransferLearningCNN(nn.Module):
    """
    Transfer Learning CNN with strong backbone architectures.
    Optimized for high accuracy and recall in disease detection.
    """
    
    def __init__(self, model_name: str = 'resnet50', 
                 num_classes: int = 5, pretrained: bool = True):
        super(TransferLearningCNN, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        elif model_name == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported transfer learning model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_attention_maps(self, x):
        """Generate attention maps for model interpretability."""
        if 'resnet' in self.model_name:
            # Use Grad-CAM for ResNet
            return self._generate_gradcam(x)
        else:
            # Use built-in attention for EfficientNet
            return self.backbone.forward_features(x)
    
    def _generate_gradcam(self, x):
        """Generate Grad-CAM attention maps."""
        # This would implement Grad-CAM for ResNet
        # Simplified version - full implementation would require gradients
        features = self.backbone.layer4(
            self.backbone.layer3(
                self.backbone.layer2(
                    self.backbone.layer1(
                        self.backbone.maxpool(
                            self.backbone.relu(
                                self.backbone.bn1(
                                    self.backbone.conv1(x)
                                )
                            )
                        )
                    )
                )
            )
        )
        return F.adaptive_avg_pool2d(features, (7, 7))


class VisionTransformer(nn.Module):
    """
    Vision Transformer architecture for robust contextual understanding.
    Incorporates attention mechanisms for interpretable disease detection.
    """
    
    def __init__(self, model_name: str = 'vit_tiny_patch16_224', 
                 num_classes: int = 5, pretrained: bool = True):
        super(VisionTransformer, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load Vision Transformer models
        if model_name == 'vit_tiny_patch16_224':
            self.backbone = timm.create_model('vit_tiny_patch16_224', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        elif model_name == 'vit_small_patch16_224':
            self.backbone = timm.create_model('vit_small_patch16_224', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        elif model_name == 'swin_tiny_patch4_window7_224':
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224', 
                                            pretrained=pretrained, 
                                            num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
        # Add crop-specific attention layer
        self.crop_attention = CropSpecificAttention(embed_dim=384 if 'tiny' in model_name else 768)
    
    def forward(self, x, return_attention=False):
        if return_attention:
            # Extract attention weights for visualization
            features = self.backbone.forward_features(x)
            attention_weights = self.crop_attention(features)
            output = self.backbone.forward_head(features)
            return output, attention_weights
        else:
            return self.backbone(x)
    
    def get_attention_maps(self, x):
        """Extract attention maps for disease localization."""
        with torch.no_grad():
            # Get patch embeddings
            features = self.backbone.forward_features(x)
            
            # Apply crop-specific attention
            attention_maps = self.crop_attention(features)
            
            return attention_maps


class CropSpecificAttention(nn.Module):
    """
    Custom attention mechanism tailored for agricultural disease patterns.
    """
    
    def __init__(self, embed_dim: int = 384):
        super(CropSpecificAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.attention_head = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # Disease-specific attention patterns
        self.disease_queries = nn.Parameter(torch.randn(5, embed_dim))  # 5 common disease patterns
        
        # Color-space attention (for HSV/LAB features)
        self.color_attention = nn.Linear(embed_dim, 3)  # RGB channels
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Expand disease queries for batch
        disease_queries = self.disease_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.attention_head(
            disease_queries, x, x
        )
        
        # Apply color-space attention
        color_weights = torch.softmax(self.color_attention(attended_features), dim=-1)
        
        return attention_weights, color_weights


class AgriEnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures for robust predictions.
    Implements weighted voting based on model confidence and crop type.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(AgriEnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:
            self.weights = nn.Parameter(torch.tensor(weights))
        
        # Confidence estimation layer
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.num_models * 5, 64),  # Assuming 5 classes
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_individual=False):
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # [batch, num_models, num_classes]
        
        # Weighted ensemble
        weights = torch.softmax(self.weights, dim=0)
        ensemble_pred = torch.sum(stacked_preds * weights.view(1, -1, 1), dim=1)
        
        # Estimate confidence
        flat_preds = stacked_preds.view(x.shape[0], -1)
        confidence = self.confidence_estimator(flat_preds)
        
        if return_individual:
            return ensemble_pred, predictions, confidence
        else:
            return ensemble_pred, confidence


class CropSpecificFeatureExtractor(nn.Module):
    """
    Crop-specific feature extractor focusing on HSV/LAB color spaces
    and texture patterns specific to maize and sugarcane diseases.
    """
    
    def __init__(self, crop_type: str = 'maize'):
        super(CropSpecificFeatureExtractor, self).__init__()
        
        self.crop_type = crop_type
        
        # Color space feature extractors
        self.hsv_extractor = self._build_color_extractor('hsv')
        self.lab_extractor = self._build_color_extractor('lab')
        
        # Texture feature extractor
        self.texture_extractor = self._build_texture_extractor()
        
        # Crop-specific parameters
        if crop_type == 'maize':
            self.disease_color_ranges = {
                'blight': [(30, 80), (40, 255), (40, 200)],  # HSV ranges for blight
                'rust': [(10, 25), (100, 255), (100, 255)]   # HSV ranges for rust
            }
        elif crop_type == 'sugarcane':
            self.disease_color_ranges = {
                'red_rot': [(0, 10), (120, 255), (50, 255)],  # Reddish lesions
                'yellow_leaf': [(20, 30), (100, 255), (150, 255)]  # Yellowing
            }
    
    def _build_color_extractor(self, color_space: str):
        """Build color space specific feature extractor."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )
    
    def _build_texture_extractor(self):
        """Build texture feature extractor using Gabor-like filters."""
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),  # Large kernel for texture
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128)
        )
    
    def forward(self, rgb_image, hsv_image=None, lab_image=None):
        """
        Extract crop-specific features from multi-channel inputs.
        
        Args:
            rgb_image: RGB image tensor
            hsv_image: Optional HSV converted image
            lab_image: Optional LAB converted image
            
        Returns:
            Combined feature vector
        """
        # Extract features from different color spaces
        rgb_features = self.texture_extractor(rgb_image)
        
        if hsv_image is not None:
            hsv_features = self.hsv_extractor(hsv_image)
        else:
            hsv_features = self.hsv_extractor(rgb_image)
        
        if lab_image is not None:
            lab_features = self.lab_extractor(lab_image)
        else:
            lab_features = self.lab_extractor(rgb_image)
        
        # Combine features
        combined_features = torch.cat([rgb_features, hsv_features, lab_features], dim=1)
        
        return combined_features


def create_agri_model(model_type: str, crop_type: str, num_classes: int, 
                     pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create agricultural disease detection models.
    
    Args:
        model_type: Type of model ('lightweight', 'transfer', 'vit', 'ensemble')
        crop_type: Type of crop ('maize', 'sugarcane')  
        num_classes: Number of disease classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured model instance
    """
    
    if model_type == 'lightweight':
        model_name = kwargs.get('model_name', 'mobilenet_v3_small')
        return LightweightCNN(model_name=model_name, 
                             num_classes=num_classes, 
                             pretrained=pretrained)
    
    elif model_type == 'transfer':
        model_name = kwargs.get('model_name', 'resnet50')
        return TransferLearningCNN(model_name=model_name,
                                  num_classes=num_classes,
                                  pretrained=pretrained)
    
    elif model_type == 'vit':
        model_name = kwargs.get('model_name', 'vit_tiny_patch16_224')
        return VisionTransformer(model_name=model_name,
                                num_classes=num_classes,
                                pretrained=pretrained)
    
    elif model_type == 'ensemble':
        # Create ensemble of different architectures
        lightweight_model = LightweightCNN(num_classes=num_classes, pretrained=pretrained)
        transfer_model = TransferLearningCNN(num_classes=num_classes, pretrained=pretrained)
        vit_model = VisionTransformer(num_classes=num_classes, pretrained=pretrained)
        
        models = [lightweight_model, transfer_model, vit_model]
        weights = kwargs.get('ensemble_weights', [0.3, 0.4, 0.3])
        
        return AgriEnsembleModel(models=models, weights=weights)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)) -> Dict:
    """
    Get comprehensive model summary including parameters, FLOPs, and memory usage.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        
    Returns:
        Dictionary containing model statistics
    """
    try:
        from torchsummary import summary
        from thop import profile
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Calculate FLOPs and parameters
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'parameters': params,
            'flops': flops,
            'model_size_mb': model_size_mb,
            'input_size': input_size
        }
        
    except ImportError:
        logger.warning("torchsummary or thop not installed. Install for detailed model summary.")
        return {
            'parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': 'Unknown',
            'flops': 'Unknown'
        }


if __name__ == "__main__":
    # Example usage and testing
    print("🌾 AgriVision-AI Model Architectures")
    print("="*50)
    
    # Test different model types
    for crop in ['maize', 'sugarcane']:
        num_classes = len(AgriDiseaseConfig.MAIZE_DISEASES) if crop == 'maize' else len(AgriDiseaseConfig.SUGARCANE_DISEASES)
        
        print(f"\n🔬 Testing models for {crop.upper()} (Classes: {num_classes})")
        
        # Lightweight model
        lightweight = create_agri_model('lightweight', crop, num_classes)
        print(f"✅ Lightweight CNN: {sum(p.numel() for p in lightweight.parameters()):,} parameters")
        
        # Transfer learning model  
        transfer = create_agri_model('transfer', crop, num_classes)
        print(f"✅ Transfer Learning CNN: {sum(p.numel() for p in transfer.parameters()):,} parameters")
        
        # Vision Transformer
        vit = create_agri_model('vit', crop, num_classes)
        print(f"✅ Vision Transformer: {sum(p.numel() for p in vit.parameters()):,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            light_out = lightweight(dummy_input)
            transfer_out = transfer(dummy_input)
            vit_out = vit(dummy_input)
            
        print(f"📊 Output shapes - Light: {light_out.shape}, Transfer: {transfer_out.shape}, ViT: {vit_out.shape}")
    
    print("\n🚀 All model architectures initialized successfully!")
