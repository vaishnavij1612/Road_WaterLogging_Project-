"""
preprocessing.py
================
Standardized data preprocessing and augmentation pipeline for road waterlogging.

Key Design Principles:
1. Interpolation Safety:
   - Images are resized using Bilinear or Bicubic interpolation.
   - Segmentation masks are ALWAYS resized using Nearest-Neighbor (NEAREST)
     interpolation to preserve discrete class label values without blurring.
2. Synchronized Augmentation:
   - For segmentation tasks, spatial augmentations (flips, rotations, scaling)
     are applied identically to both the input image and corresponding mask.
   - Photometric augmentations (brightness, contrast, color jitter) are applied
     only to the input image, never to the mask.
3. Colab / CPU / GPU Compatibility:
   - Built on PyTorch and torchvision with flexible tensor or PIL pipelines.
"""

import random
from typing import Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF



# Standard ImageNet normalization parameters for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classification_transforms(
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> T.Compose:
    """
    Returns image transformation pipeline for classification tasks.
    
    Args:
        img_size: Target (height, width) tuple.
        is_training: If True, applies data augmentations.
    """
    if is_training:
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class DualSegmentationTransform:
    """
    Synchronized transformation applied simultaneously to image and segmentation mask.
    Guarantees exact geometric alignment between image and ground-truth mask.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        hflip_prob: float = 0.5,
        max_rotation_deg: float = 15.0
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.hflip_prob = hflip_prob
        self.max_rotation_deg = max_rotation_deg

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coordinated transforms.
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (Grayscale or L mode with integer class indices)
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Resize image with Bilinear, mask with Nearest Neighbor
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_training:
            # Synchronized Random Horizontal Flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Synchronized Random Rotation
            if self.max_rotation_deg > 0:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # Photometric transform on image only
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        # Convert image to tensor and normalize
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Convert mask to LongTensor (class indices) without scaling
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)

        return img_tensor, mask_tensor


def load_and_preprocess_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Safely load and preprocess a single image for inference.
    Returns (preprocessed_tensor_1_C_H_W, original_pil_image).
    """
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")

    transform = get_classification_transforms(img_size=target_size, is_training=False)
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension: [1, 3, H, W]

    return tensor, pil_img
