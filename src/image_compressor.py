#!/usr/bin/env python3
"""
Image Compressor for GPT4O Token Optimization

This module provides image compression functionality to reduce token consumption
when sending images to GPT4O API. It implements various compression strategies
based on OpenAI's official token calculation rules.
"""

import os
import math
from pathlib import Path
from PIL import Image, ImageOps
import logging
from typing import Tuple, Optional, Dict, Any
import base64
import tempfile

logger = logging.getLogger(__name__)

class ImageCompressor:
    """
    Image compressor optimized for GPT4O token reduction.
    
    Based on OpenAI's official token calculation:
    - Images are scaled to max 1024x1024 pixels
    - Then scaled to 768x768 pixels
    - Divided into 512x512 pixel blocks
    - Each block consumes 170 tokens + 85 base tokens
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the image compressor.
        
        Args:
            config: Configuration dictionary with compression settings
        """
        self.config = config or {}
        
        # Compression settings
        self.max_dimension = self.config.get('max_dimension', 512)  # Target max dimension
        self.quality = self.config.get('quality', 85)  # JPEG quality
        self.optimize = self.config.get('optimize', True)  # Enable optimization
        self.delete_original = self.config.get('delete_original', True)  # Delete original images after compression
        

        
        logger.info(f"ImageCompressor initialized with max_dimension={self.max_dimension}, quality={self.quality}, optimize={self.optimize}")
    
    def calculate_target_size(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate target size to minimize token consumption.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple of (target_width, target_height)
        """
        # If image is already small enough, return original size
        if width <= self.max_dimension and height <= self.max_dimension:
            return width, height
        
        # Calculate scaling factor to fit within max_dimension
        scale_factor = min(self.max_dimension / width, self.max_dimension / height)
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure dimensions are even numbers (better compression)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        return new_width, new_height
    

    
    def compress_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Compress an image to minimize token consumption.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image (optional)
            
        Returns:
            Path to compressed image
        """
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                original_width, original_height = img.size
                logger.info(f"Original image: {original_width}x{original_height}")
                
                # Calculate target size
                target_width, target_height = self.calculate_target_size(original_width, original_height)
                
                # Resize image
                if target_width != original_width or target_height != original_height:
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Determine output path - use same name to replace original
                if output_path is None:
                    # Use same path to replace original image
                    output_path = image_path
                
                # Save compressed image
                img.save(output_path, quality=self.quality, optimize=self.optimize)
                
                logger.info(f"Compressed image: {target_width}x{target_height}")
                
                # Image compressed and replaced original file
                logger.info(f"Image compressed and replaced original: {image_path}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error compressing image {image_path}: {e}")
            return image_path  # Return original path if compression fails
    
    def compress_images_batch(self, image_paths: list, output_dir: Optional[str] = None) -> list:
        """
        Compress multiple images in batch and replace originals.
        
        Args:
            image_paths: List of image paths to compress
            output_dir: Output directory for compressed images (optional)
            
        Returns:
            List of compressed image paths (same as input paths)
        """
        compressed_paths = []
        
        for image_path in image_paths:
            try:
                if output_dir:
                    # Create output path in specified directory
                    input_path = Path(image_path)
                    output_path = Path(output_dir) / f"{input_path.stem}_compressed{input_path.suffix}"
                else:
                    output_path = None
                
                compressed_path = self.compress_image(image_path, str(output_path) if output_path else None)
                compressed_paths.append(compressed_path)
                
            except Exception as e:
                logger.error(f"Error in batch compression for {image_path}: {e}")
                compressed_paths.append(image_path)  # Use original if compression fails
        
        return compressed_paths
    
    def get_compression_stats(self, image_paths: list) -> Dict[str, Any]:
        """
        Get compression statistics for a list of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with compression statistics
        """
        stats = {
            'total_images': len(image_paths),
            'file_size_savings': 0.0
        }
        
        total_original_size = 0
        total_compressed_size = 0
        
        for image_path in image_paths:
            try:
                # Get original file size
                original_size = os.path.getsize(image_path)
                total_original_size += original_size
                
                # Compress and get compressed info
                compressed_path = self.compress_image(image_path)
                
                # Get compressed file size
                compressed_size = os.path.getsize(compressed_path)
                total_compressed_size += compressed_size
                
            except Exception as e:
                logger.error(f"Error getting stats for {image_path}: {e}")
        
        # Calculate file size savings
        if total_original_size > 0:
            stats['file_size_savings'] = ((total_original_size - total_compressed_size) / total_original_size) * 100
        
        return stats
    
    def create_optimized_config(self, max_dimension: int = 512) -> Dict[str, Any]:
        """
        Create optimized configuration for image compression.
        
        Args:
            max_dimension: Target max dimension for images
            
        Returns:
            Optimized configuration dictionary
        """
        return {
            'max_dimension': max_dimension,
            'quality': 85,
            'optimize': True,
            'delete_original': True
        }


def create_compressor_for_gpt4o(max_dimension: int = 512) -> ImageCompressor:
    """
    Create an image compressor optimized for GPT4O.
    
    Args:
        max_dimension: Target max dimension for images
        
    Returns:
        Configured ImageCompressor instance
    """
    config = ImageCompressor().create_optimized_config(max_dimension)
    return ImageCompressor(config)

