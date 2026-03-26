"""
Utilities for processing and resizing images from Jupyter notebook execution.

This module provides functionality to resize images to prevent 413 errors
when returning large images through the MCP protocol.
"""

import io
from typing import Tuple
from PIL import Image


def resize_image_if_needed(image_data: bytes, max_size: int = 512) -> bytes:
    """Resize image data if it exceeds the maximum dimensions.
    
    Args:
        image_data: The image data as bytes (PNG format)
        max_size: Maximum width/height in pixels (default: 512)
        
    Returns:
        Resized image data as bytes (PNG format)
    """
    try:
        # Open the image from bytes
        img = Image.open(io.BytesIO(image_data))
        
        # Check if resizing is needed
        width, height = img.size
        if width <= max_size and height <= max_size:
            # No resizing needed
            return image_data
            
        # Calculate new dimensions maintaining aspect ratio
        new_width, new_height = _calculate_resize_dimensions(width, height, max_size)
        
        # Resize the image using high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to bytes buffer as PNG
        output_buffer = io.BytesIO()
        resized_img.save(output_buffer, format='PNG', optimize=True)
        
        return output_buffer.getvalue()
        
    except Exception as e:
        # If resizing fails for any reason, return original image data
        print(f"Warning: Failed to resize image: {e}")
        return image_data


def _calculate_resize_dimensions(width: int, height: int, max_size: int) -> Tuple[int, int]:
    """Calculate new dimensions for resizing while maintaining aspect ratio.
    
    Args:
        width: Original width
        height: Original height
        max_size: Maximum allowed dimension
        
    Returns:
        Tuple of (new_width, new_height)
    """
    # Determine the scaling factor
    if width > height:
        # Width is the limiting dimension
        scale = max_size / width
    else:
        # Height is the limiting dimension  
        scale = max_size / height
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Ensure dimensions are at least 1 pixel
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return new_width, new_height


def get_image_dimensions(image_data: bytes) -> Tuple[int, int]:
    """Get the dimensions of an image from byte data.
    
    Args:
        image_data: The image data as bytes
        
    Returns:
        Tuple of (width, height) or (0, 0) if unable to read
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        return img.size
    except Exception:
        return (0, 0)