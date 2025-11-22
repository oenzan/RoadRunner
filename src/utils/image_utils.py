"""
Utility functions for image processing and visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path):
    """
    Load an image from file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        BGR image or None if loading failed
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    return image


def save_image(image, output_path):
    """
    Save an image to file
    
    Args:
        image: Image to save
        output_path: Path where to save the image
        
    Returns:
        True if successful, False otherwise
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if success:
        print(f"Image saved to {output_path}")
    else:
        print(f"Error: Could not save image to {output_path}")
    return success


def display_image(image, title="Image", figsize=(10, 6)):
    """
    Display an image using matplotlib
    
    Args:
        image: BGR image to display
        title: Window title
        figsize: Figure size as (width, height)
    """
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_images_grid(images, titles=None, figsize=(15, 10), cols=3):
    """
    Display multiple images in a grid
    
    Args:
        images: List of BGR images
        titles: List of titles for each image
        figsize: Figure size as (width, height)
        cols: Number of columns in the grid
    """
    n_images = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n_images)]
    
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, (image, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        
        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(rgb_image)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_test_image_road():
    """
    Create a simple test image with road lanes
    
    Returns:
        BGR image with lane markings
    """
    # Create a black image
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with gray road color
    image[:] = (80, 80, 80)
    
    # Draw left lane line
    cv2.line(image, (150, height), (250, int(height * 0.5)), (255, 255, 255), 5)
    
    # Draw right lane line
    cv2.line(image, (490, height), (390, int(height * 0.5)), (255, 255, 255), 5)
    
    # Draw center dashed line
    for i in range(5):
        y_start = height - i * 100
        y_end = y_start - 50
        if y_end > height * 0.5:
            cv2.line(image, (320, y_start), (320, y_end), (255, 255, 0), 3)
    
    return image


def create_test_image_signs():
    """
    Create a simple test image with colored shapes (simulating traffic signs)
    
    Returns:
        BGR image with colored shapes
    """
    # Create a white image
    height, width = 480, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw red circle (stop sign simulation)
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)
    
    # Draw yellow triangle (warning sign simulation)
    pts = np.array([[400, 100], [350, 200], [450, 200]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (0, 255, 255))
    
    # Draw blue rectangle (info sign simulation)
    cv2.rectangle(image, (500, 300), (600, 400), (255, 0, 0), -1)
    
    return image


def resize_image(image, width=None, height=None, maintain_aspect=True):
    """
    Resize an image
    
    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if maintain_aspect:
        if width is not None:
            aspect = width / w
            height = int(h * aspect)
        else:
            aspect = height / h
            width = int(w * aspect)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
