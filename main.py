"""
RoadRunner - Main Application
Demonstrates Line Detection, Car Detection, and Sign Detection
using Classical Computer Vision Methods (Non-Machine Learning)
"""

import cv2
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detectors import LineDetector, CarDetector, SignDetector
from utils.image_utils import (
    load_image, save_image, create_test_image_road, 
    create_test_image_signs, display_images_grid
)


def demo_line_detection(image=None, output_dir='results'):
    """
    Demonstrate line detection
    
    Args:
        image: Input image (if None, uses test image)
        output_dir: Directory to save results
    """
    print("\n=== Line Detection Demo ===")
    
    # Use test image if none provided
    if image is None:
        print("Creating test road image...")
        image = create_test_image_road()
    
    # Initialize line detector
    detector = LineDetector(
        threshold=50,
        min_line_length=30,
        max_line_gap=20
    )
    
    # Detect lines
    print("Detecting lines...")
    lines = detector.detect_lines(image)
    print(f"Found {len(lines)} lines")
    
    # Draw lines
    result = detector.draw_lines(image, lines)
    
    # Detect lane lines with ROI
    print("Detecting lane lines with ROI...")
    lane_result, lane_lines = detector.detect_lane_lines(image)
    print(f"Found {len(lane_lines)} lane lines")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_image(result, os.path.join(output_dir, 'line_detection.png'))
    save_image(lane_result, os.path.join(output_dir, 'lane_detection.png'))
    
    return result, lane_result


def demo_car_detection(image=None, output_dir='results'):
    """
    Demonstrate car detection
    
    Args:
        image: Input image (if None, skips demo)
        output_dir: Directory to save results
    """
    print("\n=== Car Detection Demo ===")
    
    if image is None:
        print("No input image provided. Skipping car detection demo.")
        print("To test car detection, provide an image with cars.")
        return None
    
    try:
        # Initialize car detector
        print("Initializing car detector...")
        detector = CarDetector(
            scale_factor=1.1,
            min_neighbors=3,
            min_size=(50, 50)
        )
        
        # Detect cars
        print("Detecting cars...")
        cars = detector.detect_cars(image)
        print(f"Found {len(cars)} cars")
        
        # Draw detections
        result = detector.draw_detections(image, cars)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        save_image(result, os.path.join(output_dir, 'car_detection.png'))
        
        return result
        
    except Exception as e:
        print(f"Car detection error: {e}")
        print("Note: Car detection requires specific cascade files and may not work with test images")
        return None


def demo_sign_detection(image=None, output_dir='results'):
    """
    Demonstrate sign detection
    
    Args:
        image: Input image (if None, uses test image)
        output_dir: Directory to save results
    """
    print("\n=== Sign Detection Demo ===")
    
    # Use test image if none provided
    if image is None:
        print("Creating test sign image...")
        image = create_test_image_signs()
    
    # Initialize sign detector
    detector = SignDetector(
        min_area=500,
        max_area=50000
    )
    
    # Detect signs
    print("Detecting traffic signs...")
    signs = detector.detect_all_signs(image)
    
    red_count = len(signs['red'])
    blue_count = len(signs['blue'])
    yellow_count = len(signs['yellow'])
    total_count = red_count + blue_count + yellow_count
    
    print(f"Found {total_count} signs:")
    print(f"  - Red signs: {red_count}")
    print(f"  - Blue signs: {blue_count}")
    print(f"  - Yellow signs: {yellow_count}")
    
    # Draw detections
    result = detector.draw_detections(image, signs)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_image(result, os.path.join(output_dir, 'sign_detection.png'))
    
    return result


def main():
    """Main application entry point"""
    print("=" * 60)
    print("RoadRunner - Computer Vision Detection System")
    print("Line Detection | Car Detection | Sign Detection")
    print("Using Classical CV Methods (Non-ML)")
    print("=" * 60)
    
    # Check for input image
    input_image = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nLoading input image: {image_path}")
        input_image = load_image(image_path)
        if input_image is None:
            print("Failed to load image. Using test images instead.")
    
    # Run demonstrations
    results = []
    titles = []
    
    # Line detection
    line_result, lane_result = demo_line_detection(input_image)
    if line_result is not None:
        results.append(line_result)
        titles.append("Line Detection")
    if lane_result is not None:
        results.append(lane_result)
        titles.append("Lane Detection (ROI)")
    
    # Car detection (only if input image provided)
    if input_image is not None:
        car_result = demo_car_detection(input_image)
        if car_result is not None:
            results.append(car_result)
            titles.append("Car Detection")
    
    # Sign detection
    sign_result = demo_sign_detection(input_image)
    if sign_result is not None:
        results.append(sign_result)
        titles.append("Sign Detection")
    
    print("\n" + "=" * 60)
    print("Detection completed! Results saved to 'results/' directory")
    print("=" * 60)
    
    # Display results (optional - commented out for headless environments)
    # if results:
    #     display_images_grid(results, titles)


if __name__ == "__main__":
    main()
