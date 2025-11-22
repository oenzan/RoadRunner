"""
Example: Car Detection
Demonstrates how to use the CarDetector class
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detectors import CarDetector


def main():
    print("Car Detection Example")
    print("-" * 40)
    print("\nNote: Car detection requires actual car images and cascade files.")
    print("This example shows how to use the detector with your own images.\n")
    
    # Check if user provided an image
    if len(sys.argv) < 2:
        print("Usage: python car_detection_example.py <path_to_image>")
        print("\nExample:")
        print("  python car_detection_example.py ../sample_images/cars.jpg")
        print("\nThe detector will:")
        print("  1. Load the image")
        print("  2. Initialize Haar Cascade classifier")
        print("  3. Detect cars at multiple scales")
        print("  4. Draw bounding boxes around detected cars")
        print("  5. Save the result")
        return
    
    image_path = sys.argv[1]
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    try:
        # Initialize car detector
        print("\nInitializing car detector...")
        detector = CarDetector(
            scale_factor=1.1,     # How much to reduce image size at each scale
            min_neighbors=3,      # Minimum neighbors for detection
            min_size=(50, 50)     # Minimum car size in pixels
        )
        
        # Detect cars
        print("Detecting cars...")
        cars = detector.detect_cars(image)
        print(f"Found {len(cars)} cars")
        
        # Print detection details
        for i, (x, y, w, h) in enumerate(cars):
            print(f"  Car {i+1}: position=({x}, {y}), size={w}x{h}")
        
        # Draw detections
        result = detector.draw_detections(image, cars, color=(0, 255, 0), thickness=3)
        
        # Save result
        output_path = 'results/example_car_detection.png'
        cv2.imwrite(output_path, result)
        print(f"\nResult saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during car detection: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure the cascade file is available")
        print("  - Try adjusting detection parameters")
        print("  - Use a high-quality image with visible cars")


if __name__ == "__main__":
    main()
