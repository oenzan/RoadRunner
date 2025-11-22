"""
Example: Sign Detection
Demonstrates how to use the SignDetector class
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detectors import SignDetector
from src.utils.image_utils import create_test_image_signs, save_image


def main():
    print("Sign Detection Example")
    print("-" * 40)
    
    # Create or load test image
    print("Creating test sign image...")
    image = create_test_image_signs()
    
    # Save original image
    save_image(image, 'results/example_sign_original.png')
    
    # Initialize sign detector with custom parameters
    detector = SignDetector(
        min_area=300,         # Minimum sign area in pixels
        max_area=100000,      # Maximum sign area in pixels
        min_circularity=0.6,  # Circularity threshold for circles
        min_triangularity=0.6 # Not used yet, for future triangularity checks
    )
    
    # Detect all signs
    print("\n1. Detecting all traffic signs...")
    signs = detector.detect_all_signs(image)
    
    # Print results
    print(f"   Red signs found: {len(signs['red'])}")
    for i, sign in enumerate(signs['red']):
        print(f"     - Sign {i+1}: {sign['shape']}, area={sign['area']:.0f} px²")
    
    print(f"   Blue signs found: {len(signs['blue'])}")
    for i, sign in enumerate(signs['blue']):
        print(f"     - Sign {i+1}: {sign['shape']}, area={sign['area']:.0f} px²")
    
    print(f"   Yellow signs found: {len(signs['yellow'])}")
    for i, sign in enumerate(signs['yellow']):
        print(f"     - Sign {i+1}: {sign['shape']}, area={sign['area']:.0f} px²")
    
    # Draw all detections
    result_all = detector.draw_detections(image, signs)
    save_image(result_all, 'results/example_sign_all.png')
    
    # Detect only red signs
    print("\n2. Detecting only red signs...")
    red_signs = detector.detect_red_signs(image)
    print(f"   Found {len(red_signs)} red signs")
    
    result_red = detector.draw_detections(image, {'red': red_signs, 'blue': [], 'yellow': []})
    save_image(result_red, 'results/example_sign_red_only.png')
    
    # Detect only blue signs
    print("\n3. Detecting only blue signs...")
    blue_signs = detector.detect_blue_signs(image)
    print(f"   Found {len(blue_signs)} blue signs")
    
    result_blue = detector.draw_detections(image, {'red': [], 'blue': blue_signs, 'yellow': []})
    save_image(result_blue, 'results/example_sign_blue_only.png')
    
    # Detect only yellow signs
    print("\n4. Detecting only yellow signs...")
    yellow_signs = detector.detect_yellow_signs(image)
    print(f"   Found {len(yellow_signs)} yellow signs")
    
    result_yellow = detector.draw_detections(image, {'red': [], 'blue': [], 'yellow': yellow_signs})
    save_image(result_yellow, 'results/example_sign_yellow_only.png')
    
    print("\n" + "-" * 40)
    print("Results saved to 'results/' directory:")
    print("  - example_sign_original.png")
    print("  - example_sign_all.png")
    print("  - example_sign_red_only.png")
    print("  - example_sign_blue_only.png")
    print("  - example_sign_yellow_only.png")


if __name__ == "__main__":
    main()
