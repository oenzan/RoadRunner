"""
Example: Line Detection
Demonstrates how to use the LineDetector class
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detectors import LineDetector
from src.utils.image_utils import create_test_image_road, save_image


def main():
    print("Line Detection Example")
    print("-" * 40)
    
    # Create or load test image
    print("Creating test road image...")
    image = create_test_image_road()
    
    # Save original image
    save_image(image, 'results/example_line_original.png')
    
    # Initialize line detector with custom parameters
    import numpy as np
    detector = LineDetector(
        rho=1,                  # Distance resolution
        theta=np.pi/180,        # Angle resolution (in radians, 1 degree)
        threshold=30,           # Minimum votes
        min_line_length=20,     # Minimum line length
        max_line_gap=15         # Maximum gap between segments
    )
    
    # Detect all lines
    print("\n1. Detecting all lines...")
    lines = detector.detect_lines(image)
    print(f"   Found {len(lines)} lines")
    
    # Draw lines
    result_all = detector.draw_lines(image, lines, color=(0, 255, 0), thickness=2)
    save_image(result_all, 'results/example_line_all_lines.png')
    
    # Detect lane lines with ROI
    print("\n2. Detecting lane lines with ROI...")
    result_lanes, lane_lines = detector.detect_lane_lines(image)
    print(f"   Found {len(lane_lines)} lane lines")
    save_image(result_lanes, 'results/example_line_lanes.png')
    
    # Custom ROI example
    print("\n3. Custom ROI detection...")
    import numpy as np
    height, width = image.shape[:2]
    
    # Define custom trapezoid ROI
    roi_vertices = np.array([[
        (width * 0.2, height),
        (width * 0.45, height * 0.55),
        (width * 0.55, height * 0.55),
        (width * 0.8, height)
    ]], dtype=np.int32)
    
    # Create ROI mask
    roi_mask = detector.create_roi_mask(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        roi_vertices[0]
    )
    
    # Detect lines in custom ROI
    custom_lines = detector.detect_lines(image, roi_mask)
    print(f"   Found {len(custom_lines)} lines in custom ROI")
    
    result_custom = detector.draw_lines(image, custom_lines, color=(255, 0, 255), thickness=2)
    save_image(result_custom, 'results/example_line_custom_roi.png')
    
    print("\n" + "-" * 40)
    print("Results saved to 'results/' directory:")
    print("  - example_line_original.png")
    print("  - example_line_all_lines.png")
    print("  - example_line_lanes.png")
    print("  - example_line_custom_roi.png")


if __name__ == "__main__":
    main()
