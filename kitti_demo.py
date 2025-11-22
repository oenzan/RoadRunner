"""
KITTI Dataset Detection Demo
Demonstrates Line, Car, and Sign Detection on KITTI Dataset sequences
"""

import cv2
import sys
import os
import urllib.request
import zipfile
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detectors import LineDetector, CarDetector, SignDetector
from utils.image_utils import save_image


def create_kitti_like_sample():
    """
    Create a KITTI-like road scene for demonstration
    """
    print("Creating KITTI-like sample image...")
    
    # Create realistic road scene (similar to KITTI)
    width, height = 1242, 375  # KITTI image dimensions
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky (gradient)
    for y in range(int(height * 0.4)):
        color = int(180 - y * 0.3)
        image[y, :] = (color, color, color)
    
    # Road
    road_y = int(height * 0.4)
    image[road_y:, :] = (80, 80, 80)
    
    # Left lane marking
    left_lane_pts = np.array([
        [int(width * 0.25), height],
        [int(width * 0.35), road_y]
    ], dtype=np.int32)
    cv2.line(image, tuple(left_lane_pts[0]), tuple(left_lane_pts[1]), (255, 255, 255), 5)
    
    # Right lane marking
    right_lane_pts = np.array([
        [int(width * 0.75), height],
        [int(width * 0.65), road_y]
    ], dtype=np.int32)
    cv2.line(image, tuple(right_lane_pts[0]), tuple(right_lane_pts[1]), (255, 255, 255), 5)
    
    # Center dashed lines
    for i in range(8):
        y_start = height - i * 60
        y_end = max(y_start - 40, road_y)
        if y_end > road_y:
            x_start = int(width * 0.5 - (height - y_start) * 0.05)
            x_end = int(width * 0.5 - (height - y_end) * 0.05)
            cv2.line(image, (x_start, y_start), (x_end, y_end), (255, 255, 0), 3)
    
    # Add some car-like rectangles
    # Car 1 (left)
    cv2.rectangle(image, (200, 250), (350, 330), (40, 40, 100), -1)
    cv2.rectangle(image, (220, 260), (330, 290), (80, 120, 180), -1)
    
    # Car 2 (center)
    cv2.rectangle(image, (520, 220), (650, 300), (50, 50, 120), -1)
    cv2.rectangle(image, (540, 230), (630, 260), (90, 130, 190), -1)
    
    # Car 3 (right)
    cv2.rectangle(image, (900, 260), (1050, 340), (30, 30, 90), -1)
    cv2.rectangle(image, (920, 270), (1030, 300), (70, 110, 170), -1)
    
    # Add some rectangular signs
    # Red stop sign (left)
    cv2.rectangle(image, (100, 180), (150, 230), (0, 0, 200), -1)
    
    # Blue info sign (right)
    cv2.rectangle(image, (1100, 190), (1150, 240), (200, 100, 0), -1)
    
    # Save to kitti_data directory
    kitti_dir = Path('kitti_data')
    kitti_dir.mkdir(exist_ok=True)
    sample_path = kitti_dir / 'kitti_like_sample.png'
    cv2.imwrite(str(sample_path), image)
    
    print(f"Created KITTI-like sample: {sample_path}")
    return str(sample_path)


def download_kitti_sample():
    """
    Download a sample KITTI dataset image for demonstration
    Uses KITTI raw data sample
    """
    kitti_dir = Path('kitti_data')
    kitti_dir.mkdir(exist_ok=True)
    
    # Try multiple KITTI sample sources
    sample_urls = [
        'https://raw.githubusercontent.com/bostondiditeam/kitti/master/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png',
        'https://github.com/charlesq34/frustum-pointnets/raw/master/kitti/image_sets/000001.png',
        'https://raw.githubusercontent.com/NVIDIA/DIGITS/master/examples/object-detection/kitti-data/000000.png'
    ]
    
    sample_path = kitti_dir / 'kitti_sample.png'
    
    if not sample_path.exists():
        print(f"Downloading KITTI sample image...")
        for idx, sample_url in enumerate(sample_urls):
            try:
                print(f"  Trying source {idx+1}/{len(sample_urls)}...")
                # Set a reasonable timeout and size limit
                req = urllib.request.Request(sample_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' not in content_type.lower():
                        print(f"  ✗ Invalid content type: {content_type}")
                        continue
                    
                    # Read with size limit (10MB max)
                    max_size = 10 * 1024 * 1024
                    data = response.read(max_size)
                    
                    # Save file
                    with open(str(sample_path), 'wb') as f:
                        f.write(data)
                    
                    print(f"  ✓ Downloaded to {sample_path}")
                    break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                if idx == len(sample_urls) - 1:
                    print("All download sources failed.")
                    print("Please manually download KITTI dataset images to kitti_data/ directory")
                    return None
    else:
        print(f"Using existing KITTI sample: {sample_path}")
    
    return str(sample_path)


def process_kitti_sequence(image_path, output_dir='kitti_results'):
    """
    Process a KITTI dataset image with all detectors
    
    Args:
        image_path: Path to KITTI image
        output_dir: Output directory for results
    """
    print("\n" + "=" * 70)
    print("KITTI Dataset Detection Demo")
    print("=" * 70)
    
    # Load image
    print(f"\nLoading KITTI image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    original_path = os.path.join(output_dir, 'original_kitti.png')
    save_image(image, original_path)
    
    # Create a combined result image
    combined_result = image.copy()
    
    # 1. LINE DETECTION
    print("\n[1/3] Running Line Detection...")
    try:
        line_detector = LineDetector(
            threshold=50,
            min_line_length=50,
            max_line_gap=30
        )
        
        # Detect lane lines with ROI
        _, lane_lines = line_detector.detect_lane_lines(image)
        print(f"      Detected {len(lane_lines)} lane lines")
        
        # Draw lines as lines (not bounding boxes)
        if len(lane_lines) > 0:
            for line in lane_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(combined_result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Save line detection result
        line_only_result = line_detector.draw_lines(image, lane_lines, color=(0, 255, 0), thickness=3)
        save_image(line_only_result, os.path.join(output_dir, 'kitti_lines.png'))
        
    except Exception as e:
        print(f"      Line detection error: {e}")
    
    # 2. CAR DETECTION
    print("\n[2/3] Running Car Detection...")
    try:
        car_detector = CarDetector(
            scale_factor=1.05,
            min_neighbors=2,
            min_size=(30, 30)
        )
        
        cars = car_detector.detect_cars(image)
        print(f"      Detected {len(cars)} cars")
        
        # Draw cars with bounding boxes
        if len(cars) > 0:
            for (x, y, w, h) in cars:
                cv2.rectangle(combined_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(combined_result, 'Car', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Save car detection result
        car_only_result = car_detector.draw_detections(image, cars, color=(255, 0, 0), thickness=2)
        save_image(car_only_result, os.path.join(output_dir, 'kitti_cars.png'))
        
    except Exception as e:
        print(f"      Car detection error: {e}")
    
    # 3. SIGN DETECTION
    print("\n[3/3] Running Sign Detection...")
    try:
        sign_detector = SignDetector(
            min_area=300,
            max_area=30000
        )
        
        signs = sign_detector.detect_all_signs(image)
        red_count = len(signs['red'])
        blue_count = len(signs['blue'])
        yellow_count = len(signs['yellow'])
        total_count = red_count + blue_count + yellow_count
        
        print(f"      Detected {total_count} signs (R:{red_count}, B:{blue_count}, Y:{yellow_count})")
        
        # Draw signs with bounding boxes
        for sign in signs['red']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(combined_result, f'Red {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for sign in signs['blue']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(combined_result, f'Blue {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for sign in signs['yellow']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(combined_result, f'Yellow {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Save sign detection result
        sign_only_result = sign_detector.draw_detections(image, signs)
        save_image(sign_only_result, os.path.join(output_dir, 'kitti_signs.png'))
        
    except Exception as e:
        print(f"      Sign detection error: {e}")
    
    # Save combined result
    combined_path = os.path.join(output_dir, 'kitti_combined_detection.png')
    save_image(combined_result, combined_path)
    
    print("\n" + "=" * 70)
    print("KITTI Detection Results:")
    print(f"  - Original: {original_path}")
    print(f"  - Lines only: {output_dir}/kitti_lines.png")
    print(f"  - Cars only: {output_dir}/kitti_cars.png")
    print(f"  - Signs only: {output_dir}/kitti_signs.png")
    print(f"  - Combined: {combined_path}")
    print("=" * 70)
    
    return combined_result


def main():
    """Main entry point for KITTI dataset demo"""
    
    # Check if user provided KITTI image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using provided KITTI image: {image_path}")
    else:
        # Try to download sample KITTI image
        print("No image provided. Attempting to download KITTI sample...")
        image_path = download_kitti_sample()
        
        if image_path is None:
            # Create KITTI-like sample if download fails
            print("\nDownload failed. Creating KITTI-like sample image...")
            image_path = create_kitti_like_sample()
    
    # Process the image
    process_kitti_sequence(image_path)


if __name__ == "__main__":
    main()
