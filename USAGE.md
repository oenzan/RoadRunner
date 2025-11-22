# RoadRunner Usage Guide

This guide provides detailed examples and usage instructions for the RoadRunner detection system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Line Detection](#line-detection)
3. [Car Detection](#car-detection)
4. [Sign Detection](#sign-detection)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)

## Quick Start

### Running the Main Demo
The easiest way to get started is to run the main application:

```bash
# Run with test images
python main.py

# Run with your own image
python main.py path/to/your/image.jpg
```

### Running Individual Examples
Each detector has its own example script in the `examples/` directory:

```bash
# Line detection example
python examples/line_detection_example.py

# Sign detection example
python examples/sign_detection_example.py

# Car detection example (requires an image with cars)
python examples/car_detection_example.py path/to/cars.jpg
```

## Line Detection

The LineDetector uses the Hough Transform to detect straight lines in images.

### Basic Usage

```python
from src.detectors import LineDetector
import cv2

# Load an image
image = cv2.imread('road.jpg')

# Create detector with default parameters
detector = LineDetector()

# Detect lines
lines = detector.detect_lines(image)

# Draw lines on the image
result = detector.draw_lines(image, lines)

# Save result
cv2.imwrite('result.jpg', result)
```

### Custom Parameters

```python
# Fine-tune detection parameters
detector = LineDetector(
    rho=1,              # Distance resolution (pixels)
    theta=np.pi/180,    # Angle resolution (radians)
    threshold=50,       # Accumulator threshold
    min_line_length=30, # Minimum line length (pixels)
    max_line_gap=10     # Maximum gap between line segments (pixels)
)
```

### Lane Detection with ROI

```python
# Detect lane lines (optimized for road images)
result, lane_lines = detector.detect_lane_lines(image)
print(f"Detected {len(lane_lines)} lane lines")
```

### Custom Region of Interest

```python
import numpy as np

# Define ROI vertices (trapezoid)
height, width = image.shape[:2]
roi_vertices = np.array([[
    (width * 0.1, height),        # Bottom left
    (width * 0.4, height * 0.6),  # Top left
    (width * 0.6, height * 0.6),  # Top right
    (width * 0.9, height)         # Bottom right
]], dtype=np.int32)

# Create ROI mask
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
roi_mask = detector.create_roi_mask(gray, roi_vertices[0])

# Detect lines only in ROI
lines = detector.detect_lines(image, roi_mask)
```

### Parameter Tuning Tips

- **Increase threshold** if you're getting too many false positives
- **Decrease min_line_length** to detect shorter lines
- **Increase max_line_gap** to connect fragmented lines
- **Adjust ROI** to focus on specific areas (e.g., road lanes)

## Car Detection

The CarDetector uses Haar Cascade Classifiers for vehicle detection.

### Basic Usage

```python
from src.detectors import CarDetector
import cv2

# Load an image
image = cv2.imread('traffic.jpg')

# Create detector
detector = CarDetector()

# Detect cars
cars = detector.detect_cars(image)
print(f"Found {len(cars)} cars")

# Draw bounding boxes
result = detector.draw_detections(image, cars)

# Save result
cv2.imwrite('cars_detected.jpg', result)
```

### Custom Parameters

```python
# Adjust detection sensitivity
detector = CarDetector(
    scale_factor=1.05,    # Smaller = more thorough but slower (1.05 - 1.3)
    min_neighbors=5,      # Higher = stricter detection (3 - 10)
    min_size=(30, 30)     # Minimum car size in pixels
)
```

### One-Step Detection and Drawing

```python
# Detect and draw in one call
result, cars = detector.detect_and_draw(image)
```

### Processing Detection Results

```python
# Get detailed information about each detection
for i, (x, y, w, h) in enumerate(cars):
    print(f"Car {i+1}:")
    print(f"  Position: ({x}, {y})")
    print(f"  Size: {w}x{h} pixels")
    print(f"  Center: ({x + w//2}, {y + h//2})")
```

### Parameter Tuning Tips

- **Lower scale_factor** (e.g., 1.05) for better detection but slower processing
- **Increase min_neighbors** to reduce false positives
- **Adjust min_size** based on expected car size in your images
- Works best with front/rear views of cars

## Sign Detection

The SignDetector uses color-based segmentation and shape analysis.

### Basic Usage

```python
from src.detectors import SignDetector
import cv2

# Load an image
image = cv2.imread('street.jpg')

# Create detector
detector = SignDetector()

# Detect all sign types
signs = detector.detect_all_signs(image)

# Print results
print(f"Red signs: {len(signs['red'])}")
print(f"Blue signs: {len(signs['blue'])}")
print(f"Yellow signs: {len(signs['yellow'])}")

# Draw all detections
result = detector.draw_detections(image, signs)

# Save result
cv2.imwrite('signs_detected.jpg', result)
```

### Detect Specific Colors Only

```python
# Detect only red signs (stop, yield, etc.)
red_signs = detector.detect_red_signs(image)

# Detect only blue signs (information)
blue_signs = detector.detect_blue_signs(image)

# Detect only yellow signs (warning)
yellow_signs = detector.detect_yellow_signs(image)
```

### Custom Parameters

```python
# Adjust detection parameters
detector = SignDetector(
    min_area=500,         # Minimum sign area (pixels²)
    max_area=50000,       # Maximum sign area (pixels²)
    min_circularity=0.7,  # Circularity threshold for circles (0-1)
    min_triangularity=0.7 # Triangularity threshold (future use)
)
```

### Analyzing Sign Details

```python
# Get detailed information about each sign
signs = detector.detect_all_signs(image)

for sign in signs['red']:
    print(f"Red sign detected:")
    print(f"  Shape: {sign['shape']}")
    print(f"  Area: {sign['area']:.0f} pixels²")
    print(f"  Bounding box: {sign['bbox']}")
    x, y, w, h = sign['bbox']
    print(f"  Position: ({x}, {y}), Size: {w}x{h}")
```

### Parameter Tuning Tips

- **Adjust min_area and max_area** based on expected sign sizes
- **Lower min_circularity** to detect less perfect circles
- Works best in good lighting conditions
- May need color range adjustments for different lighting

## Advanced Usage

### Processing Multiple Images

```python
from pathlib import Path
from src.detectors import LineDetector, SignDetector
import cv2

# Initialize detectors
line_detector = LineDetector()
sign_detector = SignDetector()

# Process all images in a directory
image_dir = Path('input_images')
output_dir = Path('output_results')
output_dir.mkdir(exist_ok=True)

for img_path in image_dir.glob('*.jpg'):
    print(f"Processing {img_path.name}...")
    
    # Load image
    image = cv2.imread(str(img_path))
    
    # Detect lines
    lines = line_detector.detect_lines(image)
    line_result = line_detector.draw_lines(image, lines)
    
    # Detect signs
    signs = sign_detector.detect_all_signs(image)
    sign_result = sign_detector.draw_detections(image, signs)
    
    # Save results
    cv2.imwrite(str(output_dir / f'lines_{img_path.name}'), line_result)
    cv2.imwrite(str(output_dir / f'signs_{img_path.name}'), sign_result)
```

### Combining Multiple Detectors

```python
from src.detectors import LineDetector, CarDetector, SignDetector
import cv2

# Load image
image = cv2.imread('road_scene.jpg')
result = image.copy()

# Initialize all detectors
line_detector = LineDetector()
car_detector = CarDetector()
sign_detector = SignDetector()

# Detect lines
lines = line_detector.detect_lines(image)
result = line_detector.draw_lines(result, lines, color=(0, 255, 0))

# Detect cars
cars = car_detector.detect_cars(image)
result = car_detector.draw_detections(result, cars, color=(255, 0, 0))

# Detect signs
signs = sign_detector.detect_all_signs(image)
result = sign_detector.draw_detections(result, signs)

# Save combined result
cv2.imwrite('combined_detection.jpg', result)
```

### Custom Visualization

```python
import cv2

# Custom line drawing
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
    # Add endpoints
    cv2.circle(result, (x1, y1), 5, (255, 0, 0), -1)
    cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)

# Custom car detection visualization
for i, (x, y, w, h) in enumerate(cars):
    # Draw filled rectangle with transparency
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    # Add label with background
    label = f"Car {i+1}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (label_w, label_h), _ = cv2.getTextSize(label, font, 0.6, 2)
    cv2.rectangle(result, (x, y-label_h-10), (x+label_w, y), (0, 255, 0), -1)
    cv2.putText(result, label, (x, y-5), font, 0.6, (0, 0, 0), 2)
```

## API Reference

### LineDetector

**Constructor:**
```python
LineDetector(rho=1, theta=np.pi/180, threshold=50, 
             min_line_length=50, max_line_gap=10)
```

**Methods:**
- `detect_lines(image, region_of_interest=None)` - Detect lines in image
- `draw_lines(image, lines, color=(0,255,0), thickness=2)` - Draw detected lines
- `detect_lane_lines(image)` - Detect road lane lines with ROI
- `create_roi_mask(image, vertices)` - Create region of interest mask
- `preprocess_image(image)` - Preprocess image for detection

### CarDetector

**Constructor:**
```python
CarDetector(cascade_path=None, scale_factor=1.1, 
            min_neighbors=3, min_size=(50, 50))
```

**Methods:**
- `detect_cars(image)` - Detect cars in image
- `draw_detections(image, cars, color=(0,255,0), thickness=2)` - Draw bounding boxes
- `detect_and_draw(image)` - Detect and draw in one step
- `preprocess_image(image)` - Preprocess image for detection

### SignDetector

**Constructor:**
```python
SignDetector(min_area=500, max_area=50000, 
             min_circularity=0.7, min_triangularity=0.7)
```

**Methods:**
- `detect_all_signs(image)` - Detect all sign types
- `detect_red_signs(image)` - Detect red signs only
- `detect_blue_signs(image)` - Detect blue signs only
- `detect_yellow_signs(image)` - Detect yellow signs only
- `draw_detections(image, sign_dict)` - Draw sign bounding boxes

## Troubleshooting

### Line Detection Issues
- **No lines detected**: Lower the threshold parameter
- **Too many false lines**: Increase threshold or min_line_length
- **Fragmented lines**: Increase max_line_gap

### Car Detection Issues
- **No cars detected**: Lower min_neighbors or use smaller min_size
- **False positives**: Increase min_neighbors parameter
- **Cascade file error**: Ensure internet connection for automatic download

### Sign Detection Issues
- **No signs detected**: Check lighting conditions, adjust min_area
- **Wrong colors detected**: Lighting affects HSV color ranges
- **Shape misclassification**: Adjust min_circularity parameter

## Performance Tips

1. **Resize large images** before processing for faster detection
2. **Use ROI** when possible to limit processing area
3. **Adjust parameters** based on your specific use case
4. **Process video frames** selectively (e.g., every 5th frame)
5. **Use appropriate color space** (HSV for color detection)

## Examples Directory

The `examples/` directory contains standalone scripts demonstrating each detector:
- `line_detection_example.py` - Line detection with various configurations
- `sign_detection_example.py` - Sign detection for all color types
- `car_detection_example.py` - Car detection with custom parameters

Run any example with:
```bash
python examples/<example_name>.py
```
