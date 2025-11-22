"""
Sign Detection Module using Classical Computer Vision Methods
Uses color-based detection and shape detection - non-ML approaches
"""

import cv2
import numpy as np


class SignDetector:
    """Detects traffic signs using color-based and shape-based detection (classical CV methods)"""
    
    def __init__(self, 
                 min_area=500,
                 max_area=50000,
                 min_circularity=0.7,
                 min_triangularity=0.7):
        """
        Initialize Sign Detector
        
        Args:
            min_area: Minimum area of detected shapes
            max_area: Maximum area of detected shapes
            min_circularity: Minimum circularity for circular signs (0-1)
            min_triangularity: Minimum triangularity for triangular signs (0-1)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_triangularity = min_triangularity
    
    def detect_red_signs(self, image):
        """
        Detect red traffic signs (stop signs, yield signs, etc.)
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected sign bounding boxes and contours
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color
        # Red wraps around in HSV, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._filter_sign_contours(contours)
    
    def detect_blue_signs(self, image):
        """
        Detect blue traffic signs (information signs, etc.)
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected sign bounding boxes and contours
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._filter_sign_contours(contours)
    
    def detect_yellow_signs(self, image):
        """
        Detect yellow traffic signs (warning signs, etc.)
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected sign bounding boxes and contours
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create mask for yellow color
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._filter_sign_contours(contours)
    
    def _filter_sign_contours(self, contours):
        """
        Filter contours based on area and shape characteristics
        
        Args:
            contours: List of contours from cv2.findContours
            
        Returns:
            List of filtered contours with bounding boxes
        """
        filtered_signs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio (should be close to 1 for most signs)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if 0.5 < aspect_ratio < 2.0:  # Allow some variation
                filtered_signs.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'shape': self._classify_shape(contour)
                })
        
        return filtered_signs
    
    def _classify_shape(self, contour):
        """
        Classify the shape of a contour (circle, triangle, rectangle, etc.)
        
        Args:
            contour: Input contour
            
        Returns:
            Shape classification string
        """
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        num_vertices = len(approx)
        
        # Classify based on number of vertices
        if num_vertices == 3:
            return 'triangle'
        elif num_vertices == 4:
            return 'rectangle'
        elif num_vertices > 6:
            # Check circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > self.min_circularity:
                    return 'circle'
            return 'polygon'
        else:
            return 'polygon'
    
    def detect_all_signs(self, image):
        """
        Detect all types of traffic signs
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary with red, blue, and yellow sign detections
        """
        red_signs = self.detect_red_signs(image)
        blue_signs = self.detect_blue_signs(image)
        yellow_signs = self.detect_yellow_signs(image)
        
        return {
            'red': red_signs,
            'blue': blue_signs,
            'yellow': yellow_signs
        }
    
    def draw_detections(self, image, sign_dict):
        """
        Draw bounding boxes around detected signs
        
        Args:
            image: Input image to draw on
            sign_dict: Dictionary of sign detections from detect_all_signs()
            
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        
        # Draw red signs
        for sign in sign_dict.get('red', []):
            x, y, w, h = sign['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, f'Red {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw blue signs
        for sign in sign_dict.get('blue', []):
            x, y, w, h = sign['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, f'Blue {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw yellow signs
        for sign in sign_dict.get('yellow', []):
            x, y, w, h = sign['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(result, f'Yellow {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return result
