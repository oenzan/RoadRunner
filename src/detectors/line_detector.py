"""
Line Detection Module using Classical Computer Vision Methods
Uses Hough Transform for line detection - a non-ML approach
"""

import cv2
import numpy as np


class LineDetector:
    """Detects lines in images using Hough Transform (classical CV method)"""
    
    def __init__(self, 
                 rho=1, 
                 theta=np.pi/180, 
                 threshold=50,
                 min_line_length=50,
                 max_line_gap=10):
        """
        Initialize Line Detector with Hough Transform parameters
        
        Args:
            rho: Distance resolution of the accumulator in pixels
            theta: Angle resolution of the accumulator in radians
            threshold: Accumulator threshold parameter
            min_line_length: Minimum line length
            max_line_gap: Maximum allowed gap between line segments
        """
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def preprocess_image(self, image):
        """
        Preprocess image for line detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image with edges detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def detect_lines(self, image, region_of_interest=None):
        """
        Detect lines in the image using Hough Transform
        
        Args:
            image: Input BGR image
            region_of_interest: Optional ROI mask (None means use full image)
            
        Returns:
            List of detected lines as (x1, y1, x2, y2) coordinates
        """
        # Preprocess the image
        edges = self.preprocess_image(image)
        
        # Apply region of interest mask if provided
        if region_of_interest is not None:
            edges = cv2.bitwise_and(edges, edges, mask=region_of_interest)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines if lines is not None else []
    
    def draw_lines(self, image, lines, color=(0, 255, 0), thickness=2):
        """
        Draw detected lines on the image
        
        Args:
            image: Input image to draw on
            lines: List of lines from detect_lines()
            color: Line color in BGR format
            thickness: Line thickness
            
        Returns:
            Image with drawn lines
        """
        line_image = image.copy()
        
        if len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
        
        return line_image
    
    def create_roi_mask(self, image, vertices):
        """
        Create a region of interest mask
        
        Args:
            image: Input image
            vertices: List of vertices defining the ROI polygon
            
        Returns:
            Binary mask
        """
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [vertices], 255)
        return mask
    
    def detect_lane_lines(self, image):
        """
        Detect lane lines in a road image
        Optimized for road lane detection
        
        Args:
            image: Input road image
            
        Returns:
            Image with detected lane lines and the lines themselves
        """
        height, width = image.shape[:2]
        
        # Define region of interest for lane detection (trapezoidal shape)
        roi_vertices = np.array([[
            (width * 0.1, height),
            (width * 0.4, height * 0.6),
            (width * 0.6, height * 0.6),
            (width * 0.9, height)
        ]], dtype=np.int32)
        
        # Create ROI mask
        roi_mask = self.create_roi_mask(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            roi_vertices[0]
        )
        
        # Detect lines in ROI
        lines = self.detect_lines(image, roi_mask)
        
        # Draw lines
        result = self.draw_lines(image, lines, color=(0, 0, 255), thickness=3)
        
        return result, lines
