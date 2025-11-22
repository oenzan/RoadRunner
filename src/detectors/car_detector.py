"""
Car Detection Module using Classical Computer Vision Methods
Uses Haar Cascade Classifier - a non-ML approach (feature-based detection)
"""

import cv2
import numpy as np
import urllib.request
import os


class CarDetector:
    """Detects cars in images using Haar Cascade Classifier (classical CV method)"""
    
    def __init__(self, cascade_path=None, scale_factor=1.1, min_neighbors=3, min_size=(50, 50)):
        """
        Initialize Car Detector with Haar Cascade
        
        Args:
            cascade_path: Path to Haar cascade XML file
            scale_factor: Parameter specifying how much the image size is reduced at each scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Download or use provided cascade file
        if cascade_path is None:
            cascade_path = self._download_cascade()
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
    
    def _download_cascade(self):
        """Download Haar cascade file for car detection if not available"""
        cascade_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(cascade_dir, exist_ok=True)
        
        cascade_path = os.path.join(cascade_dir, 'haarcascade_car.xml')
        
        if not os.path.exists(cascade_path):
            # Try to download from OpenCV GitHub repository
            url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
            try:
                print(f"Downloading Haar cascade for car detection...")
                urllib.request.urlretrieve(url, cascade_path)
                print(f"Cascade downloaded to {cascade_path}")
            except Exception as e:
                print(f"Warning: Could not download cascade file: {e}")
                print("Car detection may not work properly without the cascade file.")
                # Create a placeholder path - detector will need manual cascade file
                pass
        
        return cascade_path
    
    def preprocess_image(self, image):
        """
        Preprocess image for car detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def detect_cars(self, image):
        """
        Detect cars in the image using Haar Cascade
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected car bounding boxes as (x, y, w, h) tuples
        """
        # Preprocess the image
        gray = self.preprocess_image(image)
        
        # Detect cars
        cars = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return cars
    
    def draw_detections(self, image, cars, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected cars
        
        Args:
            image: Input image to draw on
            cars: List of car detections from detect_cars()
            color: Box color in BGR format
            thickness: Box line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        
        for (x, y, w, h) in cars:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(result, 'Car', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def detect_and_draw(self, image):
        """
        Detect cars and draw bounding boxes in one step
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (result_image, detections)
        """
        cars = self.detect_cars(image)
        result = self.draw_detections(image, cars)
        return result, cars
