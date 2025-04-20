import cv2
import numpy as np

class HumanDetector:
    def __init__(self):
        # Threshold values from the original implementation
        self.THRESH_VALUE = 120
        self.MAX_THRESH_VALUE = 255
        self.MIN_CNTR_HUMN_AREA = 8
        self.MAX_CNTR_HUMN_AREA = 350
        
    def detect(self, frame):
        """
        Detect humans in a thermal image frame using simple thresholding and contour detection
        Args:
            frame: CV2 image (thermal) in mono8 format
        Returns:
            List of (x, y, w, h) tuples for detected humans
        """
        # Binary thresholding stage
        _, thresh = cv2.threshold(frame, self.THRESH_VALUE, self.MAX_THRESH_VALUE, cv2.THRESH_BINARY)

        # Contour detection stage
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store human detections
        detections = []
        
        # Calculate areas of the detected contours
        for contour in contours:
            area = cv2.contourArea(contour)
            # Human blob filtration stage
            if self.MIN_CNTR_HUMN_AREA <= area <= self.MAX_CNTR_HUMN_AREA:
                # Fitting bounding boxes over our contours of interest (humans)
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, w, h))
                
        return detections
