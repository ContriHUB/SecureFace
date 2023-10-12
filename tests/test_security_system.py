import unittest
import cv2
import os
import numpy as np
from unittest.mock import mock_open, patch
from app.security_system import SecuritySystem


class TestSecuritySystem(unittest.TestCase):
    def setUp(self):
        # Initialize the SecuritySystem with a test model and cascade classifier
        self.security_system = SecuritySystem(algorithm='LBPH', model_path='test_model.xml', cascade_path='test_cascade.xml')

    def test_access_logging(self):
        # Ensure access attempts are logged correctly
        person_name = 'Test User'
        is_authorized = True
        #check whether file exists at correct path
        log_file_path = 'app/access_logs.log'
        self.assertTrue(os.path.exists(log_file_path))
        
        #Clean up the log file before running the test
        with open(log_file_path, 'w') as log_file:
            log_file.write("")
        
        with patch('logging.Logger.info') as mocked_logger:
            self.security_system.log_access_attempt(person_name, is_authorized)
            
            mocked_logger.assert_called_with(f'Person: {person_name}, Authorized: {is_authorized}')

    def test_face_detection(self):
        # Ensure face detection works correctly
        image_path = 'tests/test_images/face.jpg'
        image = cv2.imread(image_path)

        # Call the recognize_face method
        processed_image = self.security_system.recognize_face(image)

        # Check if faces were detected in the processed image
        self.assertTrue(len(processed_image) > 0)
        
        # Check if the image is not entirely black
        self.assertTrue(np.mean(processed_image) > 0)  
        
        face_cascade = cv2.CascadeClassifier(self.security_system.cascade_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        #Check for Detected Faces Being Non-Overlapping
        for i in range(len(faces)):
            x1, y1, w1, h1 = faces[i]
            for j in range(i + 1, len(faces)):
                x2, y2, w2, h2 = faces[j]
                self.assertFalse(x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)
                

    def test_facial_recognition(self):
        # TODO: Implement this test
        # Ensure facial recognition works correctly
        # Call the recognize_face method
        # Check if recognized face labels are present in the processed image
        pass

if __name__ == '__main__':
    unittest.main()
