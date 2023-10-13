import unittest
import cv2
import os
import numpy as np
from unittest.mock import mock_open, patch
import sys
import os
sys.path.insert(1,"..\\SecureFace-Branch")
from app.security_system import SecuritySystem


class TestSecuritySystem(unittest.TestCase):
    def setUp(self):
        # Initialize the SecuritySystem with a test model and cascade classifier
        self.security_system = SecuritySystem(algorithm = "LBPH", model_path = "tests/test_models/model1.xml", cascade_path = "tests/haar_face.xml")

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
            # validate the log format
            mocked_logger.assert_called_with(f'Person: {person_name}, Authorized: {is_authorized}')

    def test_face_detection(self):
        # Ensure face detection works correctly
        image_path = 'tests/test_images/face.jpg'
        image = cv2.imread(image_path)

        # Call the recognize_face method
        processed_image = self.security_system.recognize_face(image)

        # Check if faces were detected in the processed image
        self.assertTrue(len(processed_image) > 0)
        
        face_cascade = cv2.CascadeClassifier(self.security_system.cascade_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                

    def test_facial_recognition(self):
        # TODO: Implement this test
        # Ensure facial recognition works correctly
        # Call the recognize_face method
        # Check if recognized face labels are present in the processed image
        image_directory = "tests/test_images"
        files = os.listdir(image_directory)

        # Filtering out only the files (excluding directories)
        files = [file for file in files if os.path.isfile(os.path.join(image_directory, file))]

        # Checking accuracy on each image to calculate net accuracy
        prediction_accuracy = 0
        for file in files:
            person_name_ground_truth = file[:-6]
            auth_status_ground_truth = self.security_system.is_person_authorized(person_name_ground_truth)
            image_path = os.path.join(image_directory,file)
            image = cv2.imread(image_path)
            person_name, auth_status,processed_image = self.security_system.recognize_face(image)
            if(person_name == person_name_ground_truth and auth_status == auth_status_ground_truth):
                prediction_accuracy += 1
        prediction_accuracy /= len(files)
        # Since we are iterating over all images in the file, this can be scaled larger if needed.
        self.assertTrue(prediction_accuracy>=0.6)

if __name__ == '__main__':
    unittest.main()
