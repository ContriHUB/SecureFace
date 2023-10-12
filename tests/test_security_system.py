import unittest
import cv2
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

        # Call the log_access_attempt method
        self.security_system.log_access_attempt(person_name, is_authorized)

        # Check if the access_logs.log file contains the log entry
        with open('app/access_logs.log', 'r') as log_file:
            log_contents = log_file.read()
            self.assertIn(f'Person: {person_name}, Authorized: {is_authorized}', log_contents)

    def test_face_detection(self):
        # Ensure face detection works correctly
        image_path = 'tests/test_images/face.jpg'
        image = cv2.imread(image_path)

        # Call the recognize_face method
        processed_image = self.security_system.recognize_face(image)

        # Check if faces were detected in the processed image
        self.assertTrue(len(processed_image) > 0)

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
