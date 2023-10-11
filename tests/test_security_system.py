import unittest
import cv2
import sys
sys.path.insert(1,"..\\SecureFace-Branch")
from app.security_system import SecuritySystem
import pytesseract

class TestSecuritySystem(unittest.TestCase):
    def setUp(self):
        # Initialize the SecuritySystem with a test model and cascade classifier
        self.security_system = SecuritySystem(algorithm = "LBPH", model_path = "tests/test_models/test_model_LBPH.xml", cascade_path = "tests/haar_face.xml")

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
        pytesseract.pytesseract.tesseract_cmd= r'tests/Tesseract-OCR/tesseract.exe'
        
        image_path = "tests/test_images/test_face.png"
        image = cv2.imread(image_path)
        image = cv2.resize(image,(400,600))
        face,processed_image = self.security_system.recognize_face(image)
        x,y,w,h = face
        label_image = processed_image[y-35:y-5,x:x+200]
        label_image = cv2.cvtColor(label_image,cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(label_image)
        extracted_text = str.split(extracted_text,"\n")
        self.assertTrue(extracted_text[0][:4]=="Name" and extracted_text[1][:20]=="Authorization Status")

if __name__ == '__main__':
    unittest.main()
