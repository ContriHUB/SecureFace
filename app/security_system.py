import cv2
import os
import csv
import logging

class SecuritySystem:
    def __init__(self, algorithm='LBPH', model_path=None, cascade_path=None):
        # Initialize face recognition model based on the specified algorithm
        if algorithm == 'LBPH':
            self.face_recognizer = cv2.face_LBPHFaceRecognizer.create()
        elif algorithm == 'Eigen':
            self.face_recognizer = cv2.face_EigenFaceRecognizer.create()
        elif algorithm == 'Fisher':
            self.face_recognizer = cv2.face_FisherFaceRecognizer.create()
        else:
            raise ValueError("Invalid algorithm specified")

        # TODO 1
        # Load the trained face recognition model

        # Specify the path to the custom Haar Cascade classifier

        # Load authorized persons from CSV
        
        # Load label-to-name mapping from CSV
        
        # TODO 2
        # Configure logging
        # Logs are to be made to the access_logs.log file
        # Set the logging level to INFO
        # Use the following format: '%(asctime)s - %(message)s'
        # Your code goes here

    def load_authorized_persons(self):
        authorized_persons = {}
        # TODO 1
        # Load authorized persons from CSV
        # Your code goes here

        return authorized_persons
    
    def load_label_to_name(self, algorithm):
        label_to_name = {}
        # TODO 1
        # Load label-to-name mapping from CSV
        # Your code goes here

        return label_to_name

    def recognize_face(self, frame):
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the custom Haar Cascade classifier for face detection.Earlier:face_cascade = cv2.CascadeClassifier(self.cascade_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # TODO 3
        # Perform face detection
        # Iterate through each detected face
        # Extract the detected face region
        # Perform facial recognition on the detected face
        # Draw bounding box around the detected face
        # Annotate the frame with the recognized user's name and authorization status
        # Log access attempt
        # Your code goes here
        return frame

    def get_person_name(self, label, confidence):
        # TODO 1
        # Return the name of the person based on the label
        # If the confidence is less than 80, return 'Unknown'
        # Your code goes here
        pass


    def is_person_authorized(self, person_name):
        return self.authorized_persons.get(person_name, False)

    def log_access_attempt(self, person_name, is_authorized):
        # TODO 2
        # Log access attempt in the access_logs.log file
        # Logs should be in the following format:
        # <timestamp> - Person: <person_name>, Authorized: <is_authorized>
        # Your code goes here

        # If unauthorized access, send a notification (you need to implement this)
        if not is_authorized:
            self.send_firebase_notification(person_name)

    def send_firebase_notification(self, person_name):
        # Implement Firebase Cloud Messaging notification
        # You'll need to set up Firebase and integrate Firebase Cloud Messaging.
        pass
