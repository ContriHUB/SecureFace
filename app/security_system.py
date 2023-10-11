import cv2
import os
import csv
import logging
from datetime import datetime

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
        if(model_path==None):
            self.model_path = "trained_models/trained_face_model_LBPH.xml"
        self.face_recognizer.read(self.model_path)
        # Specify the path to the custom Haar Cascade classifier
        if(cascade_path==None):
            self.cascade_path = "haar_face.xml"
        # Load authorized persons from CSV
        self.authorized_persons = self.load_authorized_persons()
        # Load label-to-name mapping from CSV
        self.label_to_name = self.load_label_to_name(algorithm)
        # TODO 2
        # Configure logging
        # Logs are to be made to the access_logs.log file
        # Set the logging level to INFO
        # Use the following format: '%(asctime)s - %(message)s'
        # Your code goes here
        
        logging.basicConfig(
            filename='app/access_logs.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def load_authorized_persons(self):
        authorized_persons = {}
            # TODO 1
            # Load authorized persons from CSV
            # Your code goes here
        with open("authorized_persons.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                authorized_persons[row[0]] = row[1]
        del authorized_persons["Name"]
        return authorized_persons
    
    def load_label_to_name(self, algorithm):
        label_to_name = {}
            # TODO 1
            # Load label-to-name mapping from CSV
            # Your code goes here
        with open("label_to_name.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                label_to_name[row[0]] = row[1]
        return label_to_name

    def recognize_face(self, frame):
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the custom Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(self.cascade_path)

        # TODO 3
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))
        # Iterate through each detected face
        for face in faces:
            # Extract the detected face region
            x,y,w,h = face
            X,Y = x+w,y+h
            image = gray_frame[y:Y, x:X]
            # Perform facial recognition on the detected face
            label, confidence = self.face_recognizer.predict(image)
            person_name = self.get_person_name(label,confidence)
            authorization_status = self.is_person_authorized(person_name)
            if(authorization_status==True):
                color = (0,200,100)
                is_authorized = "True"
            else:
                color = (0,35,200)
                is_authorized = "False"
            # Draw bounding box around the detected face
            l= 20
            t= 5
            
            cv2.rectangle(frame,face,color,1)
            cv2.line(frame,(x,y),(x+l,y),color,t)
            cv2.line(frame,(x,y),(x,y+l),color,t)
            cv2.line(frame,(X-l,y),(X,y),color,t)
            cv2.line(frame,(X,y),(X,y+l),color,t)
            cv2.line(frame,(x,Y-l),(x,Y),color,t)
            cv2.line(frame,(x,Y),(x+l,Y),color,t)
            cv2.line(frame,(X-l,Y),(X,Y),color,t)
            cv2.line(frame,(X,Y-l),(X,Y),color,t)
            # Annotate the frame with the recognized user's name and authorization status
           
            cv2.rectangle(frame, [x,y-35,200,30], color,-1)
            cv2.putText(frame,f"Name: {person_name}",(x+5,y-20),cv2.FONT_HERSHEY_PLAIN,0.8,(30,30,30),1)
            cv2.putText(frame,f"Authorization Status: {authorization_status}",(x+5,y-10),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,0),1)
            # Log access attempt
            self.log_access_attempt(person_name,is_authorized)
        return frame

    def get_person_name(self, label, confidence):
        # TODO 1
            # Return the name of the person based on the label
            # If the confidence is less than 80, return 'Unknown'
            # Your code goes here
            if(confidence<80 or (label not in self.label_to_name)):
                return "Unknown"
            else:
                return self.label_to_name[label]

    def is_person_authorized(self, person_name):
        return self.authorized_persons.get(person_name, False)

    def log_access_attempt(self, person_name, is_authorized):
        # TODO 2
        # Log access attempt in the access_logs.log file
        # Logs should be in the following format:
        # <timestamp> - Person: <person_name>, Authorized: <is_authorized>
        # Your code goes here

        logging.info(f'Person: {person_name}, Authorized: {is_authorized}')
        
        # If unauthorized access, send a notification (you need to implement this)
        if not is_authorized:
            self.send_firebase_notification(person_name)

    def send_firebase_notification(self, person_name):
        # Implement Firebase Cloud Messaging notification
        # You'll need to set up Firebase and integrate Firebase Cloud Messaging.
        pass
