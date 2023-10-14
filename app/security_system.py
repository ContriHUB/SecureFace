import cv2
import os
import csv
import logging
from datetime import datetime

class SecuritySystem:
    model_path = None
    cascade_path = None
    def __init__(self, algorithm='LBPH', model_path=None, cascade_path=None):
        
        # Initialize face recognition model based on the specified algorithm
        
        self.algorithm = algorithm
        if self.algorithm == 'LBPH':
            self.face_recognizer = cv2.face_LBPHFaceRecognizer.create()
        elif self.algorithm == 'Eigen':
            self.face_recognizer = cv2.face_EigenFaceRecognizer.create()
        elif self.algorithm == 'Fisher':
            self.face_recognizer = cv2.face_FisherFaceRecognizer.create()
        else:
            raise ValueError("Invalid algorithm specified")

        # Load the trained face recognition model
        
        if(model_path is None or not(os.path.exists(model_path))):
            self.model_path = "app/trained_models/trained_face_model_LBPH.xml"
        else:
            self.model_path = model_path

        self.face_recognizer.read(self.model_path)
        
        if(cascade_path is None or not(os.path.exists(cascade_path))):
            self.cascade_path = "app/haar_face.xml"
        else:
            self.cascade_path = cascade_path
            
        # Load authorized persons from CSV
        
        self.authorized_persons = self.load_authorized_persons()
        
        # Load label-to-name mapping from CSV
        
        self.label_to_name = self.load_label_to_name(algorithm)
        
        # Configure logging
        
        logging.basicConfig(
            filename='app/access_logs.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def load_authorized_persons(self):
        authorized_persons = {}
        
        # Load authorized persons from CSV
        with open("app/authorized_persons.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                authorized_persons[row[0]] = row[1]
        del authorized_persons["Name"]
        return authorized_persons
    
    def load_label_to_name(self, algorithm):
        label_to_name = {}

        # Load label-to-name mapping from CSV
        with open("app/label_to_name.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                label_to_name[row[0]] = row[1]
        return label_to_name

    def recognize_face(self, frame):
        #Resizing image to maintain a standard size in LBPH
        height,width,channels = frame.shape
        ratio = (int)(640/width)
        frame = cv2.resize(frame,(ratio*width,ratio*height))
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(self.cascade_path)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))

        # Initializing values in case no faces are detected
        person_name = "Unknown"
        authorization_status = False
        for face in faces:
            # Extract the detected face region
            x,y,w,h = face
            X,Y = x+w,y+h
            image = gray_frame[y:Y, x:X]
            if(self.algorithm != 'LBPH'):
                # Resizing face images to equal size for Eigen and Fisher
                image = cv2.resize(image,(640,480))
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
            offset= 20
            thickness= 5
            
            cv2.rectangle(frame,face,color,1)
            cv2.line(frame,(x,y),(x+offset,y),color,thickness)
            cv2.line(frame,(x,y),(x,y+offset),color,thickness)
            cv2.line(frame,(X-offset,y),(X,y),color,thickness)
            cv2.line(frame,(X,y),(X,y+offset),color,thickness)
            cv2.line(frame,(x,Y-offset),(x,Y),color,thickness)
            cv2.line(frame,(x,Y),(x+offset,Y),color,thickness)
            cv2.line(frame,(X-offset,Y),(X,Y),color,thickness)
            cv2.line(frame,(X,Y-offset),(X,Y),color,thickness)
            
            # Annotate the frame with the recognized user's name and authorization status
           
            cv2.rectangle(frame, [x,y-35,200,30], color,-1)
            cv2.putText(frame,f"Name: {person_name}",(x+5,y-20),cv2.FONT_HERSHEY_PLAIN,0.8,(30,30,30),1)
            cv2.putText(frame,f"Authorization Status: {authorization_status}",(x+5,y-10),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,0),1)
            # Log access attempt
            self.log_access_attempt(person_name,is_authorized)
        return person_name, authorization_status , frame

    def get_person_name(self, label, confidence):
        
            # Return the name of the person based on the label
            
            if(confidence<80 or (label not in self.label_to_name)):
                return "Unknown"
            else:
                return self.label_to_name[label]

    def is_person_authorized(self, person_name):
        return self.authorized_persons.get(person_name, False)

    def log_access_attempt(self, person_name, is_authorized):
        
        # Log access attempt
        
        logging.info(f'Person: {person_name}, Authorized: {is_authorized}')
        
        # If unauthorized access, send a notification
        if not is_authorized:
            self.send_firebase_notification(person_name)

    def send_firebase_notification(self, person_name):
        # Implement Firebase Cloud Messaging notification
        # You'll need to set up Firebase and integrate Firebase Cloud Messaging.
        pass
