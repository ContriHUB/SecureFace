import cv2
import os
import numpy as np
import csv
from tabulate import tabulate

# Define the paths to the training dataset
train_data_dir = 'train_model/train_data'

def train_face_recognition_model(algorithm='LBPH'):
    # Initialize face recognition model based on the specified algorithm
    if algorithm == 'LBPH':
        face_recognizer = cv2.face_LBPHFaceRecognizer.create()
    elif algorithm == 'Eigen':
        face_recognizer = cv2.face_EigenFaceRecognizer.create()
    elif algorithm == 'Fisher':
        face_recognizer = cv2.face_FisherFaceRecognizer.create()
    else:
        raise ValueError("Invalid algorithm specified")

    faces = []
    labels = []

    # Create a dictionary to map label (numeric) to person's name
    label_to_name = {}
    label_count = {}
    
    
    # Load training data
    for label, person_name in enumerate(os.listdir(train_data_dir)):
        label_to_name[label] = person_name
        # Get directory path using person's name
        person_dir = os.path.join(train_data_dir, person_name)

        for filename in os.listdir(person_dir):
            # Check for images in the file and append them together
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')): 
                image_path = os.path.join(person_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Resizing image as Fisher and Eigen require images of equal size
                    image = cv2.resize(image,(640,480))
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    faces.append(image)
                    labels.append(label)
                    label_count[label] = label_count.get(label,0)+1

    # Train the face recognition model
    face_recognizer.train(np.array(faces), np.array(labels))

    # Save the trained model to a .xml file in the app/trained_models folder

    # Getting complete path
    model_filename = os.path.join('app', 'trained_models', f'trained_face_model_{algorithm}.xml')

    face_recognizer.save(model_filename)

    # Save label-to-name mapping to a CSV file
    label_to_name_filename = os.path.join('app', 'label_to_name.csv')
    with open(label_to_name_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for label, person_name in label_to_name.items():
            writer.writerow([label, person_name])

    # Print a summary of the training
    
    print("Success! Training Completed.\n")
    print(f"Algorithm Used:{algorithm}")
    print(f'Number of users trained: {len(set(labels))}')
    headers = ["Label","Name","Number of Images Trained"]
    data = [];
    for label, count in label_count.items():
        person_name = label_to_name[label]
        row = [label,person_name,count];
        data.append(row)
    table = tabulate(data,headers,tablefmt = "pretty")
    print(table)
    
if __name__ == '__main__':
    # Specify the algorithm ('LBPH', 'Eigen', or 'Fisher')
    selected_algorithm = 'LBPH'
    train_face_recognition_model(selected_algorithm)
