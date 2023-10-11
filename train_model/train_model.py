import cv2
import os
import numpy as np
import csv

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

    # TODO 1
    # Load training data
    # Walk through each sub-folder
    # Assume each sub-folder is a different person
    # Append each image to the training set
    # Append the corresponding label to the labels list
    # Add the person's name to the label-to-name dictionary using the label as the key
    # Your code goes here
    data_dir = 'train_data'

    for label, person_name in enumerate(os.listdir(data_dir)):
        label_to_name[label] = person_name
        person_dir = os.path.join(data_dir, person_name)

        for filename in os.listdir(person_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')): 
                image_path = os.path.join(person_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
                
                if image is not None:
                    faces.append(image)
                    labels.append(label)

    # Train the face recognition model
    face_recognizer.train(faces, np.array(labels))

    # Save the trained model to a .xml file in the app/trained_models folder
    # The name of the file should be according to example presented in docs

    # getting complete path
    model_filename = os.path.join('..', 'app', 'trained_models', f'trained_face_model_{algorithm}.xml')

    face_recognizer.save(model_filename)

    # Save label-to-name mapping to a CSV file
    label_to_name_filename = os.path.join('..', 'app', 'label_to_name.csv')
    with open(label_to_name_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for label, name in label_to_name.items():
            writer.writerow([label, name])


    # TODO 2
    # Print a summary of the training
    # Summary could include number of users trained, number of images per user, etc.
    print(f'Training complete. Number of users trained: {len(set(labels))}')

if __name__ == '__main__':
    # Specify the algorithm ('LBPH', 'Eigen', or 'Fisher')
    selected_algorithm = 'LBPH'
    train_face_recognition_model(selected_algorithm)
