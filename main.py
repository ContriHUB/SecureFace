import os
import cv2
from train_model.train_model import train_face_recognition_model
from app.security_system import SecuritySystem

def train_new_model():
    # Specify the algorithm ('LBPH', 'Eigen', or 'Fisher') for training
    selected_algorithm = input("Select the algorithm for training (LBPH/Eigen/Fisher): ").strip()
    train_face_recognition_model(selected_algorithm)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def load_existing_model(algorithm):
    # Load a pre-trained model based on the specified algorithm
    model_path = f'app/trained_models/trained_face_model_{algorithm}.xml'
    return SecuritySystem(algorithm, model_path)

#new
#recognizing using LBPG algoq
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#extracting informations
model_file = 'app/trained_models/trained_face_model_LBPH.xml'  # Update with the appropriate model file

try:
    face_recognizer.read(model_file)
except cv2.error as e:
    print(f"We are  unable to open model file '{model_file}': {e}")
    exit()
label_to_name = {}
label_to_name_file = 'app/label_to_name.csv'  # you should update with the appropriate label-to-name file
try:
    with open(label_to_name_file) as f:
        for line in f:
            label, name = line.strip().split(',')
            label_to_name[int(label)] = name
except IOError:
    print(f"Unable to open label-to-name file '{label_to_name_file}'")
    exit()
    
def recognize_face_dummy(frame):
 
    # recognized_faces = [(50, 50, 100, 100), (200, 200, 80, 80)]  
    # return recognized_faces
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
         label, confidence = face_recognizer.predict(gray_image[y:y+h, x:x+w])
    
         if confidence < 50:
            recognized_label = label_to_name[label]
         else:
             recognized_label = "Unknown"
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

         cv2.putText(frame, recognized_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
    return faces


def real_time_detection(security_system):
    
    capture_vid = cv2.VideoCapture(0)  
    # Now perform face recognition on the frame
    #Now while the condition is true , we will capture the image
    while True:
        ret, frame = capture_vid.read()
        # Perform face recognition on the frame
        recognized_faces = recognize_face_dummy(frame)

       
        for (x, y, w, h) in recognized_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cv2.imshow('Real-time Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
        # Press 'q' to quit
       

    # Release the video capture object and close the OpenCV window
    capture_vid.release()
    cv2.destroyAllWindows()

    pass

def batch_processing(security_system):
    val_data_dir = 'val_data'
    predicted_val_data_dir = 'predicted_val_data'

    # TODO 2
    # Create the 'predicted_val_data' directory if it doesn't exist
    # Walk through the 'val_data' directory and perform face recognition on each image
    # Get the relative path from 'val_data' to the image
    # Get the directory part (excluding the file name)
    # Create the corresponding directory structure in 'predicted_val_data'
    # Create the output path for the processed image
    # Save the processed image in the corresponding directory
    pass

def main():
    print("Welcome to the SecureFace Access Control System!")

    while True:
        print("\nChoose an option:")
        print("1. Train a new face recognition model")
        print("2. Load an existing model and run the system")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            train_new_model()
        elif choice == '2':
            algorithm = input("Enter the algorithm used for training (LBPH/Eigen/Fisher): ").strip()
            security_system = load_existing_model(algorithm)

            print("\nChoose an option:")
            print("1. Real-time Detection")
            print("2. Batch Processing of Validation Data")
            print("3. Return to Main Menu")
            sub_choice = input("Enter your choice (1/2/3): ").strip()

            if sub_choice == '1':
                real_time_detection(security_system)
              
            elif sub_choice == '2':
                batch_processing(security_system)
            elif sub_choice == '3':
                continue
            else:
                print("Invalid choice. Returning to the main menu.")
        elif choice == '3':
            print("Exiting the SecureFace Access Control System")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == '__main__':
    main()