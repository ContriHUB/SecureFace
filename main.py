import os
import cv2
from train_model.train_model import train_face_recognition_model
from app.security_system import SecuritySystem

def train_new_model():
    # Specify the algorithm ('LBPH', 'Eigen', or 'Fisher') for training
    selected_algorithm = input("Select the algorithm for training (LBPH/Eigen/Fisher): ").strip()
    train_face_recognition_model(selected_algorithm)

def load_existing_model(algorithm):
    # Load a pre-trained model based on the specified algorithm
    model_path = f'app/trained_models/trained_face_model_{algorithm}.xml'
    return SecuritySystem(algorithm, model_path)

def real_time_detection(security_system):
    # TODO 1
    capture_vid = cv2.VideoCapture(0)  
    # Now perform face recognition on the frame and while the condition is true, we will capture the image
    while True:
        ret, frame = capture_vid.read()
        # Perform face recognition on the frame
        recognized_faces = security_system.recognize_face(frame)
        cv2.imshow('Real-time Face Recognition', frame)
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
    #Release the video capture object and close the OpenCV window
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
