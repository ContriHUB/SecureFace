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
    # Capture video from a camera (you can adjust the video source)
    
        # Perform face recognition on the frame

        # Press 'q' to quit

    # Release the video capture object and close the window
    pass

def batch_processing(security_system):
    val_data_dir = 'val_data'
    predicted_val_data_dir = 'predicted_val_data'

    # TODO 2
    # Create the 'predicted_val_data' directory if it doesn't exist
    
    if(not(os.path.exists(predicted_val_data_dir) and os.path.isdir(predicted_val_data_dir))):
        os.makedirs(predicted_val_data_dir)
        
    # Walk through the 'val_data' directory and perform face recognition on each image
    # Get the relative path from 'val_data' to the image
    # Get the directory part (excluding the file name)
    # Create the corresponding directory structure in 'predicted_val_data'
    # Create the output path for the processed image
    # Save the processed image in the corresponding directory
    
    for directory in os.listdir(val_data_dir):
        dir_path = os.path.join(val_data_dir,directory)
        
        predicted_path = os.path.join(predicted_val_data_dir,directory)
        if(not(os.path.exists(predicted_path) and os.path.isdir(predicted_path))):
            os.makedirs(predicted_path)
        
        for file in os.listdir(dir_path):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(dir_path,file)
                image = cv2.imread(image_path)
                person_name, auth_status, predicted_image = security_system.recognize_face(image)
                cv2.imwrite(os.path.join(predicted_path,file),predicted_image)

    print("Batch Processing Successful.\n")
    
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
