# SecureFace Access Control System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.1-blue.svg)](https://opencv.org/releases/)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-green.svg)

The SecureFace Access Control System is an open-source project designed to provide facial recognition-based access control for your applications. It allows you to recognize authorized individuals and log access attempts. With this system, you can train and deploy face recognition models, perform real-time face recognition, and process batches of images for authentication.

The SecureFace Access Control System is a small-scale project designed to enhance security and streamline access control in a office environment using facial recognition technology. This system replaces traditional access cards with face detection and recognition to provide a convenient and secure way to grant access to authorized personnel.

![System Demo](demo.gif)

## Features

- Train face recognition models using different algorithms (LBPH, Eigen, Fisher).
- Real-time face recognition with webcam support.
- Batch processing of validation data to identify authorized individuals.
- Log access attempts and unauthorized access in access_logs.log.
- Send Firebase notifications for unauthorized access (requires Firebase setup).

## Project Structure

```
SecureFace
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE.md
├── README.md
├── app
│   ├── access_logs.log
│   ├── authorized_persons.csv
│   ├── haar_face.xml
│   ├── label_to_name.csv
│   ├── security_system.py
│   └── trained_models
│       ├── trained_face_model_Eigen.xml
│       ├── trained_face_model_Fisher.xml
│       └── trained_face_model_LBPH.xml
├── docs
│   ├── project_description.md
│   └── technical-doc.md
├── main.py
├── predicted_val_data
│   ├── ben_afflek
│   │   ├── 1.jpg
│   │   ├── ...
│   │   └── n.jpg
│   │
│   │   ...
│   │
│   └── [User Name]
│       ├── 1.jpg
│   │   ├── ...
│   │   └── n.jpg
├── requirements.txt
├── tests
│   ├── haar_face.xml
│   └── test_security_system.py
│   └── test_images
│       ├── Unknown_1.png
│       ├── Unknown_2.png
│       └── Unknown_3.png
├── train_model
│   ├── train_data
│   │   ├── Ben Afflek
│   │   │   ├── 1.jpg
│   │   │   ├── ...
│   │   │   └── n.jpg
│   │   │
│   │   │   ...
│   │   │
│   │   ├── [User Name]
│   │   │   ├── 1.jpg
│   │   │   ├── ...
│   │   │   └── n.jpg
│   │   │
│   │   │   ...
│   │   │
│   │   └── Mindy Kaling
│   │       ├── 1.jpg
│   │   │   ├── ...
│   │   │   └── n.jpg
│   └── train_model.py
└── val_data
    ├── ben_afflek
    │   ├── 1.jpg
    │   ├── ...
    │   └── n.jpg
    │
    │   ...
    │
    └── [User Name]
        ├── 1.jpg
        ├── ...
        └── n.jpg
```

## Getting Started

1. Install the required dependencies by running:

   ```
   pip install -r requirements.txt
   ```

2. Train a new face recognition model or use an existing one.

3. Start the SecureFace Access Control System by running `main.py`.

4. Choose between real-time face detection or batch processing of validation data.

5. Monitor access attempts and unauthorized access in `app/access_logs.log`.

## Usage

- To train a new model, select the training option in the main menu.
- To use an existing model, load it in the main menu and choose the desired operation.
- Real-time detection captures video from a camera, recognizing authorized individuals.
- Batch processing processes images from the `val_data` folder and saves them in `predicted_val_data`.

## Testing

Testing is an essential part of maintaining the reliability and correctness of the SecureFace Access Control System. We use Python's `unittest` framework to write and run tests.

### Running Tests

To run the tests, navigate to the project's root directory and execute the following command:

```bash
python -m unittest discover tests
```

This command will discover and run all the test cases located in the tests directory. Ensure that you have set up the necessary test images and configurations before running the tests.

### Writing Tests
We encourage contributions to our test suite to cover various aspects of the system's functionality. If you'd like to contribute tests, please refer to the tests directory for examples of existing test cases.

For more details on writing and contributing tests, see our [Contribution Guidelines](CONTRIBUTING.md).

## Documentation

For detailed documentation, please refer to the [project_description.md](docs/project_description.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Authors

- [Sanskar Omar](https://github.com/sanskaromar)

## Acknowledgments

- [Contributor Names]
- [Sahil Dutta] (https://github.com/Aqua-16)
