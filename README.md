# Surveillance and Security System

This repository contains the code for a surveillance and security system using ESP32-CAMs to capture video, perform action recognition, and detect and track individuals exhibiting abnormal behavior.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)


## Overview

This project is divided into two main phases:

1. **Action Recognition**: Captures video frames from multiple ESP32-CAMs, processes them to recognize actions using an EfficientNet-B0 model for feature extraction and an LSTM for classification. If an abnormal action is detected, the system saves the last 5 minutes of video from all cameras.

2. **Face Detection and Summarization**: Detects faces in the saved videos using YOLOv8, extracts face embeddings with a ResNet model, and tracks the individual(s) involved in the abnormal behavior across multiple cameras. A summary video is then created for each individual.

## Features

- Real-time video capture from multiple ESP32-CAMs.
- Action recognition to detect abnormal behavior.
- Face detection and embedding extraction.
- Tracking and summarization of individuals across multiple cameras.

## Project Structure

```plaintext
surveillance_system/
│
├── FrameCapture/
│   ├── __init__.py
│   └── capture.py
│
├── ActionRecognition/
│   ├── __init__.py
|   ├── model.py
│   └── recognizer.py
│
├── FaceDetection  /
│   ├── __init__.py
│   └── FaceDetection.py.py
│
├── FaceRecognition/
│   ├── __init__.py
│   └── represent.py
│
├── FaceVerification/
│   ├── __init__.py
│   └── verify.py
│
├── Summarization/
│   ├── __init__.py
│   └── summary.py
│
├── Shared/
│   ├── __init__.py
│   └── utils.py
│
└── main.py
```


## Installation
1. Clone the repository:
```plaintext
git clone https://github.com/yourusername/surveillance_system.git
cd surveillance_system
```
2. Create a virtual environment and activate it:
```plaintext
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install the required dependencies:
```plaintext
pip install -r requirements.txt
```


## Usage
1. Set up the ESP32-CAMs: Ensure your ESP32-CAMs are configured and connected to your network.
2. Start the frame capture module:
```plaintext
python main.py
```
3. Monitor the output: The system will log the process of capturing frames, recognizing actions, detecting faces, and summarizing movements.
4. Review the results: Check the BASE_OUTPUT_FOLDER directory for saved frames and summary videos.

   
## Dependencies
- OpenCV: For image and video processing.
- NumPy: For numerical operations.
- WebSockets: For communication between ESP32-CAMs and the server.
- asyncio: For asynchronous programming.
- aiofiles: For asynchronous file operations.
- concurrent.futures: For concurrent programming.
- ZeroConf: For network service discovery.
- PyTorch: For deep learning models.
- EfficientNet: For feature extraction in action recognition.
- YOLOv8: For face detection.
- ResNet: For face embedding extraction.

  
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
1. Fork the repository
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.
