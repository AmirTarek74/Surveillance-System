# Surveillance and Security System

This repository contains the code for a surveillance and security system using ESP32-CAMs to capture video, perform action recognition, and detect and track individuals exhibiting abnormal behavior.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

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


