import os
import threading
import cv2
import numpy as np 
from PIL import Image
from datetime import datetime
import time 
import torch
from torchvision import transforms


from Shared.utils import *
from ActionRecognition import ActionRecognition,ActionRecognizer
from FaceDetection import FaceDetector
from FaceVerification import FaceVerification
from FaceRecognition import FaceRepresention
from Summarization import Summary
from FrameCapture import FrameCapture

def capture_and_process_frames():
    # Start Frames Capture
    frame_capture_thread.start()

    #action reconition model settings
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    num_classes = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_model = ActionRecognition(num_classes).to(device)
    chk = load_checkpoints('Weights/ActionRecognition.pt')
    action_model.load_state_dict(chk['model_state_dict'])
    action_model.eval()
    action_recognizer = ActionRecognizer()
    
    # Face Detection model settings
    face_detector = FaceDetector()  #returns faces in the image
    

    #face represention model settings
    
    face_repsenter = FaceRepresention()

    # face verification settings
    face_verifcation = FaceVerification()
    
    #summarization settings
    videos_dir = '----'
    OUTPUT_DIR = '-----'
    summarizer = Summary()
    
    CAMS = ['cam1','cam2','cam3']
    detected_frames = []
    isAbnormal = False
    while True:
        ###  frames captured from 3 cams
        for cam_identifier in CAMS:
            frames = get_latest_frames(cam_identifier)
            detected_frames = []
            if frames!=-1:
                #if None not in frames:
                trans_frames = [transform(frame).to(device) for frame in frames]
                stacked_frames = torch.stack(trans_frames, dim=0)
                action = action_recognizer.recognize_action(action_model,device,stacked_frames)
                
                if action==1:
                    # write how to handel abnomral here
                    print('Abnormal')
                    isAbnormal = True
                    detected_frames = frames
                    break
                else:
                    print('Normal')
                time.sleep(0.5)
        if isAbnormal:
            break
    wanted_Faces = AbnormalAcionFaces(detected_frames,face_detector,face_repsenter,face_verifcation)
    summarizer.summarize(OUTPUT_DIR,videos_dir,wanted_Faces,face_detector,face_repsenter,face_verifcation) 


  
            
  

def get_latest_frames(cam_identifier, num_frames=20):
    """
    Retrieve the latest frames from the saved directory.
    
    Args:
        cam_identifier (str): Identifier for the camera.
        num_frames (int): Number of latest frames to retrieve.
        
    Returns:
        List of latest frames.
    """
    cam_folder = os.path.join(BASE_OUTPUT_FOLDER, cam_identifier)
    
    if not os.path.exists(cam_folder):
        print(f"Camera folder for {cam_identifier} does not exist.")
        return -1

    # Get list of all frame files
    frame_files = sorted(
        [f for f in os.listdir(cam_folder) if f.endswith('.jpg')],
        key=lambda x: x.split('_')[-1].replace('.jpg', ''), reverse=True)

    # Select the latest frames
    if len(frame_files)>=num_frames:
        latest_frame_files = frame_files[-num_frames:]
    else:
        return -1

    latest_frames = []
    for frame_file in latest_frame_files:
        frame_path = os.path.join(cam_folder, frame_file)
        frame = cv2.imread(frame_path)
        frame = Image.fromarray(frame)
        if frame is not None:
            latest_frames.append(frame)
        else:
            print(f"Failed to read frame: {frame_path}")

    return latest_frames


    

if __name__ == "__main__":
    BASE_OUTPUT_FOLDER = "ESP32-CAM"
    frame_capture = FrameCapture()
    frame_capture_thread = threading.Thread(target=frame_capture.start)
    try:
        capture_and_process_frames()
       
        
    #summary()
    except KeyboardInterrupt: 
        print("Main program Terminated.")
    finally:
        frame_capture.stop()
        frame_capture_thread.join()