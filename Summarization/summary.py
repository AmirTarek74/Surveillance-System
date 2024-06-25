from FaceRecognition import FaceRepresention
from FaceDetection import FaceDetector
from FaceVerification import FaceVerification
import os 
import cv2

class Summary:

    def summarize(self,OUTPUT_DIR,videos_dir,ABNORMAL_Frame,detector_model,represent_model,device):

        detector = FaceDetector()
        repsention = FaceRepresention()
        verification = FaceVerification()
        cam1_videos_path = os.path.join(videos_dir,'cam1')
        cam2_videos_path = os.path.join(videos_dir,'cam2')
        cam3_videos_path = os.path.join(videos_dir,'cam3')

        cam1_videos = os.listdir(cam1_videos_path)
        cam2_videos = os.listdir(cam2_videos_path)
        cam3_videos = os.listdir(cam3_videos_path)
        
        wanted_faces = detector.detect_faces(detector_model,ABNORMAL_Frame)
        embddings = [repsention.represent(represent_model,device,face) for face in wanted_faces]

        distance_function = 'cosine'
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = 480,480
        video_writers = []
        for i in range(1,len(wanted_faces)+1):
            writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR,f'Video_Summary for Person {i}.mp4'), fourcc, fps, (width, height))
            video_writers.append(writer)
        
        for v in range(len(cam1_videos)):
            cap1 = cv2.VideoCapture(os.path.join(cam1_videos_path,cam1_videos[v]))
            cap2 = cv2.VideoCapture(os.path.join(cam2_videos_path,cam2_videos[v]))
            cap3 = cv2.VideoCapture(os.path.join(cam3_videos_path,cam3_videos[v]))

            while cap1.isOpened():
                ret1,frame1 = cap1.read()
                ret2,frame2 = cap2.read()
                ret3,frame3 = cap3.read()
                if not ret1 or not ret2 or not ret3:
                    break
                
                faces1 = detector.detect_faces(detector_model,frame1)
                embddings1 = [repsention.represent(represent_model,device,face) for face in faces1]
                faces2 = detector.detect_faces(detector_model,frame2)
                embddings2 = [repsention.represent(represent_model,device,face) for face in faces2]
                faces3 = detector.detect_faces(detector_model,frame3)
                embddings3 = [repsention.represent(represent_model,device,face) for face in faces3]


                for idx,true_embdding  in enumerate(embddings):
                    for embdding in embddings1:
                        if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                            print(f'Person {idx+1} found')
                            frame1 = cv2.resize(frame1,(height,width))
                            video_writers[idx].write(frame1)
                    for embdding in embddings2:
                        if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                            print(f'Person {idx+1} found')
                            frame2 = cv2.resize(frame2,(height,width))
                            video_writers[idx].write(frame2)
                    
                    for embdding in embddings3:
                        if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                            print(f'Person {idx+1} found')
                            frame3 = cv2.resize(frame3,(height,width))
                            video_writers[idx].write(frame3)
            
            cap1.release()
            cap2.release()
            cap3.release()
        for writer in video_writers:
            writer.release()
        for i in range(1,len(wanted_faces)+1):
            print(f"Summary for Person {i} is saved at {os.path.join(OUTPUT_DIR,f'Video_Summary for Person {i}')}.mp4 ")
        