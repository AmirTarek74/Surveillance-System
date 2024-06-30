import os 
import cv2

class Summary:

    def summarize(self,OUTPUT_DIR,videos_dir,wanted_faces,detector,repsention,verification):

        
        cam1_videos_path = os.path.join(videos_dir,'cam1')
        cam2_videos_path = os.path.join(videos_dir,'cam2')
        cam3_videos_path = os.path.join(videos_dir,'cam3')

        cam1_videos = os.listdir(cam1_videos_path)
        cam2_videos = os.listdir(cam2_videos_path)
        cam3_videos = os.listdir(cam3_videos_path)
        
       
        embddings = [repsention.represent(face) for face in wanted_faces]

        distance_function = 'cosine'
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = 480,480
        video_writers = []
        for i in range(1,len(wanted_faces)+1):
            writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR,f'Video_Summary for Person {i}.mp4'), fourcc, fps, (width, height))
            video_writers.append(writer)
        #select maximum number  of videos    
        max_length = max([len(cam1_videos),len(cam2_videos),len(cam3_videos)])
        #flags for remaning videos
        VideosLeftCam1 = True
        VideosLeftCam2 = True
        VideosLeftCam3 = True

        for v in range(max_length):
            if len(cam1_videos)>v:
                cap1 = cv2.VideoCapture(os.path.join(cam1_videos_path,cam1_videos[v]))
            else:
                VideosLeftCam1 = False

            if len(cam2_videos)>v:    
                cap2 = cv2.VideoCapture(os.path.join(cam2_videos_path,cam2_videos[v]))
            else:
                VideosLeftCam2 = False

            if len(cam3_videos)>v:
                cap3 = cv2.VideoCapture(os.path.join(cam3_videos_path,cam3_videos[v]))
            else:
                VideosLeftCam3 = False

            #flags to indicate the
            while VideosLeftCam1 or VideosLeftCam2 or VideosLeftCam3:
                if VideosLeftCam1:
                    ret1,frame1 = cap1.read()
                else:
                    ret1 = False
                if VideosLeftCam2:
                    ret2,frame2 = cap2.read()
                else:
                    ret2 = False
                if VideosLeftCam3:
                    ret3,frame3 = cap3.read()
                else:
                    ret3 = False
                #if not ret1 or not ret2 or not ret3:
                 #   break
                if not ret1 and not ret2 and not ret3:
                    break
                if ret1:
                    faces1 = detector.detect_faces(frame1)
                    embddings1 = [repsention.represent(face) for face in faces1] 
                

                if ret2:
                    faces2 = detector.detect_faces(frame2)
                    embddings2 = [repsention.represent(face) for face in faces2]
                
                if ret3:
                    faces3 = detector.detect_faces(frame3)
                    embddings3 = [repsention.represent(face) for face in faces3]
                

                for idx,true_embdding  in enumerate(embddings):
                    if ret1:
                        for embdding in embddings1:
                            if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                                print(f'Person {idx+1} found')
                                frame1 = cv2.resize(frame1,(height,width))
                                video_writers[idx].write(frame1)
                    if ret2:
                        for embdding in embddings2:
                            if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                                print(f'Person {idx+1} found')
                                frame2 = cv2.resize(frame2,(height,width))
                                video_writers[idx].write(frame2)
                    if ret3:
                        for embdding in embddings3:
                            if verification.verify(embdding,true_embdding,metric=distance_function)==1:
                                print(f'Person {idx+1} found')
                                frame3 = cv2.resize(frame3,(height,width))
                                video_writers[idx].write(frame3)
            if VideosLeftCam1:
                cap1.release()
            if VideosLeftCam2:
                cap2.release()
            if VideosLeftCam3:
                cap3.release()
        for writer in video_writers:
            writer.release()
        for i in range(1,len(wanted_faces)+1):
            print(f"Summary for Person {i} is saved at {os.path.join(OUTPUT_DIR,f'Video_Summary for Person {i}')}.mp4 ")
        