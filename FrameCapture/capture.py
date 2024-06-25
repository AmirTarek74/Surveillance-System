import cv2
import numpy as np
import socket
import os
from datetime import datetime
import aiofiles
import asyncio
import concurrent.futures
import websockets
from zeroconf import ServiceInfo, Zeroconf

BASE_OUTPUT_FOLDER = "ESP32-CAM"
if not os.path.exists(BASE_OUTPUT_FOLDER):
    os.makedirs(BASE_OUTPUT_FOLDER)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error getting local IP: {e}")
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

def create_video_writer(output_folder, cam_identifier, frame_size, fps=15):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{cam_identifier}_{timestamp}.avi"
    filepath = os.path.join(output_folder, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
    return video_writer

def process_and_save_frame(frame_data, time, output_folder, cam_identifier, video_writer, frames_count, max_frames_per_video, fps):
    try:
        frame = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"Failed to decode frame from {cam_identifier}")
            return None, None

        timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 255)
        thickness = 1
        org = (10, 30)
        org2 = (10, 50)
        cv2.putText(frame, timestamp_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, cam_identifier, org2, font, font_scale, color, thickness, cv2.LINE_AA)
        if video_writer is None or frames_count >= max_frames_per_video:
            if video_writer:
                video_writer.release()
            frames_count = 0
            frame_size = (frame.shape[1], frame.shape[0])
            video_writer = create_video_writer(BASE_OUTPUT_FOLDER, cam_identifier, frame_size, fps)

        video_writer.write(frame)
        frames_count += 1

        return video_writer, frames_count

    except Exception as e:
        print(f"Error processing frame for {cam_identifier}: {e}")
        return None, None

async def receive_frames(websocket, path):
    cam_ip = websocket.remote_address[0]
    cam_path = path.strip("/")
    cam_identifier = cam_path.split("/")[0]
    print(f"WebSocket connection established from {cam_identifier}")

    output_folder = os.path.join(BASE_OUTPUT_FOLDER, cam_identifier)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fps = 15
    duration = 30
    video_writer = None
    frames_count = 0
    max_frames_per_video = duration * fps

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    try:
        while True:
            frame_data = await websocket.recv()
            
            time = datetime.now()
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
            frame_filename = f"frame_{cam_identifier}_{timestamp}.jpg"
            frame_filepath = os.path.join(output_folder, frame_filename)
            
            async with aiofiles.open(frame_filepath, "wb") as f:
                await f.write(frame_data)
                print(f"Received frame saved: {frame_filename} from {cam_identifier}")
            
            video_writer, frames_count = await asyncio.get_event_loop().run_in_executor(
                executor, process_and_save_frame, frame_data, time, output_folder, cam_identifier, video_writer, frames_count, max_frames_per_video, fps
            )

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed for {cam_identifier}: {e}")
    except Exception as e:
        print(f"Error receiving frame from {cam_identifier}: {e}")
    finally:
        if video_writer:
            video_writer.release()
        print(f"Connection cleanup for {cam_identifier} with IP {cam_ip}")
        await websocket.close()

class FrameCapture:
    def __init__(self):
        self.local_ip = get_local_ip()
        self.start_server = websockets.serve(receive_frames, self.local_ip, 81)
        self.zeroconf = Zeroconf()
        self.service_info = ServiceInfo(
            "_ws._tcp.local.",
            "SERVER._ws._tcp.local.",
            addresses=[socket.inet_aton(self.local_ip)],
            port=81,
            properties={},
        )
        self.zeroconf.register_service(self.service_info)
        print("mDNS service registered")

    def start(self):
        asyncio.get_event_loop().run_until_complete(self.start_server)
        asyncio.get_event_loop().run_forever()

    def stop(self):
        self.zeroconf.unregister_service(self.service_info)
        self.zeroconf.close()
