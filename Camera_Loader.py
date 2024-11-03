import cv2 as cv
import torch

# Getting the Video Capture through the Webcam
camera_capture = cv.VideoCapture(0)

def Frame_Reader():
    isTrue, frame = camera_capture.read()   # Reading each Frame
    if not isTrue: pass
    
    return frame

def Pre_Process_Frame(frame):
    # Pre-Processing each frame
    processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    processed_frame = cv.flip(processed_frame, 1)
    
    # Converting the frame to a PyTorch Tensor
    processed_frame_tensor = torch.tensor(processed_frame)
    
    return processed_frame_tensor