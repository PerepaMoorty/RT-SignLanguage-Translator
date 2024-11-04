import cv2 as cv
import torch

def Frame_Reader(camera_capture):  # Accept camera_capture as an argument
    isTrue, frame = camera_capture.read()  # Reading each Frame
    return frame if isTrue else None

def Pre_Process_Frame(frame):
    # Pre-Processing each frame
    processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    processed_frame = cv.flip(processed_frame, 1)

    # Converting the frame to a PyTorch Tensor
    processed_frame_tensor = torch.tensor(processed_frame, dtype=torch.float32)  # Ensure proper type
    processed_frame_tensor = processed_frame_tensor / 255.0  # Normalize to [0, 1] range
    processed_frame_tensor = processed_frame_tensor.unsqueeze(0)  # Add batch dimension

    return processed_frame_tensor

def Display_Prediction(prediction):
    print(prediction)