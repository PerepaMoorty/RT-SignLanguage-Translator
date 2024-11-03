import cv2 as cv
import torch

# Getting the Video Capture through the Webcam
camera_capture = cv.VideoCapture(0)

def Frame_Reader():
    isTrue, frame = camera_capture.read()   # Reading each Frame
    if not isTrue:
        return None  # Return None if the frame isn't captured properly
    return frame

def Pre_Process_Frame(frame):
    # Pre-Processing each frame
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    processed_frame = cv.flip(processed_frame, 1)  # Flip the frame horizontally

    # Resize the frame to 28x28
    processed_frame = cv.resize(processed_frame, (28, 28))

    # Normalize the pixel values (0-1 range)
    processed_frame = processed_frame / 255.0

    # Convert the frame to a PyTorch Tensor and add a batch dimension
    processed_frame_tensor = torch.tensor(processed_frame, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    return processed_frame_tensor  # Return the pre-processed tensor frame