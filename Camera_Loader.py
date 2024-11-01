import cv2 as cv

# Getting the Video Capture through the Webcam
camera_capture = cv.VideoCapture(0)

def Frame_Reader():
    isTrue, frame = camera_capture.read()   # Reading each Frame
    return isTrue, frame

def Pre_Process_Frame(frame):
    processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    processed_frame = cv.flip(processed_frame, 1)
    return processed_frame