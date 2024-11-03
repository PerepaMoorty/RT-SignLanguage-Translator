import cv2 as cv
import torch
from Camera_Loader import Frame_Reader, Pre_Process_Frame
from load_model import load_and_infer  # Load the model

def start_sign_detection():
    # Load the model first
    model = load_and_infer()  # Load the model and return it

    # Start the video capture from the webcam
    camera_capture = cv.VideoCapture(0)

    while True:
        frame = Frame_Reader()  # Read a frame from the webcam

        if frame is None:
            break

        processed_frame = Pre_Process_Frame(frame)  # Pre-process the frame
        
        # Run inference on the processed frame using the loaded model
        with torch.no_grad():
            prediction = model.model(processed_frame.unsqueeze(0).to(model.device))  # Adjust as necessary
            predicted_class = torch.argmax(prediction, dim=1)  # Get the predicted class

        cv.imshow('Sign Detection', frame)  # Show the frame with detected signs
        print("Predicted class:", predicted_class.cpu().numpy())  # Print the prediction

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera_capture.release()
    cv.destroyAllWindows()