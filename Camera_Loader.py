import cv2 as cv

# Reading through the Webcam
camera_capture = cv.VideoCapture(0)

while True:
    isTrue, frame = camera_capture.read()   # Reading each frame in the Video Capture
    if not isTrue: break
    
    # Pre-Processing each frame
    frame = cv.flip(frame, 1) 
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    
    # Displaying the Image for testing purposes
    cv.imshow('Camera Capture', frame)
    
    # The Key 'D' will close the window
    if cv.waitKey(1) & 0xFF == ord('d'):
        break
    
cv.destroyAllWindows()