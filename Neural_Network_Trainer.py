import os, time, torch
import cv2 as cv
from Camera_Loader import Frame_Reader, Pre_Process_Frame, Display_Prediction
from Dataset_Loader import train_data_tensor, train_label_tensor, test_data_tensor, test_label_tensor
from Neural_Network_Definition import Neural_Network

MODEL_COUNT = 25
models = []
models_accuracy = []

def Train_And_Save():
    start = time.time()

    for count in range(MODEL_COUNT):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Model Number: {count + 1}\n')

        models.append(Neural_Network(epochs=10, learning_rate=0.001, batch_size=64))
        models[count].Train_Model(train_data_tensor, train_label_tensor)
        models_accuracy.append(models[count].Test_Model(test_data_tensor, test_label_tensor))

    end = time.time()
    print(f'The Total training time: {end - start:.4f} seconds\n')

    # Checking which model got the highest accuracy
    model_index = models_accuracy.index(max(models_accuracy))

    # Re-Testing the Accurate Model
    print(f'The most accurate model\'s index is: {model_index}\n')
    print(f'The accuracy is: {models_accuracy[model_index]:.4f}')
    print(f'The retested accuracy is: {models[model_index].Test_Model(test_data_tensor, test_label_tensor):.4f}')

    # Save the model after training
    torch.save(models[model_index].model.state_dict(), 'Trained_Model.pth')
    print(f'\nModel {model_index + 1} has been saved.')
        
def Load_And_Eval():
    # Loading the Model
    model = Neural_Network(10, 0.001, 64)
    model.model.load_state_dict(torch.load('Trained_Model.pth')) if os.path.exists('Trained_Model.pth') else None
    
    # Error check if Trained Model doesn't exist
    if model is None: 
        print("Trained Model doesn't exist! Training Models now!")
        Train_And_Save()      
        Load_And_Eval()
        return
    
    # Setting the Model into Evaluation mode
    model.model.eval()
    
    # Getting Data from Camera
    capture = cv.VideoCapture(0)
    prediction = Neural_Network.Evaluate(model, Pre_Process_Frame(Frame_Reader(capture)))
    
    while True:
        frame = Frame_Reader(capture)  # Read frame from the camera
        if frame is None:
            break  # Break if there is an issue with the camera

        # Preprocess the frame for the model
        processed_frame = Pre_Process_Frame(frame)
        
        # Display the prediction
        prediction = model.Evaluate(processed_frame)
        Display_Prediction(prediction)


        # Display the camera feed
        cv.imshow('Camera Feed', frame)  # Show the camera feed window

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()