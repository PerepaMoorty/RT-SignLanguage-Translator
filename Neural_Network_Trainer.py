from Camera_Loader import Frame_Reader, Pre_Process_Frame
from Dataset_Loader import Show_Tensor_Shape
from Dataset_Loader import train_data_tensor, train_label_tensor
from Dataset_Loader import test_data_tensor, test_label_tensor
from Neural_Network_Definition import *
import os
import time
import torch

MODEL_COUNT = 10
models = []
models_accuracy = []

def Trainer():
    start = time.time()

    for count in range(MODEL_COUNT):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Model Number: {count}\n')

        models.append(Neural_Network(epochs=10, learning_rate=0.001, batch_size=64))
        models[count].Train_Model(train_data_tensor, train_label_tensor)
        models_accuracy.append(models[count].Test_Model(test_data_tensor, test_label_tensor))

    end = time.time()
    print(f'The Total training time: {end - start:.4f} seconds\n')

    # Checking which model got the highest accuracy
    model_index = models_accuracy.index(max(models_accuracy))

    print(f'The most accurate model\'s index is: {model_index}\n')
    print(f'The accuracy is: {models_accuracy[model_index]:.4f}')
    print(f'The retested accuracy is: {models[model_index].Test_Model(test_data_tensor, test_label_tensor):.4f}')

    # Saving the highest accuracy model
    torch.save(models[model_index].model.state_dict(), 'Trained_Model.pth')  # Save the model state
    print("Model has been saved as 'Trained_Model.pth'.")  # Confirmation message