import os, time

# Program Runtime Info 
start_time = time.time()
os.system('cls' if os.name == 'nt' else 'clear')

from Camera_Loader import Frame_Reader, Pre_Process_Frame
from Dataset_Loader import Show_Tensor_Shape
from Dataset_Loader import train_data_tensor, train_label_tensor
from Dataset_Loader import test_data_tensor, test_label_tensor
from Neural_Network_Definition import *

model = Neural_Network(epochs=10, learning_rate=0.001, batch_size=64)

model.Train_Model(train_data_tensor, train_label_tensor)
print('\n\nAccuracy: ', model.Test_Model(test_data_tensor, test_label_tensor))

# Program Runtime Info 
end_time = time.time()
print(f'\nTotal Runtime: {end_time - start_time :.4f} seconds')