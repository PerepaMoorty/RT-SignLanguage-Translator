import torch
from Neural_Network_Definition import Neural_Network

def load_and_infer():
    # Load the model
    model = Neural_Network(epochs=10, learning_rate=0.001, batch_size=64)  # Create an instance
    model.model.load_state_dict(torch.load('Trained_Model.pth'))  # Load the saved state dict
    print("Model has been loaded from 'Trained_Model.pth'.")  # Confirmation message
    model.model.eval()  # Set the model to evaluation mode