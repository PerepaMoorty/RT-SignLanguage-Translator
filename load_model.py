import torch
from Dataset_Loader import test_data_tensor, test_label_tensor  # Import your test data tensors
from Neural_Network_Definition import Neural_Network

def load_and_infer(processed_frame):
    # Load the model
    model = Neural_Network(epochs=10, learning_rate=0.001, batch_size=64)  # Create an instance
    model.model.load_state_dict(torch.load('Trained_Model.pth'))  # Load the saved state dict
    model.model.eval()  # Set the model to evaluation mode

    # Ensure the input frame is of correct shape
    processed_frame = processed_frame.unsqueeze(0).to(model.device)  # Add batch dimension and move to device

    # Run inference on the processed frame
    with torch.no_grad():  # Disable gradient calculation
        predictions = torch.argmax(model.model(processed_frame), dim=1)  # Get predicted classes

    return predictions.cpu().numpy()  # Return predictions as a NumPy array