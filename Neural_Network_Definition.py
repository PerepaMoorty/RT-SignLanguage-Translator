import torch

class Neural_Network:
    def __init__(self, epochs, learning_rate, batch_size):
        # Defining the Model Format
        self.model = torch.nn.Sequential(
            # First Layer of the Network
            torch.nn.Conv2d(28*28, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            # Second Layer of the Network
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            # Third Layer of the Network
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            # Flattening
            torch.nn.Flatten(),
            
            # Linear Classification - 1
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            
            # Linear Classification - 2
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            
            # Linear Classification - 3 [To Output]
            torch.nn.Linear(128, 24)
        )
        
        # Defining Hyperparams
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Defining the Model's Loss Function and Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def Train_Model(self, train_data_tensor, train_label_tensor):
        # Batching the Input Tensors
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data_tensor, train_label_tensor), batch_size=self.batch_size, shuffle=True)
        
        # Training for epochs
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data, batch_label in data_loader:
                self.optimizer.zero_grad()  # Removing all gradients
                
                # Calculating the loss for the model in the current epoch 
                loss = self.criterion(self.model(batch_data), batch_label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                print(f'Epoch: {epoch + 1} / {self.epochs}\nLoss: {total_loss / len(data_loader)}')  # Displaying the current Loss while Training
                
    def Test_Model(self, test_data_tensor, test_label_tensor):
        self.model.eval()   # Setting the Model in evaluation
        
        with torch.no_grad():   # DIsbaling Gradient Calculation
            data_tensor, label_tensor = torch.tensor(test_data_tensor).unsqueeze(1), torch.tensor(test_label_tensor)
            accuracy = (torch.argmax(self.model(data_tensor), dim=1) == label_tensor).float().mean().item()
            
        return accuracy