import torch


class Neural_Network:
    def __init__(self, epochs, learning_rate, batch_size):
        # Defining a device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defining the Model Format
        self.model = torch.nn.Sequential(
            # First Layer of the Network
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
            torch.nn.Linear(128 * 2 * 2, 256),
            torch.nn.ReLU(),
            # Linear Classification - 2
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            # Linear Classification - 3 [To Output]
            torch.nn.Linear(128, 25),
        ).to(self.device)

        # Defining Hyperparams
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Defining the Model's Loss Function and Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.NAdam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Mapping output class to letters
        self.class_to_letter = {
            i: chr(65 + i) for i in range(25)
        }  # A-Z (excluding one letter if 25 classes)

    def Train_Model(self, train_data_tensor, train_label_tensor):
        # Batching the Input Tensors
        train_data_tensor = train_data_tensor.to(self.device)
        train_label_tensor = train_label_tensor.to(self.device)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data_tensor, train_label_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Training for epochs
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data, batch_label in data_loader:
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(
                    self.device
                )

                # ReShaping the Batching Tensor
                batch_data = batch_data.view(-1, 1, 28, 28)

                # -1 -- infers to batch size
                # 1 -- Refering to the images being grayscale [3 for RGB]
                # 28, 28 -- Image Dimensions

                self.optimizer.zero_grad()  # Removing all gradients

                # Calculating the loss for the model in the current epoch
                loss = self.criterion(self.model(batch_data), batch_label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch: {epoch + 1} / {self.epochs}\nLoss: {total_loss / len(data_loader):.4f}\n"
            )  # Displaying the current Loss while Training

    def Test_Model(self, test_data_tensor, test_label_tensor):
        self.model.eval()  # Setting the Model in evaluation

        test_data_tensor = test_data_tensor.to(self.device)
        test_label_tensor = test_label_tensor.to(self.device)

        with torch.no_grad():  # Disbaling Gradient Calculation
            data_tensor, label_tensor = (
                test_data_tensor.clone().detach().unsqueeze(1),
                test_label_tensor.clone().detach(),
            )
            accuracy = (
                (torch.argmax(self.model(data_tensor), dim=1) == label_tensor)
                .float()
                .mean()
                .item()
            )

        return accuracy

    def Evaluate(self, data_tensor):
        # Move the data tensor to the correct device (CPU or GPU)
        data_tensor = data_tensor.to(self.device)

        # Ensure the tensor has the shape (1, 1, 28, 28) by adding dimensions as needed
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
        elif data_tensor.dim() == 3:
            data_tensor = data_tensor.unsqueeze(
                0
            )  # Shape: (1, 1, 28, 28) for single-channel images

        # Normalize the tensor to [0, 1] assuming the values are in [0, 255]
        data_tensor = data_tensor.float() / 255.0

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient computation as it's not needed for evaluation
        with torch.no_grad():
            # Perform prediction and get the index of the highest logit score
            prediction = self.model(data_tensor)
            predicted_label = torch.argmax(prediction, dim=1).item()

            # Map the predicted label to the corresponding letter
            predicted_letter = self.class_to_letter[predicted_label]
            print(self.class_to_letter)

        return predicted_letter
