import kagglehub as kag
import pandas as pd
import torch

# Downloading the dataset
dataset_path = kag.dataset_download("datamunge/sign-language-mnist")

# Reading the .CSV Files for both training and testing datasets [using pandas]
train_dataset_csv = pd.read_csv(dataset_path + "/sign_mnist_train.csv")
test_dataset_csv = pd.read_csv(dataset_path + "/sign_mnist_test.csv")

# Extracting the Label and Pixel Information from the .CSV Files
# Training Data
extract_train_labels = train_dataset_csv.iloc[:, 0].values
extract_train_data = train_dataset_csv.iloc[:, 1:].values
# Testing Data
extract_test_labels = test_dataset_csv.iloc[:, 0].values
extract_test_data = test_dataset_csv.iloc[:, 1:].values

# Converting the Labels and Pixel Information to Tensors
# Training Data
train_label_tensor = torch.tensor(extract_train_labels, dtype=torch.long)
train_data_tensor = torch.tensor(extract_train_data, dtype=torch.float32)
train_data_tensor = train_data_tensor.view(-1, 28, 28)

# Testing Data
test_label_tensor = torch.tensor(extract_test_labels, dtype=torch.long)
test_data_tensor = torch.tensor(extract_test_data, dtype=torch.float32)
test_data_tensor = test_data_tensor.view(-1, 28, 28)


def Show_Tensor_Shape():
    print("Training Data: ", train_data_tensor.shape)
    print("Training Labels: ", train_label_tensor.shape)
    print("testing Data: ", test_data_tensor.shape)
    print("Testing Labels: ", test_label_tensor.shape)
