import os
import torch
import torch.nn as nn
from utils.model import NeuralNet
from utils.data_utils import *
from utils.viz_utils import *
from training import train_model
from testing import test_model

def main():
    '''
    The main function that coordinates the training and testing of the neural network on the CIFAR-10 dataset.

    Steps:
    - Sets the input size and number of classes for the CIFAR-10 dataset.
    - Defines the training parameters such as epochs, batch size, and learning rate.
    - Initializes the model, loss function, and optimizer.
    - Loads the training, validation, and test datasets.
    - Trains the model and visualizes the training and validation losses.
    - Tests the trained model on the test dataset and prints the accuracy.

    '''
    # Create the figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    input_size = 3*32*32  # CIFAR-10 images are 3-channel RGB images with 32x32 pixels (3*32*32)
    num_classes = 10  # CIFAR-10 has 10 classes
    epochs = 50 # Number of epochs for training HW
    batch_size = 32 # Number of samples per batch HW
    learning_rate = 0.001 # Learning rate for the optimizer HW

    # Define the device for computation (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# unfortunately, I dont have an NVIDIA GPU :(
 
    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size = input_size, num_classes = num_classes)# Initiliaze and move the model to the device
    model = model.to(device);
    criterion = nn.CrossEntropyLoss() # Define the loss function (CrossEntropy for classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Choose optimizer with a learning rate, I wanna use Adam optimizer



    # Get the training and test datasets
    train_set, test_set = get_train_and_test_set()#
    
    # Create DataLoader for the training set
    train_loader = get_trainloader(train_set, batch_size=batch_size) #
    
    # Split the test set into validation and test subsets
    val_indices, test_indices = split_testset(test_set= test_set) #
    
    # Create DataLoaders for validation and test subsets
    val_loader  = get_validationloader(test_set=test_set, val_indices= val_indices, batch_size=batch_size) # 
    test_loader = get_testloader(test_set = test_set, test_indices=test_indices, batch_size=batch_size) # 
    
    # Train the model and track losses for each epoch
    train_losses, val_losses = train_model(model=model,device=device,epochs=epochs,loss_fn=criterion,optimizer=optimizer,train_loader=train_loader,val_loader=val_loader) #
    
    # Visualize the training and validation loss curves
    #
    visualize_train_val_losses(train_losses, val_losses)
    
    # Test the model on the test dataset, get visualizations and print accuracy
    #
    test_model(device, test_loader)

if __name__ == '__main__':
    main()