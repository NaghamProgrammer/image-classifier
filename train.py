import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import utility_functions as utils
import model_functions as model_utils

# Define command-line arguments
parser = argparse.ArgumentParser(description="Train a deep learning model on a flower dataset.")
parser.add_argument('data_dir', type=str, help="Path to the dataset directory")
parser.add_argument('--save_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
parser.add_argument('--arch', type=str, default='vgg16', help="Model architecture (e.g., vgg16, resnet18)")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training")
parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in the classifier")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training")

args = parser.parse_args()

# Load and preprocess the data
trainloader, validloader, class_to_idx = utils.load_data(args.data_dir)

# Build the model
model, criterion, optimizer = model_utils.build_model(
    arch=args.arch,
    hidden_units=args.hidden_units,
    learning_rate=args.learning_rate,
    class_to_idx=class_to_idx,
    gpu=args.gpu
)

# Train the model
model_utils.train_model(
    model=model,
    trainloader=trainloader,
    validloader=validloader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=args.epochs,
    gpu=args.gpu
)

# Save the checkpoint
model_utils.save_checkpoint(
    model=model,
    save_dir=args.save_dir,
    arch=args.arch,
    hidden_units=args.hidden_units,
    class_to_idx=class_to_idx
)