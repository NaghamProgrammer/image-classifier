import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from PIL import Image
import numpy as np

def load_data(data_dir):
    """Load and preprocess the dataset."""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Create dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=32)
    
    return trainloader, validloader, train_data.class_to_idx

def process_image(image_path):
    """Preprocess an image for the model."""
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255.0
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)

def get_flower_names(classes, category_names):
    """Map class indices to flower names."""
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return [cat_to_name[str(cls)] for cls in classes]