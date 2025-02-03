import torch
from torch import nn, optim
import torchvision.models as models

def build_model(arch, hidden_units, learning_rate, class_to_idx, gpu):
    """Build and configure the model."""
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier
    if arch == 'vgg16':
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'resnet18':
        model.fc = nn.Sequential(
            nn.Linear(512, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    
    # Move to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if arch == 'vgg16' else model.fc.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    """Train the model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation loop
        model.eval()
        valid_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

def save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx):
    """Save the model checkpoint."""
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

def load_checkpoint(checkpoint_path, gpu):
    """Load the model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model, _, _ = build_model(
        arch=checkpoint['arch'],
        hidden_units=checkpoint['hidden_units'],
        learning_rate=0.001,
        class_to_idx=checkpoint['class_to_idx'],
        gpu=gpu
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(image, model, topk, gpu):
    """Predict the class of an image."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]