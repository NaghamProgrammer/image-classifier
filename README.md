# 🌸 Image Classifier: Flower Recognition AI 🌸


## 🌟 About This Project

This **state-of-the-art image classifier** can recognize **102 different flower species** with impressive accuracy! Built using PyTorch's deep learning framework, it leverages transfer learning with the powerful **VGG16 architecture** to deliver professional-grade results.

Perfect for:
- Botany enthusiasts 🌿
- AI/ML learners 🧠
- Gardening apps development 🌻
- Educational tools 📚

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **High Accuracy** | Achieves ~88% validation accuracy |
| **Transfer Learning** | Uses pretrained VGG16 model |
| **Easy Prediction** | Simple API for image classification |
| **Checkpoint System** | Save and resume training |
| **Customizable** | Adaptable to other image datasets |

## 🛠️ Technical Stack

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/flower-classifier.git
cd flower-classifier
```

2. **Set up a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
flower-classifier/
├── notebooks/               # Jupyter notebooks
│   └── Image Classifier Project.ipynb
├── src/                     # Source code
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   └── predict.py          # Prediction functions
├── data/                    # Dataset
│   ├── train/              # Training images
│   ├── valid/              # Validation images
│   └── test/               # Test images
├── checkpoints/             # Saved models
├── cat_to_name.json         # Category mappings
└── README.md                # This file
```

## 🧠 Training the Model

```python
# Sample training code
from src.train import train_model

model = train_model(
    data_dir='data/flowers',
    arch='vgg16',
    learning_rate=0.001,
    hidden_units=4096,
    epochs=4,
    gpu=True
)
```

## 🔍 Making Predictions

```python
from src.predict import predict

# Predict flower species
probs, classes = predict(
    image_path='test_flower.jpg',
    checkpoint='checkpoints/flower_classifier.pth',
    top_k=5
)

# Display results
print(f"Top predictions: {classes}")
print(f"Probabilities: {probs}")
```

## 📊 Performance Metrics


| Metric | Value |
|--------|-------|
| Training Accuracy | 92% |
| Validation Accuracy | 88% |
| Test Accuracy | 87% |
| Training Time | ~15 min (on GPU) |

## 🎨 Customization Options

1. **Try different architectures**:
```python
# Available options: 'vgg16', 'resnet18', 'alexnet'
model = train_model(arch='resnet18')
```

2. **Adjust hyperparameters**:
```python
model = train_model(
    learning_rate=0.0001,  # Lower learning rate
    hidden_units=2048,     # Fewer hidden units
    epochs=10             # More training epochs
)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Project Maintainer - [Nagham Wael](naghamw63@gmail.com)

Project Link: [https://github.com/NaghamProgrammer/flower-classifier](https://github.com/yourusername/flower-classifier)

---

Made with ❤️ and PyTorch by Nagham Wael
