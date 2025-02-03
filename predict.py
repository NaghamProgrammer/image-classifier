import argparse
import torch
from torchvision import transforms
from PIL import Image
import model_functions as model_utils
import utility_functions as utils

# Define command-line arguments
parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
parser.add_argument('image_path', type=str, help="Path to the input image")
parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint")
parser.add_argument('--top_k', type=int, default=5, help="Return the top K most likely classes")
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Path to the category names mapping file")
parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

args = parser.parse_args()

# Load the model checkpoint
model = model_utils.load_checkpoint(args.checkpoint, args.gpu)

# Preprocess the image
image = utils.process_image(args.image_path)

# Predict the class
probs, classes = model_utils.predict(image, model, args.top_k, args.gpu)

# Map classes to flower names
flower_names = utils.get_flower_names(classes, args.category_names)

# Display the results
print("Top K Predictions:")
for i in range(args.top_k):
    print(f"{flower_names[i]}: {probs[i]:.4f}")