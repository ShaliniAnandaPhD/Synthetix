import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import DepthEstimationModel  # Assuming model.py contains the modified ResNet model

def load_model():
    model = DepthEstimationModel()
    model.load_state_dict(torch.load('models/depth_estimation_model.pth'))
    model.eval()
    return model

def predict_depth(image_path, model):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth_map = model(input_tensor)
    # Convert depth_map to an image and save or display it

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Map Prediction")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    args = parser.parse_args()

    model = load_model()
    predict_depth(args.image_path, model)
