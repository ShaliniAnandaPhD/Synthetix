import argparse
from PIL import Image
import torch
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_model():
    # Adjust the path to where your model weights are stored
    model_path = "models/model_weights.pth"
    model = MidasNet(model_path, non_negative=True)
    return model

def process_image(image_path, model):
    # Define transformations for the input image
    transform = Compose([
        Resize(384), 
        NormalizeImage(), 
        PrepareForNet()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform({"image": image})["image"]
    
    with torch.no_grad():
        depth_map = model(input_tensor.unsqueeze(0))
    
    return depth_map

def main(input_image, output_image):
    model = load_model()
    depth_map = process_image(input_image, model)
    # Convert depth_map to image and save
    depth_map_image = Image.fromarray(depth_map.squeeze().cpu().numpy())
    depth_map_image.save(output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation using MiDaS")
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-image", type=str, required=True, help="Path to save output depth map")
    args = parser.parse_args()
    
    main(args.input_image, args.output_image)
