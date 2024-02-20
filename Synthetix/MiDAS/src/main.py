import argparse
from PIL import Image
from torchvision.transforms import Compose
import torch
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_model():
    model_path = "../models/model_weights.pth"
    model = MidasNet(model_path, non_negative=True)
    return model

def process_image(image_path, model):
    transform = Compose([Resize(384), NormalizeImage(), PrepareForNet()])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform({"image": image})["image"]
    with torch.no_grad():
        depth_map = model(input_tensor.unsqueeze(0))
    return depth_map

def main(input_image, output_image):
    model = load_model()
    depth_map = process_image(input_image, model)
    depth_map.save(output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation using MiDaS")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-image", type=str, required=True, help="Path to the output depth map image")
    args = parser.parse_args()
    main(args.input_image, args.output_image)
