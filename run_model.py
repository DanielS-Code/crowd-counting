import sys
import logging
import argparse
import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from matplotlib import cm as CM
from matplotlib import pyplot as plt



from model import CSRNet

def run_model(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = cv2.imread(args.image_path)
    image = t(image).to(device)
    image = torch.stack([image])
    ckpt = torch.load(args.model_path, map_location=torch.device(device))
    model = CSRNet(training=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)  # Connect to gpu or cpu
    model.eval()
    with torch.no_grad():
        output = model(image)
        count = output.sum().item()
    _, model_name = os.path.split(args.model_path)
    model_name = model_name.split('.')[0]
    print(f'Model: {model_name}, Count: {count}, Image: {args.image_path}')
    plt.axis('off')
    output = np.squeeze(output, axis=0)
    output = np.squeeze(output, axis=0).cpu()
    plt.imshow(output, cmap=CM.jet, interpolation='sinc', vmin=0, vmax=args.inter)
    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Trained model file')
    parser.add_argument('image_path', type=str, help='Image path to test on')
    parser.add_argument('--inter', '-i', type=float, default=3,
                        help='Path to pretrained model')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run_model(parse_arguments(sys.argv[1:]))
