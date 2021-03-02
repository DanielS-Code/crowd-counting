import sys
import logging
import argparse
import torch
import cv2
import numpy as np
import os
from torchvision import transforms
import json


from model import CSRNet

def count(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.info('Counting using {} device'.format(device))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ckpt = torch.load(args.model_path, map_location=torch.device(device))
    model = CSRNet(training=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)  # Connect to gpu or cpu
    model.eval()
    _, model_name = os.path.split(args.model_path)
    model_name = model_name.split('.')[0]
    with torch.no_grad():
        if args.a:
            results = {}
            with open(args.image_path) as infile:
                image_paths = json.load(infile)
            for image_path in image_paths:
                image = cv2.imread(image_path)
                image = t(image).to(device)
                image = torch.stack([image])
                count = model(image).sum().item()
                results[image_path] = count
                logging.info(f'Image: {image_path}, Count:{count}')
            with open(os.path.join('data/statistics', f'shanghai_{model_name}_output_counts.json'), 'w') as outfile:
                json.dump(results, outfile)
        else:
            image = cv2.imread(args.image_path)
            image = t(image).to(device)
            image = torch.stack([image])
            count = model(image).sum().item()
            print(f'Model: {model_name}, Count: {count}, Image: {args.image_path}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Trained model file')
    parser.add_argument('image_path', type=str, help='Image path to test on')
    parser.add_argument('-a', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    count(parse_arguments(sys.argv[1:]))
