import sys
import json
import logging
import argparse
import torch
import numpy as np
import os

from skimage.measure import compare_ssim as ssim
from model import CSRNet
from model_vgg19 import CSRNet_VGG19
from augmentation import create_dataloader

def evaluate_ensemble(args):


    # Metrics used for evaluation
    metrics = {'mae': 0, 'rmse': 0, 'ssim': 0}
    # Use either CPU or GPU (the later is preferable)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.info('Evaluating using {} device'.format(device))

    # Create dataset
    with open(args.test_json) as infile:
        image_paths = json.load(infile)
    test_loader = create_dataloader(image_paths)
    logging.info('Evaluating on {} images'.format(len(test_loader)))

    # Load model
    logging.info('Building model')
    ckpt_16 = torch.load(args.model_16_path, map_location=torch.device(device))
    ckpt_19 = torch.load(args.model_19_path, map_location=torch.device(device))

    model_16 = CSRNet(training=False).to(device)
    model_19 = CSRNet_VGG19(training=False).to(device)

    model_16.load_state_dict(ckpt_16['state_dict'])
    model_16.to(device)
    model_19.load_state_dict(ckpt_19['state_dict'])
    model_19.to(device)

    # Test the model using MSE and MAE
    logging.info('Starting evaluation')
    model_16.eval()
    model_19.eval()
    with torch.no_grad():
      for i, (image, target) in enumerate(test_loader):

          image = np.squeeze(image, axis=0)
          target = np.swapaxes(target,0,1)

          image = image.to(device)
          target = target.to(device)

          output_16 = model_16(image)
          output_19 = model_19(image)
          output = args.w * output_16 + (1-args.w) * output_19

          mse = (output.sum() - target.sum()) ** 2
          mae = torch.abs(output.sum() - target.sum())
          # SSIM
          ssim_target = target.squeeze().cpu().numpy()
          ssim_output = output.detach().squeeze().cpu().numpy()
          instance_ssim = ssim(ssim_target, ssim_output)
          # Update metrics
          metrics['mae'] += mae.item()
          metrics['rmse'] += mse.item()
          metrics['ssim'] += instance_ssim

          # Log every n
          if i % 10 == 0:
              image_info = 'Image:{}'.format(i)
              instance_info = ':Error:{:0.4f}:SquaredError:{:0.4f}'.format(
                  mae, mse)
              avg_info = '\tMAE = {:0.4f}\tRMSE = {:0.4f}'.format(
                  metrics['mae'] / (i + 1), np.sqrt(metrics['rmse'] / (i + 1)))
              logging.info(image_info + instance_info + avg_info)

    # Obtain average
    metrics['ssim'] /= len(test_loader)
    metrics['mae'] /= len(test_loader)
    metrics['rmse'] = np.sqrt(metrics['rmse'] / len(test_loader)).item()
    logging.info(f'Final Metrics: {metrics}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_16_path', type=str, help='Trained model file')
    parser.add_argument('model_19_path', type=str, help='Trained model file')
    parser.add_argument('test_json', type=str, help='Image paths to test on')
    parser.add_argument('w', type=float, help='VGG16 weight')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)
    evaluate_ensemble(parse_arguments(sys.argv[1:]))
