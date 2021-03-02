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

def evaluate(args):

    if not os.path.exists('data/metrics'):
        os.mkdir('data/metrics')
    model_name = os.path.basename(args.model_path).split('.')[0]
    test_path_file_name = os.path.basename(args.test_json).split('.')[0]
    metrics_file_path = f'data/metrics/metrics_{model_name}_{test_path_file_name}.json'

    if os.path.exists(metrics_file_path):
        with open(metrics_file_path) as infile:
            metrics = json.load(infile)
            logging.info(f'Final Metrics: {metrics}')
            exit()

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
    ckpt = torch.load(args.model_path, map_location=torch.device(device))
    if args.vgg19 :
      logging.info('Building VGG19')
      model = CSRNet_VGG19(training=False).to(device)
    else:
      model = CSRNet(training=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)  # Connect to gpu or cpu

    # Test the model using MSE and MAE
    logging.info('Starting evaluation')
    model.eval()
    with torch.no_grad():
      for i, (image, target) in enumerate(test_loader):

          image = np.squeeze(image, axis=0)
          target = np.swapaxes(target,0,1)

          image = image.to(device)
          target = target.to(device)

          output = model(image)

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

    # Save metrics
    with open(metrics_file_path, 'w') as outfile:
        json.dump(metrics, outfile)
        logging.info(f'Metrics saved to {metrics_file_path}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Trained model file')
    parser.add_argument('test_json', type=str, help='Image paths to test on')
    parser.add_argument('-vgg19', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)
    evaluate(parse_arguments(sys.argv[1:]))
