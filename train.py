import argparse
import json
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from augmentation import create_dataloader
from model import CSRNet
from model_vgg19 import CSRNet_VGG19

LR = 1e-6
BATCH_SIZE = 1  # Images of different sizes
MOMENTUM = 0.95
DECAY = 5 * 1e-4
START_EPOCH = 1
EPOCHS = 200
REDUCTION = 'sum'
SAVE_MODEL_FREQ = 2  # Save temp model in case of failure every x epochs
PRINT_FREQ = 50
CHECKPOINTS_FILE_PATH = 'data/ckpt'
TRAIN_DATA_FILE_PATH = 'data/train'


def save_checkpoint(model, optimizer, epoch, best_loss, name="model"):
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'best_loss': best_loss}
    ckpt_file = os.path.join(CHECKPOINTS_FILE_PATH, f'{name}.pth.tar')
    logging.info(f'Creating checkpoint {ckpt_file} for epoch {epoch}')
    torch.save(state, ckpt_file)


def load_checkpoint(model, optimizer, filename, device):
    ckpt = torch.load(filename, map_location=device)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    best_loss = ckpt['best_loss']
    return model.to(device), optimizer, best_loss, start_epoch

def train(args):
    global START_EPOCH
    best_pred = None

    if not os.path.exists(CHECKPOINTS_FILE_PATH):
        os.mkdir(CHECKPOINTS_FILE_PATH)

    epoch_data_file_path = TRAIN_DATA_FILE_PATH + f'/epoch_data_{args.name}.json'
    epoch_data = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'train_time': [], 'evaluation_time': [],
                  'total_time': []}

    # Set device CPU/GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device} for training')

    # Create Model
    logging.info('Building model')
    if args.vgg19 :
      logging.info('Building VGG19')
      model = CSRNet_VGG19(training=True).to(device)
    else:
      model = CSRNet(training=True).to(device)
    criterion = nn.MSELoss(reduction=REDUCTION)

    optimizer = torch.optim.SGD(model.parameters(), LR, momentum=MOMENTUM, weight_decay=DECAY)

    if args.pretrained:
        logging.info(f'Loading checkpoint from {args.pretrained}')
        model, optimizer, best_pred, START_EPOCH = load_checkpoint(
            model, optimizer, args.pretrained, device)
        logging.info('Continue training at epoch {}'.format(START_EPOCH))
        if os.path.exists(epoch_data_file_path):
            with open(epoch_data_file_path) as infile:
                epoch_data = json.load(infile)
    # Logging info
    if args.augment_type:
        logging.info(f'Performing train augmentation of type {args.augment_type}')

    if not os.path.exists(TRAIN_DATA_FILE_PATH):
      os.mkdir(TRAIN_DATA_FILE_PATH)

    metadata_file_path = TRAIN_DATA_FILE_PATH + f'/metadata_data_{args.name}.json'
    if not os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'w') as outfile:
            metadata = {}
            metadata['learning_rate'] = LR
            metadata['momentum'] = MOMENTUM
            metadata['decay'] = DECAY
            metadata['augmentation_type'] = args.augment_type
            metadata['reduction'] = REDUCTION
            metadata['start_epoch'] = START_EPOCH
            metadata['pretrained'] = args.pretrained
            logging.info(f'Saving metadata: {metadata} in {metadata_file_path}')
            json.dump(metadata, outfile)

    # Get training and test file paths
    with open(args.train_json) as infile:
        train_image_paths = json.load(infile)
    with open(args.test_json) as infile:
        test_image_paths = json.load(infile)

    for epoch in range(START_EPOCH, EPOCHS):

        start_time = time.time()  # measure each epoch time

        logging.info('Training')

        train_loss = train_model(model, criterion, optimizer, epoch, device, args, train_image_paths)

        end_train_time = time.time()

        logging.info('Evaluating')

        test_loss, test_mae = evaluate(model, criterion, device, test_image_paths)

        end_evaluation_time = time.time()

        # Save checkpoints
        if best_pred is None or test_mae < best_pred:
            best_pred = test_mae
            save_checkpoint(model, optimizer, epoch, best_pred, name=f'best_pred_model_{args.name}')
        if epoch % SAVE_MODEL_FREQ == 0:
            save_checkpoint(model, optimizer, epoch, best_pred, name=f'temp_model_{args.name}')

        # Update metrics
        epoch_data['train_loss'].append(np.mean(train_loss))
        epoch_data['test_loss'].append(test_loss)
        epoch_data['test_mae'].append(test_mae)
        epoch_data['total_time'].append(end_evaluation_time - start_time)
        epoch_data['train_time'].append(end_train_time - start_time)
        epoch_data['evaluation_time'].append(end_evaluation_time - end_train_time)
        with open(epoch_data_file_path, 'w') as outfile:
            logging.info(f'Saving loss data in {epoch_data_file_path}')
            json.dump(epoch_data, outfile)

    # Save last model
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_FILE_PATH, f'last_model_{args.name}.pth.tar'))


def train_model(model, criterion, optimizer, current_epoch, device, args, train_image_paths):
    model.train()
    train_loader = create_dataloader(train_image_paths,
                                     augment_type=args.augment_type,
                                     shuffle=True)
    # Metrics
    running_loss = []

    for i, (image, target) in enumerate(train_loader):

        image, target = reshape_loaded_items(image, target)

        image = image.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        
        logging.debug(f'loss output shape {output.shape}')
        logging.debug(f'loss target shape {target.shape}')
        loss = criterion(output, target)

        # backard + optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss.append(loss.item())

        # Log results
        if i % PRINT_FREQ == 0:
            epoch_text = 'Epoch [{}/{}] ({}/{}) '.format(current_epoch, EPOCHS, i, len(train_loader))
            loss_text = 'Avg loss = {:0.3f}'.format(np.mean(running_loss))
            logging.info(epoch_text + loss_text)
    return np.mean(running_loss)


def evaluate(model, criterion, device, test_image_paths):
    model.eval()
    with torch.no_grad():
        test_loader = create_dataloader(test_image_paths)
        test_mae = 0
        test_loss = 0
        for image, target in test_loader:
            logging.debug(f'eval image shape {image.shape}')
            logging.debug(f'eval target shape {target.shape}')
            image, target = reshape_loaded_items(image, target)
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            # Calculate MAE
            test_mae += abs((output.sum() - target.sum()).item())
            # Clculate MSE
            test_loss += criterion(output, target).item()
        test_mae /= len(test_loader)
        test_loss /= len(test_loader)
        logging.info('MAE = {:0.4f}'.format(test_mae))
        logging.info('LOSS = {:0.4f}'.format(test_loss))
    return test_loss, test_mae


def reshape_loaded_items(image, target):
    logging.debug(f'loader image dim is {image.ndim}')
    logging.debug(f'loader image shape {image.shape}')
    logging.debug(f'loader target shape {target.shape}')
    image = np.squeeze(image, axis=0)
    target = np.swapaxes(target, 0, 1)
    logging.debug(f'image shape {image.shape}')
    logging.debug(f'target shape {target.shape}')
    return image, target


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Train name')
    parser.add_argument('train_json', type=str, help='Path to train.json')
    parser.add_argument('test_json', type=str, help='Path to val.json')
    parser.add_argument('--augment_type', '-at', type=str, default=None,
                        help='Train name')
    parser.add_argument('--pretrained', '-p', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('-vgg19', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)
    train(parse_arguments(sys.argv[1:]))
