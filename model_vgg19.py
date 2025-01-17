import torch.nn as nn
from torchvision import models

class CSRNet_VGG19(nn.Module):
  def __init__(self, training=True):
    super().__init__()
    # M -> MaxPooling2D
    self.frontend_topology = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256 ,'M', 512, 512, 512, 512]
    self.backend_topology = [512, 512, 512, 256, 128, 64]
    # Layers
    self.frontend = make_layers(self.frontend_topology)
    self.backend = make_layers(self.backend_topology, in_channels=512, dilation=True)
    self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    # Load weights for training phase
    if training:
      vgg16 = models.vgg19(pretrained=True)
      self.initialize_weights()

      # Fetch part of pretrained model
      vgg16_dict = vgg16.state_dict()
      fronted_dict = self.frontend.state_dict()
      transfer_dict = {}

      for name, weights in vgg16_dict.items():
          common_op = '.'.join(name.split('.')[1:])
          if common_op in fronted_dict.keys() and 'features' in name:
              transfer_dict[common_op] = weights

      # Transfer weights
      self.frontend.load_state_dict(transfer_dict)

  def forward(self, x):
    x = self.frontend(x)
    x = self.backend(x)
    return self.output_layer(x)

  def initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          nn.init.normal_(m.weight, std=0.01)
          if m.bias is not None:
              nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init_constant_(m.bias, 0)


def make_layers(topology, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for layer in topology:
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = layer
    return nn.Sequential(*layers)

if __name__ == '__main__':
    print(CSRNet_VGG19(training=False))