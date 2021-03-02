import os
import sys
import logging
import groundtruth_generator

def generate_density_maps(dataset_dir):

  logging.info('Generating part A density map')

  for split in ['train_data', 'test_data']:
      section_dir = os.path.join(dataset_dir, 'part_A_final', split)
      save_dir = os.path.join(section_dir, 'densities')
      if not os.path.exists(save_dir):
          logging.info('Creating {} directory'.format(save_dir))
          os.mkdir(save_dir)
      groundtruth_generator.Shanghai(section_dir, save_dir).create_groundtruth()

  logging.info('Generating part B density map')

  for split in ['train_data', 'test_data']:
      section_dir = os.path.join(dataset_dir, 'part_B_final', split)
      save_dir = os.path.join(section_dir, 'densities')
      if not os.path.exists(save_dir):
          logging.info('Creating {} directory'.format(save_dir))
          os.mkdir(save_dir)
      groundtruth_generator.Shanghai(section_dir, save_dir).create_groundtruth()
      
  logging.info('Finished generating density maps')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_density_maps(sys.argv[1:][0])