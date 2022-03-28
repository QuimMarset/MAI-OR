from tensorflow import keras
from utils import *

class ImageGenerator(keras.Sequence):

    def __init__(self, batch_size, image_size, image_paths, mask_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths) // self.batch_size
        
    def __getitem__(self, batch_index):
        index = batch_index * self.batch_size

        batch_image_paths = self.image_paths[index : index+self.batch_size]
        batch_mask_paths = self.mask_paths[index : index+self.batch_size]

        batch_images = load_images(batch_image_paths, self.batch_size, self.image_size)
        batch_masks = load_masks(batch_mask_paths, self.batch_size, self.image_size)

        return batch_images, batch_masks