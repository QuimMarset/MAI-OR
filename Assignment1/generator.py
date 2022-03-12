import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
import os
import random
from dataset import Dataset


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, dataset_loader, seed=0):
        self.batch_size = batch_size
        self.dataset_loader = dataset_loader
        self.num_images = self.dataset_loader.get_num_images()
        self.seed = seed
        self.num_batches = self.num_images//self.batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # Returns tuple (input, target) correspond to batch batch_index
        index = idx * self.batch_size

        batch_images, batch_classes = self.dataset_loader.get_batch(index, self.batch_size)
        return batch_images, batch_classes

    def on_epoch_end(self):
        self.dataset_loader.shuffle_paths()


class TrainImageGenerator(ImageGenerator):

    def __init__(self, batch_size, dataset_loader, seed=0):
        super().__init__(batch_size, dataset_loader, seed)
        self.num_placed = 0
    

    def __getitem__(self, idx):
        index = idx * self.batch_size
        
        batch_images, batch_classes, num_placed = self.dataset_loader.get_batch(index, self.batch_size)
        self.num_placed += num_placed
        return batch_images, batch_classes

    
    def on_epoch_end(self):
        super().on_epoch_end()
        avg_placed = self.num_placed/self.num_batches
        print(f'Average epoch placed: {avg_placed}')
        self.num_placed = 0