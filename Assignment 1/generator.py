import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images, classes, random_state=0):
        self.batch_size = batch_size
        self.images = images
        self.classes = classes
        self.num_images = images.shape[0]
        self.random_state = random_state

    def __len__(self):
        return self.num_images//self.batch_size

    def __getitem__(self, idx):
        # Returns tuple (input, target) correspond to batch batch_index
        index = idx * self.batch_size

        batch_images = self.images[index : index+self.batch_size]
        batch_classes = self.classes[index : index+self.batch_size]

        return batch_images, batch_classes

    def on_epoch_end(self):
        self.images, self.classes = shuffle(self.images, self.classes, random_state=self.random_state)