import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.load_save_utils import *
from utils.data_augmentation_utils import augment_image



class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images_path, npys_path, file_names, image_size):
        self.images_path = images_path
        self.npys_path = npys_path
        self.batch_size = batch_size
        self.image_shape = (image_size, image_size)
        self.file_names = file_names
        self.num_batches = len(self.file_names)//self.batch_size
        self.on_epoch_end()


    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):
        batch_index = index * self.batch_size
        images_masks, depths = self.load_batch(batch_index)
        return images_masks, depths


    def on_epoch_end(self):
        np.random.shuffle(self.file_names)


    def load_batch(self, batch_index):
        images_masks = np.zeros((self.batch_size, *self.image_shape, 4))
        depths = np.zeros((self.batch_size, *self.image_shape, 1))

        batch_file_names = self.file_names[batch_index : batch_index+self.batch_size]

        for (i, file_name) in enumerate(batch_file_names):
            image, depth, mask = load_sample_from_files(self.images_path, self.npys_path, file_name, self.image_shape)
            images_masks[i, :, :, :3] = image
            images_masks[i, :, :, 3] = mask
            depths[i] = tf.expand_dims(depth, axis=-1)

        return images_masks, depths



class DataGeneratorAugmentation(DataGenerator):

    def __init__(self, batch_size, images_path, npys_path, file_names, image_size, segmentation_objects):
        self.segmentation_objects = segmentation_objects
        super().__init__(batch_size, images_path, npys_path, file_names, image_size)


    def load_batch(self, batch_index):
        images_masks = np.zeros((self.batch_size, *self.image_shape, 4))
        depths = np.zeros((self.batch_size, *self.image_shape, 1))

        batch_file_names = self.file_names[batch_index : batch_index+self.batch_size]

        for (i, file_name) in enumerate(batch_file_names):
            image, depth, mask = load_sample_from_files(self.images_path, self.npys_path, file_name, self.image_shape)

            if np.random.random() < 0.5:
                image = augment_image(image, self.segmentation_objects, self.image_shape[0])

            images_masks[i, :, :, :3] = image
            images_masks[i, :, :, 3] = mask
            depths[i] = np.expand_dims(tf.image.convert_image_dtype(depth, tf.float32), axis=-1)

        return images_masks, depths