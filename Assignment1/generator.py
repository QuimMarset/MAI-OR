import matplotlib.pyplot as plt
import os
from tensorflow import keras
import numpy as np
from other_utils import *
from dataset import Dataset, TrainDataset, TrainBalancedDataset


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, images_path, annotations_path, image_names, image_size, seed):
        self.batch_size = batch_size
        self.dataset_loader = Dataset(images_path, annotations_path, image_names, image_size, seed)
        self.num_batches = len(image_names)//self.batch_size


    def __len__(self):
        return self.num_batches


    def __getitem__(self, idx):
        index = idx * self.batch_size
        batch_images, batch_classes = self.dataset_loader.get_batch(index, self.batch_size)
        return batch_images, batch_classes


    def on_epoch_end(self):
        self.dataset_loader.shuffle_paths()


class TrainImageGenerator(ImageGenerator):

    def __init__(self, batch_size, images_path, annotations_path, image_names, image_size, seg_objects, 
        augmen_mode, num_to_place, prob_augment, seed):
        
        self.batch_size = batch_size
        self.dataset_loader = TrainDataset(images_path, annotations_path, image_names, image_size, seg_objects, 
            augmen_mode, num_to_place, prob_augment, seed)
        self.num_batches = len(image_names)//self.batch_size
        self.num_placed = 0
    

    def __getitem__(self, idx):
        index = idx * self.batch_size
        batch_images, batch_classes, num_placed = self.dataset_loader.get_batch(index, self.batch_size)
        self.num_placed += num_placed
        return batch_images, batch_classes

    
    def on_epoch_end(self):
        super().on_epoch_end()
        avg_placed = self.num_placed/self.num_batches
        print(f'Average epoch placed: {avg_placed:.2f}')
        self.num_placed = 0


class TrainBalancedImageGenerator(TrainImageGenerator):

    def __init__(self, batch_size, images_path, annotations_path, image_names, image_size, 
        place_per_label, segmentation_objects, augmentation_mode, histogram_path, seed):
        
        self.batch_size = batch_size
        self.place_per_label = place_per_label
        self.dataset_loader = TrainBalancedDataset(images_path, annotations_path, image_names, 
            image_size, place_per_label.copy(), segmentation_objects, augmentation_mode, seed)
        self.num_batches = len(image_names)//self.batch_size
        self.num_placed = 0
        self.num_classes = get_num_classes()
        self.label_counter = np.zeros(self.num_classes)
        self.histogram_path = histogram_path
        self.plot_histogram = True


    def __getitem__(self, idx):
        batch_images, batch_classes = super().__getitem__(idx)
        batch_one_hot = np.zeros((self.batch_size, self.num_classes))

        for (index, classes_names) in enumerate(batch_classes):
            batch_one_hot[index] = to_one_hot(classes_names)

            if self.plot_histogram:
                for class_name in classes_names:
                    class_index = classes_dict[class_name]
                    self.label_counter[class_index] += 1

        return batch_images, batch_one_hot


    def plot_classes_histogram(self):
        plt.figure(figsize=(8, 8))
        labels = classes_dict.keys()
        plt.bar(labels, self.label_counter/np.sum(self.label_counter))
        plt.xticks([i for i in range(self.num_classes)], labels, rotation='vertical')
        plt.title(f'Objects classes distribution')
        plt.tight_layout()
        plt.savefig(self.histogram_path)


    def on_epoch_end(self):
        super().on_epoch_end()
        self.dataset_loader.reset_place_per_label(self.place_per_label.copy())
        
        if self.plot_histogram:
            print(f'Number of objects per class: {self.label_counter}')
            self.plot_histogram = False
            self.plot_classes_histogram()
            self.label_counter = np.zeros(self.num_classes)