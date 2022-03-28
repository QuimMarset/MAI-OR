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
        augmen_mode, num_to_place, prob_augment, epochs, seed):
        
        self.batch_size = batch_size
        self.dataset_loader = TrainDataset(images_path, annotations_path, image_names, image_size, seg_objects, 
            augmen_mode, num_to_place, prob_augment, seed)
        self.num_batches = round(len(image_names)/self.batch_size)
        
        self.epochs = epochs
        self.avg_epoch_batch_placed = np.zeros(self.epochs)
        self.epoch = 0
    

    def __getitem__(self, idx):
        index = idx * self.batch_size
        batch_images, batch_classes, avg_batch_placed = self.dataset_loader.get_batch(index, self.batch_size)

        self.avg_epoch_batch_placed[self.epoch] += avg_batch_placed

        return batch_images, batch_classes

    
    def on_epoch_end(self):
        super().on_epoch_end()

        self.avg_epoch_batch_placed[self.epoch] /= self.num_batches
        self.epoch += 1
        
        if self.epoch == self.epochs:
            avg_placed = np.mean(self.avg_epoch_batch_placed)    
            print(f'Average placed during {self.epochs} epochs: {avg_placed:.4f}')
            self.avg_epoch_batch_placed = np.zeros(self.epochs)



class TrainBalancedImageGenerator(TrainImageGenerator):

    def __init__(self, batch_size, images_path, annotations_path, image_names, image_size, 
        place_per_label, segmentation_objects, augmentation_mode, histogram_path, epochs, seed):
        
        self.batch_size = batch_size
        self.place_per_label = place_per_label
        self.dataset_loader = TrainBalancedDataset(images_path, annotations_path, image_names, 
            image_size, place_per_label.copy(), segmentation_objects, augmentation_mode, seed)
        self.num_batches = round(len(image_names)/self.batch_size)

        self.num_classes = get_num_classes()
        self.epoch = 0
        self.epochs = epochs
        self.avg_epoch_batch_placed = np.zeros(self.epochs)
        self.epoch_class_placed = np.zeros(self.num_classes)
        self.histogram_path = histogram_path


    def __getitem__(self, idx):
        batch_images, batch_classes = super().__getitem__(idx)
        batch_one_hot = np.zeros((self.batch_size, self.num_classes))

        for (index, classes_names) in enumerate(batch_classes):
            batch_one_hot[index] = to_one_hot(classes_names)
            
            for class_name in classes_names:
                class_index = classes_dict[class_name]
                self.epoch_class_placed[class_index] += 1

        return batch_images, batch_one_hot


    def plot_classes_histogram(self, avg_class_placed):
        plt.figure(figsize=(8, 8))
        labels = classes_dict.keys()
        plt.bar(labels, avg_class_placed/np.sum(avg_class_placed))
        plt.xticks([i for i in range(self.num_classes)], labels, rotation='vertical')
        plt.title(f'Objects classes distribution')
        plt.tight_layout()
        plt.savefig(self.histogram_path)


    def on_epoch_end(self):
        super().on_epoch_end()
        self.dataset_loader.reset_place_per_label(self.place_per_label.copy())
        
        if self.epoch == self.epochs:
            avg_class_placed = self.epoch_class_placed / self.epochs
            print(f'Average class objects during {self.epochs} epochs: {avg_class_placed}')
            self.plot_classes_histogram(avg_class_placed)
            self.epoch_class_placed = np.zeros((self.epochs, self.num_classes))