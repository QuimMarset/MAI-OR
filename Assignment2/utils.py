from keras.preprocessing.image import load_img
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os
import cv2
from colors_labels import *


def load_images(image_paths, num_images, image_size):
    images = np.zeros((num_images, *image_size, 3), dtype='float32')

    for i, path in enumerate(image_paths):
        images[i] = load_img(path, target_size=image_size)

    return images


def load_masks(mask_paths, num_masks, image_size):
    masks = np.zeros((num_masks, *image_size, 1), dtype='uint8')

    for i, path in enumerate(mask_paths):
        mask = load_img(path, target_size=image_size, color_mode='grayscale')
        masks[i] = np.expand_dims(mask, axis=2)

    return masks


def colour_segmentation(segmentation):
    colored_segmentation = colors[segmentation].astype('uint8')
    return colored_segmentation


def overlap_segmentation(image, segmentation):
    image_copy = image.copy()
    colored_segmentation = colour_segmentation(segmentation)
    
    overlapped_objects = cv2.addWeighted(image[segmentation != 0], 0.15, 
        colored_segmentation[segmentation != 0], 0.85, 0)

    image_copy[segmentation != 0] = overlapped_objects
    return image_copy


def create_legend(ground_truth, predicted):
    unique_ground = np.unique(ground_truth)
    unique_predicted = np.unique(predicted)
    unique_values = set(unique_ground).union(unique_predicted)
    unique_values.remove(0)

    patches = []

    for unique_value in unique_values:
        patch = Patch(facecolor=colors[unique_value]/255.0, label=labels[unique_value], edgecolor='black')
        patches.append(patch)

    return patches


def plot_segmentation(image, ground_truth, predicted_segm, path, image_name):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    for ax in axs:
        ax.axis('off')

    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    
    overlapped_ground = overlap_segmentation(image, ground_truth)
    axs[1].imshow(overlapped_ground)
    axs[1].set_title('Overlapped Ground-truth')

    overlapped_predicted = overlap_segmentation(image, predicted_segm)
    axs[2].imshow(overlapped_predicted)
    axs[2].set_title('Overlapped Predicted')

    legend = create_legend(ground_truth, predicted_segm)
    fig.legend(handles=legend, loc='lower center', ncol=8)

    fig.tight_layout()
    plt.savefig(os.path.join(path, f'segmented_{image_name}.jpg'))
    plt.show()
