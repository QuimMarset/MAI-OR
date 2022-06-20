import cv2
import numpy as np
import os
from skimage import img_as_float
from utils.load_save_utils import load_jpg_image, load_png_image


class_colors_dict = {
    8388608: 'aeroplane', 32768: 'bicycle', 8421376: 'bird', 128: 'boat', 8388736: 'bottle', 32896: 'bus', 
    8421504: 'car', 4194304: 'cat', 12582912: 'chair', 4227072: 'cow', 12615680: 'diningtable', 4194432: 'dog', 
    12583040: 'horse', 4227200: 'motorbike', 12615808: 'person', 16384: 'pottedplant', 8404992: 'sheep', 
    49152: 'sofa', 8437760: 'train', 16512: 'tvmonitor', 14737600: 'border'
}

def color2idx(image):
    return image[:, :, 0]*256**2 + image[:, :, 1]*256 + image[:, :, 2]


def get_box_from_mask(mask):
    indices = np.where(mask == 1.0)
    return [min(indices[1]), indices[0][0], max(indices[1]), indices[0][-1]]


def extract_segmentations_image(segmentations_list, image_path, seg_objects_path, seg_classes_path, image_name, image_size):
    seg_objects = load_png_image(seg_objects_path, image_name)
    seg_classes = load_png_image(seg_classes_path, image_name)

    seg_objects = color2idx(seg_objects)
    seg_classes = color2idx(seg_classes)

    image = load_jpg_image(image_path, image_name)
    image = img_as_float(cv2.resize(image, (image_size, image_size)))

    class_colors_indices = np.unique(seg_classes)
    # Remove background and object border colors
    class_colors_indices = class_colors_indices[1:-1]

    for class_color_index in class_colors_indices:

        class_objects = (seg_classes == class_color_index).astype(np.int32) * seg_objects

        class_objects_colors = np.unique(class_objects)
        # Remove background color
        class_objects_colors = class_objects_colors[1:]

        for object_color in class_objects_colors:

            mask = (class_objects == object_color).astype(np.uint8)
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            x_min, y_min, x_max, y_max = get_box_from_mask(mask)

            mask = np.expand_dims(mask[y_min:y_max+1, x_min:x_max+1], axis=-1)
            object = image[y_min:y_max+1, x_min:x_max+1]
            object_plus_mask = np.concatenate([object, mask], axis=-1)

            segmentations_list.append(object_plus_mask)


def extract_segmentations(images_path, seg_objects_path, seg_classes_path, image_size):
    segmentation_objects = []
    image_names = [image_file[:-4] for image_file in os.listdir(seg_objects_path)]

    for image_name in image_names:
        extract_segmentations_image(segmentation_objects, images_path, seg_objects_path, 
            seg_classes_path, image_name, image_size)

    return segmentation_objects