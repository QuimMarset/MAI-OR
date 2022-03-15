import cv2
import numpy as np
import os
from load_utils import get_segmentation, read_image
from other_utils import get_box_from_mask


class_colors_dict = {
    8388608: 'aeroplane', 32768: 'bicycle', 8421376: 'bird', 128: 'boat', 8388736: 'bottle', 32896: 'bus', 
    8421504: 'car', 4194304: 'cat', 12582912: 'chair', 4227072: 'cow', 12615680: 'diningtable', 4194432: 'dog', 
    12583040: 'horse', 4227200: 'motorbike', 12615808: 'person', 16384: 'pottedplant', 8404992: 'sheep', 
    49152: 'sofa', 8437760: 'train', 16512: 'tvmonitor', 14737600: 'border'
}


def color2idx(image):
    return image[:, :, 0]*256**2 + image[:, :, 1]*256 + image[:, :, 2]


def extract_segmentations_image(segmentations_list, image_path, seg_objects_path, seg_classes_path, image_name, image_size):
    seg_objects = get_segmentation(seg_objects_path, image_name)
    seg_classes = get_segmentation(seg_classes_path, image_name)

    seg_objects = color2idx(seg_objects)
    seg_classes = color2idx(seg_classes)

    image = read_image(image_path, image_name, image_size)

    class_colors_indices = np.unique(seg_classes)
    # Remove background and object border colors
    class_colors_indices = class_colors_indices[1:-1]

    for class_color_index in class_colors_indices:

        class_name = class_colors_dict[class_color_index]

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

            segmentations_list.append((object_plus_mask, class_name))


def extract_train_segmentations(images_path, seg_objects_path, seg_classes_path, train_names, image_size):
    segmentation_objects = []
    image_names = [image_file[:-4] for image_file in os.listdir(seg_objects_path)]

    for image_name in image_names:
        if image_name in train_names:
            extract_segmentations_image(segmentation_objects, images_path, seg_objects_path, 
                seg_classes_path, image_name, image_size)

    return segmentation_objects


def get_class_objects(segmentation_objects, desired_label):
    objects = []

    for segmentation_object in segmentation_objects:
        (object, label) = segmentation_object
        if label == desired_label:
            objects.append(object)

    return objects


def sort_objects_to_balance(train_classes, segmentation_objects, fraction):
    labels = list(train_classes.keys())
    counts = list(train_classes.values())
    max_ocurrences = round(np.max(counts)/fraction)
    max_label = labels[np.argmax(counts)]
    place_per_label = {}
    objects_per_label = {}

    for (label, label_counts) in zip(labels, counts):
        if label == max_label:
            continue
        else:
            num_to_place = max(max_ocurrences - label_counts, 0)

        place_per_label[label] = num_to_place
        objects_per_label[label] = get_class_objects(segmentation_objects, label)

    return place_per_label, objects_per_label