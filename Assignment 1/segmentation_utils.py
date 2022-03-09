import cv2
import numpy as np
import os
from load_utils import get_segmentation, save_segmentations, read_image, read_annotation_file
from other_utils import get_box_from_mask


def extract_segmentations_image(image_path, segmentation_path, image_name, bounding_boxes, image_size):
    segmentation_image = get_segmentation(segmentation_path, image_name)
    image = read_image(image_path, image_name, image_size)

    objects_plus_masks = []

    for bounding_box in bounding_boxes:

        slice_rows = slice(bounding_box[1], bounding_box[3] + 1)
        slice_columns = slice(bounding_box[0], bounding_box[2] + 1)

        segmentation_subimage = segmentation_image[slice_rows, slice_columns]
        # Remove pixels with value 0 (i.e. background) and 220 (i.e. object border)
        filtered_subimage = segmentation_subimage[(segmentation_subimage > 0) & (segmentation_subimage < 220)]

        values, counts = np.unique(filtered_subimage, return_counts=True)
        # Pick the color with the highest number of pixels, assuming it the object contained in the bounding box
        value = values[np.argmax(counts)]

        mask = (segmentation_image == value).astype(np.float32)
        # The original image is resized, so is the mask to extract the object
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        
        # Given that some bounding boxes contain more objects, stretch it to only contain that object
        box = get_box_from_mask(mask)
        slice_box_rows = slice(box[1], box[3] + 1)
        slice_box_columns = slice(box[0], box[2] + 1)
        
        extracted_object = image*np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        extracted_object = extracted_object[slice_box_rows, slice_box_columns]
        mask = mask[slice_box_rows, slice_box_columns]

        object_plus_mask = np.concatenate((extracted_object, np.expand_dims(mask, axis=-1)), axis=-1)
        objects_plus_masks.append(object_plus_mask)
    
    return objects_plus_masks


def extract_segmentations(dataset_path, segmentation_folder, annotations_folder):
    segmentation_path = os.path.join(dataset_path, segmentation_folder)
    annotations_path = os.path.join(dataset_path, annotations_folder)
    
    image_names = os.listdir(segmentation_path)

    segmentation_objects = {}

    for image_name in image_names:
        classes, bounding_boxes, _, _ = read_annotation_file(annotations_path, image_name)

        objects, classes = extract_segmentations_image(segmentation_path, image_name, bounding_boxes)
        segmentation_objects[image_name] = (objects, classes)

    save_segmentations(segmentation_objects)


def filter_segmentations_train(segmentation_objects, train_names):
    train_segmentations = []
    for train_name in train_names:
        if train_name in segmentation_objects:
            train_segmentations.append(segmentation_objects[train_name])
    return train_segmentations