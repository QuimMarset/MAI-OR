import numpy as np
import cv2
from skimage import transform
from other_utils import get_box_from_mask
import os
from other_utils import scale_bounding_box
from load_utils import read_annotation_file


def rotate_(object, mask):
    rotation_angle = np.random.randint(-20, 20)
    rotated_object = transform.rotate(object, rotation_angle, resize=True)
    rotated_mask = transform.rotate(mask, rotation_angle, resize=True, order=0)
    return rotated_object, rotated_mask


def scale_(object, mask, image_size):
    min_factor = 0.8
    max_factor = 1.2
    max_dim = max(object.shape[0], object.shape[1])

    if max_dim > image_size or max_dim*min_factor:
        min_factor = (image_size*0.2)/max_dim
        max_factor = (image_size*0.5)/max_dim

    elif max_dim > 0.75*image_size:
        min_factor = 0.5
        max_factor = 0.8

    elif max_dim < image_size and max_dim*max_factor > image_size:
        max_factor = 1.0
    
    scale_factor = np.random.uniform(min_factor, max_factor)
    scaled_object = transform.rescale(object, scale=scale_factor, multichannel=True)
    scaled_mask = transform.rescale(mask, scale=scale_factor, order=0, anti_aliasing=False)

    if scaled_object.shape[0] > image_size or scaled_object.shape[1] > image_size:
        print(object.shape, scaled_object.shape, scale_factor)

    return scaled_object, scaled_mask


def transform_(object, mask, image_size):
    rotated_object, rotated_mask = rotate_(object, mask)
    scaled_object, scaled_mask = scale_(rotated_object, rotated_mask, image_size)

    # Need to get a new bounding box as the transformations have resized the image
    bounding_box = get_box_from_mask(scaled_mask)
    slice_row = slice(bounding_box[1], bounding_box[3] + 1)
    slice_column = slice(bounding_box[0], bounding_box[2] + 1)

    return scaled_object[slice_row, slice_column], scaled_mask[slice_row, slice_column]


def calculate_position(object_shape, image_size):
    max_height = image_size - object_shape[0]
    max_width = image_size - object_shape[1]
    row = np.random.randint(0, max_height+1)
    col = np.random.randint(0, max_width+1)
    return row, col


def compute_IoU(image_boxes, object_boxes):
    image_boxes = np.array(image_boxes)
    num_image_boxes = len(image_boxes)
    num_object_boxes = len(object_boxes)

    areas_boxes = (image_boxes[:, 2] - image_boxes[:, 0]) * (image_boxes[:, 3] - image_boxes[:, 1])
    areas_objects = (object_boxes[:, 2] - object_boxes[:, 0]) * (object_boxes[:, 3] - object_boxes[:, 1])

    image_boxes = np.expand_dims(image_boxes, axis=1)

    x_min = np.maximum(image_boxes[:, :, 0], object_boxes[:, 0])
    y_min = np.maximum(image_boxes[:, :, 1], object_boxes[:, 1])
    x_max = np.minimum(image_boxes[:, :, 2], object_boxes[:, 2])
    y_max = np.minimum(image_boxes[:, :, 3], object_boxes[:, 3])

    areas_intersec = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    
    areas_objects = np.tile(areas_objects, (num_image_boxes, 1))
    areas_boxes = np.tile(np.expand_dims(areas_boxes, axis=0).transpose(), (1, num_object_boxes))
    
    IoUs = areas_intersec / (areas_objects + areas_boxes - areas_intersec)
    return IoUs


def check_overlap(image_boxes, object_boxes):
    IoUs = compute_IoU(image_boxes, object_boxes)
    some_is_valid = np.all(IoUs <= 0.15, axis=0)
    indices = np.where(some_is_valid)[0]

    if indices.shape[0] > 0:
        return object_boxes[indices[0], :2]
    else:
        return None


def generate_object_boxes(object_shape, image_size, num_boxes):
    boxes = np.zeros((num_boxes, 4), dtype=int)
    boxes[:, :2] = [calculate_position(object_shape, image_size) for _ in range(num_boxes)]
    boxes[:, 2:] = boxes[:, :2] + object_shape - 1
    return boxes


def calculate_position_without_overlap(object_shape, image_boxes, image_size, max_tries=20):
    object_boxes = generate_object_boxes(object_shape, image_size, max_tries)
    return check_overlap(image_boxes, object_boxes)


def add_object_to_image(image, object, position, mask):
    row = position[0]
    col = position[1]
    height = object.shape[0]
    width = object.shape[1]

    slice_row = slice(row, row + height)
    slice_col = slice(col, col + width)

    mask = np.expand_dims(mask, axis=-1)
    image[slice_row, slice_col] = image[slice_row, slice_col]*(1 - mask) + object*mask
    return image


def corrupt_image(image, classes, boxes, segmentation_objects, num_to_place, overlap, image_size):
    num_placed = 0

    for _ in range(num_to_place):
        index = np.random.choice(len(segmentation_objects))
        objects_plus_masks, classes_names, _ = segmentation_objects[index]
        
        index2 = np.random.choice(len(objects_plus_masks))
        object_plus_mask = objects_plus_masks[index2]
        class_name = classes_names[index2]

        transformed_object, transformed_mask = transform_(object_plus_mask[:, :, :3], object_plus_mask[:, :, 3], image_size)
        shape = transformed_mask.shape

        if not overlap:
            object_position = calculate_position_without_overlap(shape, boxes, image_size)
            if object_position is None:
                continue
        else:
            object_position = calculate_position(shape, image_size)

        # The bounding box inside the image where we want to add the object
        bounding_box = [object_position[1], object_position[0], object_position[1] + shape[1] - 1, 
            object_position[0] + shape[0] - 1]

        num_placed += 1

        image = add_object_to_image(image, transformed_object, object_position, transformed_mask)
        classes.append(class_name)
        boxes.append(bounding_box)

    return image, classes, num_placed