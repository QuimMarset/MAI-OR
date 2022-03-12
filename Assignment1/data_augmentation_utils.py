import numpy as np
import cv2
from skimage import transform
from other_utils import get_box_from_mask
import os
from other_utils import scale_bounding_box
from load_utils import read_annotation_file


def rotate_(object, mask):
    rotation_angle = np.random.randint(-15, 15)
    rotated_object = transform.rotate(object, rotation_angle, resize=True)
    rotated_mask = transform.rotate(mask, rotation_angle, resize=True, order=0)
    return rotated_object, rotated_mask


def scale_(object, mask, image_size):
    min_factor = 0.8
    max_factor = 1.2
    max_dim = max(object.shape[0], object.shape[1])

    if max_dim > image_size or max_dim*min_factor:
        min_factor = (image_size - round(image_size/20))/max_dim
        max_factor = (image_size - round(image_size/10))/max_dim

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
    row = np.random.randint(0, max_height)
    col = np.random.randint(0, max_width)
    return row, col


def compute_IoU(bounding_box_1, bounding_box_2):
    (x_min_1, y_min_1, x_max_1, y_max_1) = bounding_box_1
    (x_min_2, y_min_2, x_max_2, y_max_2) = bounding_box_2
    
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    x_min = max(x_min_1, x_min_2)
    y_min = max(y_min_1, y_min_2)
    x_max = min(x_max_1, x_max_2)
    y_max = min(y_max_1, y_max_2)
    area_intersec = max(0, x_max - x_min) * max(0, y_max - y_min)

    IoU = area_intersec / (area_1 + area_2 - area_intersec)
    return IoU


def are_overlapping(bounding_box_1, bounding_box_2):
    IoU = compute_IoU(bounding_box_1, bounding_box_2)
    return IoU > 0.15


def check_overlap(image_bounding_boxes, object_position, object_shape):
    object_bounding_box = [object_position[1], object_position[0], object_position[1] + object_shape[1] - 1,
        object_position[0] + object_shape[0] - 1]

    for image_bounding_box in image_bounding_boxes:
        if are_overlapping(image_bounding_box, object_bounding_box):
            return True
    return False


def calculate_empty_positions(image_size, bounding_boxes):
    empty = np.zeros((image_size, image_size))
    for bounding_box in bounding_boxes:
        [col1, row1, col2, row2] = bounding_box
        empty[row1:row2+1, col1:col2+1] = 1.0

    indices = np.where(empty == 0.0)
    zipped = list(zip(indices[0], indices[1]))
    empty_pos = np.empty(len(zipped), dtype=object)
    empty_pos[:] = zipped
    
    return empty_pos


def calculate_empty_positions_all(annotations_path, image_size):
    positions = {}
    image_names = [image_file[:-4] for image_file in os.listdir(annotations_path)]

    for image_name in image_names:
        _, bounding_boxes, width, height = read_annotation_file(annotations_path, image_name)
        scaled_boxes = [scale_bounding_box(bb, image_size, width, height) for bb in bounding_boxes]

        positions[image_name] = calculate_empty_positions(image_size, scaled_boxes)

    return positions


def calculate_position_without_overlap(empty_positions, object_shape, image_boxes, image_size, max_tries=5):
    filtered_pos = empty_positions[empty_positions + object_shape < (image_size, image_size)]
    exists_overlap = True
    num_tries = 0

    while exists_overlap and num_tries < max_tries:
        position = filtered_pos[np.random.choice(len(filtered_pos))]
        exists_overlap = check_overlap(image_boxes, position, object_shape)
        num_tries += 1

    if not exists_overlap:
        empty_positions = empty_positions[(empty_positions < position) | (empty_positions > position + object_shape)]
        return position, empty_positions
    else:
        return None, None


def add_object_to_image(image, object, position, mask):
    row = position[0]
    col = position[1]
    height = object.shape[0]
    width = object.shape[1]

    slice_row = slice(row, row + height)
    slice_col = slice(col, col + width)

    mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    image[slice_row, slice_col] = image[slice_row, slice_col]*(1 - mask) + object*mask
    return image


def corrupt_image(image, classes, boxes, segmentation_objects, num_to_place, overlap, image_size):
    num_placed = 0
    num_tries = 0
    max_tries = num_to_place*5

    empty_positions = calculate_empty_positions(image_size, boxes)

    while num_placed < num_to_place and num_tries < max_tries:
        num_tries += 1

        index = np.random.choice(len(segmentation_objects))
        objects_plus_masks, classes_names, _ = segmentation_objects[index]
        
        index2 = np.random.choice(len(objects_plus_masks))
        object_plus_mask = objects_plus_masks[index2]
        class_name = classes_names[index2]

        transformed_object, transformed_mask = transform_(object_plus_mask[:, :, :3], object_plus_mask[:, :, 3], image_size)
        shape = transformed_object.shape

        if not overlap:
            object_position, empty_positions = calculate_position_without_overlap(empty_positions, shape, boxes, image_size)
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

    return image, classes