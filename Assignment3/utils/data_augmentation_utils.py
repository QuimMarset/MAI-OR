import numpy as np
from skimage import transform


def rotate_(object, mask, resize=True):
    rotation_angle = np.random.randint(-20, 20)
    rotated_object = transform.rotate(object, rotation_angle, resize=resize)
    rotated_mask = transform.rotate(mask, rotation_angle, resize=resize, order=0)
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

    assert scaled_object.shape[0] < image_size and scaled_object.shape[1] < image_size, \
        'Scaled object size needs to be smaller than the image size'

    return scaled_object, scaled_mask


def get_box_from_mask(mask):
    indices = np.where(mask == 1.0)
    return [min(indices[1]), indices[0][0], max(indices[1]), indices[0][-1]]


def transform_(object, mask, image_size):
    rotated_object, rotated_mask = rotate_(object, mask)
    scaled_object, scaled_mask = scale_(rotated_object, rotated_mask, image_size)

    # Need to get a new bounding box as the transformations have resized the image

    if np.all(scaled_mask == 0):
        return None, None

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


def augment_image(image, segmentation_objects, image_size):
    num_to_place = np.random.randint(1, 3)
    num_placed = 0

    for _ in range(num_to_place + 4):
        index = np.random.choice(len(segmentation_objects))
        object_plus_mask  = segmentation_objects[index]
        object = object_plus_mask[:, :, :3]
        mask = object_plus_mask[:, :, 3]

        object, mask = transform_(object, mask, image_size)

        if object is None and mask is None:
            continue

        shape = mask.shape
        object_position = calculate_position(shape, image_size)
        image = add_object_to_image(image, object, object_position, mask)
        num_placed += 1

        if num_placed == num_to_place:
            break

    return image