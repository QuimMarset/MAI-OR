import numpy as np
import cv2
from skimage.transform import rotate, resize



class DataAugmentation:

    def __init__(self, image_size):
        self.image_size = image_size

    
    def scale_bounding_box(self, bounding_box, original_width, original_height):
        
        def _scale_coordinate(coordinate, size):
            return int(self.image_size*coordinate/size)

        bb_ymin = _scale_coordinate(bounding_box[0], original_width)
        bb_xmin = _scale_coordinate(bounding_box[1], original_height)
        bb_ymax = _scale_coordinate(bounding_box[2], original_width)
        bb_xmax = _scale_coordinate(bounding_box[3], original_height)

        return [bb_xmin, bb_ymin, bb_xmax, bb_ymax]

    
    def _get_segmentation_box(self, mask):
        indices = np.where(mask == 1.0)
        return [min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])]


    def extract_segmentation_objects(self, image, segmentation_image, original_boxes):
        objects = []
        masks = []

        for original_box in original_boxes:
            slice_original_rows = slice(original_box[1], original_box[3] + 1)
            slice_original_columns = slice(original_box[0], original_box[2] + 1)

            segmentation_subimage = segmentation_image[slice_original_rows, slice_original_columns]
            # Remove pixels with value 0 (i.e. background) and 220 (i.e. object border)
            filtered_subimage = segmentation_subimage[(segmentation_subimage > 0) & (segmentation_subimage < 220)]

            values, counts = np.unique(filtered_subimage, return_counts=True)
            # Pick the color with the highest number of pixels, assuming it the object contained in the bounding box
            index = np.argmax(counts)
            value = values[index]

            mask = (segmentation_image == value).astype(np.float32)
            # The original image is already resized, so the mask is to extract the object
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            # Given that some bounding boxes contain more objects, stretch it to only contain that object
            box = self._get_segmentation_box(mask)
            slice_box_rows = slice(box[1], box[3] + 1)
            slice_box_columns = slice(box[0], box[2] + 1)
            
            extracted_object = image*np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
            extracted_object = extracted_object[slice_box_rows, slice_box_columns]

            mask = mask[slice_box_rows, slice_box_columns]

            objects.append(extracted_object)
            masks.append(mask)

        return objects, masks

    
    def _rotate(self, object, mask):
        rotation_angle = np.random.randint(0, 360)
        rotated_object = rotate(object, rotation_angle, resize=True)
        rotated_mask = rotate(mask, rotation_angle, resize=True, order=0)
        return rotated_object, rotated_mask

    
    def _scale_mask(self, mask, new_width, new_height):
        return cv2.resize(mask, (new_height, new_width), interpolation=cv2.INTER_NEAREST)


    def _scale(self, object, mask):
        width = object.shape[0]
        height = object.shape[1]
        
        aspect_ratio = width / height

        if max(width, height) < self.image_size/5:
            factor = 0.5
        else:
            factor = 2

        if width > height:
            min_width = min(width/factor, self.image_size)
            new_width = np.random.randint(min_width, self.image_size)
            new_height = int(new_width / aspect_ratio)

        elif height > width:
            min_height = min(height/factor, self.image_size)
            new_height = np.random.randint(min_height, self.image_size)
            new_width = int(aspect_ratio * new_height)

        else:
            min_size = min(width/factor, self.image_size)
            new_size = np.random.randint(min_size, self.image_size)
            new_width = new_size
            new_height = new_size

        scaled_object = resize(object, (new_width, new_height))
        scaled_mask = self._scale_mask(mask, new_width, new_height)
        
        if np.all(scaled_mask == 0.0):
            return object, mask

        return scaled_object, scaled_mask


    def _transform(self, object, mask):
        copied_object = object.copy()
        copied_mask = mask.copy()
        rotated_object, rotated_mask = self._rotate(copied_object, copied_mask)
        scaled_object, scaled_mask = self._scale(rotated_object, rotated_mask)

        # Need to get a new bounding box as the transformations have resized the image
        bounding_box = self._get_segmentation_box(scaled_mask)
        slice_row = slice(bounding_box[1], bounding_box[3] + 1)
        slice_column = slice(bounding_box[0], bounding_box[2] + 1)

        return scaled_object[slice_row, slice_column], scaled_mask[slice_row, slice_column]


    def _calculate_position(self, object_shape):
        max_height = self.image_size - object_shape[0]
        max_width = self.image_size - object_shape[1]
        row = np.random.randint(0, max_height)
        col = np.random.randint(0, max_width)
        return row, col


    def _compute_IoU(self, bounding_box_1, bounding_box_2):
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


    def _are_overlapping(self, bounding_box_1, bounding_box_2):
        IoU = self._compute_IoU(bounding_box_1, bounding_box_2)
        return IoU > 0


    def _check_overlap(self, image_bounding_boxes, object_position, object_shape):
        object_bounding_box = [object_position[1], object_position[0], object_position[1] + object_shape[1] - 1,
            object_position[0] + object_shape[0] - 1]

        for image_bounding_box in image_bounding_boxes:
            if self._are_overlapping(image_bounding_box, object_bounding_box):
                return True
        return False


    def _calculate_position_without_overlap(self, object_shape, image_boxes):
        exists_overlap = True
        tries = 0
        while exists_overlap and tries < 20:
            position = self._calculate_position(object_shape)
            exists_overlap = self._check_overlap(image_boxes, position, object_shape)
            tries += 1

        if not exists_overlap:
            return position
        else:
            return None


    def _add_object_to_image(self, image, object, position, mask):
        row = position[0]
        col = position[1]
        height = object.shape[0]
        width = object.shape[1]

        slice_row = slice(row, row + height)
        slice_col = slice(col, col + width)

        mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        image[slice_row, slice_col] = image[slice_row, slice_col]*(1 - mask) + object*mask
        return image


    def _add_objects_to_image(self, image_data, segmentation_data, num_objects, overlap_possible):
        image_boxes = image_data['boxes']
        num_placed_objects = 0

        while num_placed_objects < num_objects:

            indices = np.random.choice(len(segmentation_data), num_objects)

            for index in indices:
                data_i = segmentation_data[index]

                if data_i['name'] == image_data['name']:
                    continue

                object_idx = np.random.choice(len(data_i['objects']))
                object = data_i['objects'][object_idx]
                mask = data_i['masks'][object_idx]
                class_index = data_i['classes'][object_idx]

                transformed_object, transformed_mask = self._transform(object, mask)
                shape = transformed_object.shape

                if not overlap_possible:
                    object_position = self._calculate_position_without_overlap(shape, image_boxes)
                else:
                    object_position = self._calculate_position(shape)

                if object_position is None:
                    continue

                # The bounding box inside the image where we want to add the object
                bounding_box = [*object_position, object_position[0] + shape[0], 
                    object_position[1] + shape[1]]

                num_placed_objects += 1
                distorted_image = self._add_object_to_image(image_data['image'], transformed_object, object_position, 
                    transformed_mask)
                image_data['image'] = distorted_image
                image_data['boxes'].append(bounding_box)
                image_data['classes'].append(class_index)


    def corrupt_training_images(self, images_data, segmentation_data, num_objects, overlap_possible, transform_objects=True, equal_classes=False):
        for image_data in images_data:
            self._add_objects_to_image(image_data, segmentation_data, num_objects, overlap_possible)
