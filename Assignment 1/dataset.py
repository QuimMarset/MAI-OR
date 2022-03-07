import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from skimage import img_as_float
from data_augmentation import DataAugmentation
import time


class Dataset:

    def __init__(self, train_val_path, test_path, image_size, seed=0):
        self.train_val_path = train_val_path
        self.test_path = test_path
        self.images_folder = "JPEGImages"
        self.segmentation_folder = "SegmentationObject"
        self.annotation_folder = "Annotations"

        self.image_size = image_size
        self.num_classes = 20
        self.seed = seed

        self.classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 
            'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 
            'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

        self.data_augmentation = DataAugmentation(image_size)

    
    def _get_xml_root(self, dataset_path, image_name):
        path = os.path.join(dataset_path, self.annotation_folder, f'{image_name}.xml')
        tree = ET.parse(path)
        root = tree.getroot()
        return root


    def _read_annotation_file_val(self, dataset_path, image_name):
        root = self._get_xml_root(dataset_path, image_name)
        classes = np.zeros(self.num_classes)
        for object in root.iter('object'):
            class_index = self.classes[object.find('name').text]
            classes[class_index] = 1
        return classes


    def _read_annotation_file_train(self, dataset_path, image_name):
        root = self._get_xml_root(dataset_path, image_name)

        classes = []
        scaled_boxes = []
        nonscaled_boxes = []

        dimensions = root.find('size')
        width = int(dimensions.find('width').text)
        height = int(dimensions.find('height').text)

        for object in root.iter('object'):
            class_name = object.find('name').text
            bb_xmin = int(object.find("bndbox/xmin").text)
            bb_ymin = int(object.find("bndbox/ymin").text)
            bb_xmax = int(object.find("bndbox/xmax").text)
            bb_ymax = int(object.find("bndbox/ymax").text)

            bounding_box = [bb_xmin, bb_ymin, bb_xmax, bb_ymax]
            rescaled_bb = self.data_augmentation.scale_bounding_box(bounding_box, width, height)

            classes.append(self.classes[class_name])
            scaled_boxes.append(rescaled_bb)
            nonscaled_boxes.append(bounding_box)

        return classes, scaled_boxes, nonscaled_boxes


    def _read_image(self, dataset_path, image_name):
        path = os.path.join(dataset_path, self.images_folder, f'{image_name}.jpg')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_as_float(cv2.resize(image, (self.image_size, self.image_size)))
        return image
    

    def _exists_segmentation(self, dataset_path, image_name):
        return os.path.exists(os.path.join(dataset_path, self.segmentation_folder, f'{image_name}.png'))

    
    def _get_segmentation(self, dataset_path, image_name):
        path = os.path.join(dataset_path, self.segmentation_folder, f'{image_name}.png')
        segmentation = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return segmentation


    def _split_train_val(self, num_images, val_percentage):
        np.random.seed(self.seed)
        val_indices = np.random.randint(0, num_images, int(num_images*val_percentage))
        train_val_split = np.zeros(num_images)
        train_val_split[val_indices] = 1
        return train_val_split

    
    def _load_train_val_data(self, val_percentage):
        images_path = os.path.join(self.train_val_path, self.images_folder)
        image_names = os.listdir(images_path)
        num_images = len(image_names)
        train_val_split = self._split_train_val(num_images, val_percentage)
        
        train_data = []
        train_classes = []
        val_images = []
        val_classes = []
        segmentation_indices = []
        train_index = 0

        for (index, image_name) in enumerate(image_names):
            image_name = image_name[:-4]
            image = self._read_image(self.train_val_path, image_name)

            if train_val_split[index]:
                classes = self._read_annotation_file_val(self.train_val_path, image_name)
                val_images.append(image)
                val_classes.append(classes)

            else:
                objects_classes, scaled_boxes, nonscaled_boxes = self._read_annotation_file_train(self.train_val_path, image_name)

                image_data = {
                    'name' : image_name,
                    'image' : image,
                    'boxes' : scaled_boxes,
                    'names_used' : []
                }

                if self._exists_segmentation(self.train_val_path, image_name):
                    segmentation = self._get_segmentation(self.train_val_path, image_name)
                    objects, masks = self.data_augmentation.extract_segmentation_objects(image, segmentation, nonscaled_boxes)

                    segmentation_indices.append(train_index)
                    

                    image_data['objects'] = objects
                    image_data['masks'] = masks
            
                train_data.append(image_data)
                train_classes.append(objects_classes)
                train_index += 1

        return train_data, train_classes, segmentation_indices, val_images, val_classes
        
    
    def _prepare_train_data(self, train_data, train_classes):
        num_images = len(train_data)
        images = np.zeros((num_images, self.image_size, self.image_size, 3))
        classes = np.zeros((num_images, self.num_classes))

        for index, (image_data, image_classes) in enumerate(zip(train_data, train_classes)):
            images[index] = image_data['image']
            classes_indices = list(set(image_classes))
            classes[index, classes_indices] = 1

        return images, classes


    def get_train_val_data(self, val_percentage, data_augmentation=False, overlap_possible=False, 
        num_objects=0, transform_objects=False, equal_classes=False):

        start_time = time.time()
        
        train_data, train_classes, segmentation_indices, val_images, val_classes = self._load_train_val_data(val_percentage)

        print(f'Time to load data: {(time.time() - start_time):.2f}')
        start_time = time.time()
        
        self.data_augmentation.corrupt_training_images(train_data, train_classes, segmentation_indices, 
            num_objects, overlap_possible, transform_objects, equal_classes)

        print(f'Time to corrupt data: {(time.time() - start_time):.2f}')
        start_time = time.time()

        train_images, train_classes = self._prepare_train_data(train_data, train_classes)

        print(f'Time to finish splitting train data: {(time.time() - start_time):.2f}')

        return train_images, np.array(val_images), train_classes, np.array(val_classes)


    def get_test_data(self):
        image_names = os.listdir(os.path.join(self.test_path, self.images_folder))
        num_images = len(image_names)

        test_images = np.zeros((num_images, self.image_size, self.image_size, 3))
        test_classes = np.zeros((num_images, self.num_classes))

        for index, image_name in enumerate(image_names):
            image_name = image_name[:-4]
            image = self._read_image(self.test_path, image_name)
            objects_classes, _ = self._read_annotation_file(self.test_path, image_name)

            test_images[index] = image
            test_classes[index] = objects_classes

        return test_images, test_classes