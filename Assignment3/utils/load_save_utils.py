import numpy as np
import os
import json
import cv2
from skimage import img_as_float
import pickle


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_jpg_image(images_path, image_name):
    image_path = os.path.join(images_path, f'{image_name}.jpg')
    return load_image(image_path)


def load_png_image(images_path, image_name):
    image_path = os.path.join(images_path, f'{image_name}.png')
    return load_image(image_path)


def load_depth_mask_from_file(load_path, sample_name):
    npy_file_path = os.path.join(load_path, f'{sample_name}.npy')
    with open(npy_file_path, 'rb') as file:
        depth = np.load(file)
        mask = np.load(file)
    return depth, mask


def load_sample_from_files(images_path, npys_path, file_name, image_shape):
    frame = load_jpg_image(images_path, file_name)
    depth_map, mask = load_depth_mask_from_file(npys_path, file_name)
    
    frame = img_as_float(frame)
    frame = cv2.resize(frame, image_shape)
    
    depth_map = cv2.resize(depth_map, image_shape)
    depth_map = depth_map / np.max(depth_map)

    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32)

    return frame, depth_map, mask
    

def save_depth_mask(depth, mask, sample_name, frame_num, save_path):
    npy_file_path = os.path.join(save_path, f'{sample_name}_{frame_num}.npy')
    with open(npy_file_path, 'wb') as file:
        np.save(file, depth)
        np.save(file, mask)


def load_train_metrics(load_path):
    path = os.path.join(load_path, 'metrics.json')
    with open(path, 'r') as file:
        metrics_dict = json.load(file)
    return metrics_dict


def save_train_metrics(metrics_dict, save_path):
    path = os.path.join(save_path, 'metrics.json')
    with open(path, 'w') as file:
        json.dump(metrics_dict, file, indent=4)


def load_pickle_dict(path):
    with open(path, "rb") as file:
        dictionary = pickle.load(file)
    return dictionary


def save_dict_to_pickle(path, dictionary):
    with open(path, "wb") as file:
        pickle.dump(dictionary, file)


def load_split_names(load_path):
    split_names = []
    with open(load_path, 'r') as file:
        for line in file.readlines():
            split_names.append(line.strip())
    return split_names


def load_train_val_split(load_path):
    train_names_path = os.path.join(load_path, 'train_split.txt')
    val_names_path = os.path.join(load_path, 'val_split.txt')
    train_names = load_split_names(train_names_path)
    val_names = load_split_names(val_names_path)
    return train_names, val_names


def save_split_names(file_names, save_path):
    with open(save_path, 'w') as file:
        for name in file_names:
            file.write(f'{name}\n')


def save_train_val_split(train_names, val_names, save_path):
    train_names_path = os.path.join(save_path, 'train_split.txt')
    val_names_path = os.path.join(save_path, 'val_split.txt')
    save_split_names(train_names, train_names_path)
    save_split_names(val_names, val_names_path)
