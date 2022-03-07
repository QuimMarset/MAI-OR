from dataset import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    root_path = './'
    train_val_path = os.path.join(root_path, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
    test_path = os.path.join('VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007')

    dataset = Dataset(train_val_path, test_path, 224)

    train_data, val_data, segmentation_data = dataset.get_train_val_data(val_percentage=0.2, 
        data_augmentation=True, overlap_possible=False, num_objects=4, transform_objects=True)

    print(f'Train length: {len(train_data)}')
    print(f'Validation length: {len(val_data)}')
    print(f'Segmentation length: {len(segmentation_data)}')