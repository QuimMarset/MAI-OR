import os
import matplotlib.pyplot as plt
from dataset import Dataset
from generator import ImageGenerator
from train import Trainer


if __name__ == "__main__":

    root_path = './Assignment 1'
    train_val_path = os.path.join(root_path, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
    test_path = os.path.join('VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007')

    image_size = 224
    num_classes = 20

    dataset = Dataset(train_val_path, test_path, 224)

    train_images, val_images, train_classes, val_classes = dataset.get_train_val_data(val_percentage=0.2, 
        data_augmentation=True, overlap_possible=False, num_objects=1, transform_objects=True)

    print(f'Train length: {train_images.shape[0]}')
    print(f'Validation length: {val_images.shape[0]}')

    batch_size = 32
    num_epochs = 2

    train_gen = ImageGenerator(batch_size, train_images, train_classes)
    val_gen = ImageGenerator(batch_size, val_images, val_classes)

    trainer = Trainer(image_size, num_classes)

    trainer.train(num_epochs, train_gen, val_gen)