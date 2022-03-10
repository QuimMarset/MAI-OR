import os
import matplotlib.pyplot as plt
from dataset import *
from generator import ImageGenerator
from train import Trainer
from load_utils import *
from segmentation_utils import extract_segmentations, filter_segmentations_train


if __name__ == "__main__":

    root_path = "./MAI-OR/Assignment1"
    train_val_path = os.path.join(root_path, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
    segmentations_path = os.path.join(train_val_path, "SegmentationObject")
    annotations_path = os.path.join(train_val_path, "Annotations")
    images_path = os.path.join(train_val_path, "JPEGImages")

    pickle_path = os.path.join(root_path, "segmentations.pkl")

    seed = 1412
    image_size = 224
    num_classes = 20
    val_percentage = 0.2

    batch_size = 32
    num_epochs = 2

    num_to_place = 3
    overlap = False
    augmentation_mode = AugmentationMode.AugmentationTransform

    images_names = [image_file[:-4] for image_file in os.listdir(images_path)]
    num_images = len(images_names)

    if not exists_segmentations_pickle(pickle_path):
        segmentation_objects = extract_segmentations(images_path, segmentations_path, annotations_path, image_size)
        save_segmentations(pickle_path, segmentation_objects)
    else:
        segmentation_objects = load_segmentations_pickle(pickle_path)

    train_names, val_names = create_train_val_split(images_names, val_percentage, seed)

    train_segmentations = filter_segmentations_train(segmentation_objects, train_names)
    del segmentation_objects

    train_dataset = TrainDataset(images_path, annotations_path, train_names, image_size, train_segmentations, 
        AugmentationMode.AugmentationTransform, overlap, num_to_place, seed)
    val_dataset = Dataset(images_path, annotations_path, val_names, image_size, seed)

    train_generator = ImageGenerator(batch_size, train_dataset, seed)
    val_generator = ImageGenerator(batch_size, val_dataset, seed)

    batch = train_generator.__getitem__(0)

    trainer = Trainer(image_size, num_classes)
   # trainer.train(num_epochs, train_generator, val_generator)