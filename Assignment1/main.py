import os
import matplotlib.pyplot as plt
from dataset import *
from generator import ImageGenerator
from train import Trainer
from load_utils import *
from segmentation_utils import extract_segmentations, filter_segmentations_train
from model_utils import mobile_name, resnet_name, inception_name


if __name__ == "__main__":

    # ==========================
    # Constants
    # ==========================

    root_path = "./Assignment1"
    train_val_path = os.path.join(root_path, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
    segmentations_path = os.path.join(train_val_path, "SegmentationObject")
    annotations_path = os.path.join(train_val_path, "Annotations")
    images_path = os.path.join(train_val_path, "JPEGImages")

    model_name = mobile_name

    results_path = os.path.join(root_path, 'results', model_name)
    model_path = os.path.join(root_path, 'models')

    pickle_path = os.path.join(root_path, "segmentations.pkl")

    seed = 1412
    image_size = 224
    num_classes = 20
    val_percentage = 0.2
    prob_augment = 0.5

    batch_size = 32
    num_epochs = 5

    num_to_place = 3
    augmentation_mode = AugmentationMode.AugmentationTransform
    overlap = True

    # ==========================
    # List images
    # ==========================

    images_names = [image_file[:-4] for image_file in os.listdir(images_path)]
    random.Random(seed).shuffle(images_names)
    num_images = len(images_names)

    # ==========================
    # Get segmentation objects
    # ==========================

    if not exists_path(results_path):
        create_results_folder(results_path)

    if not exists_path(pickle_path):
        segmentation_objects = extract_segmentations(images_path, segmentations_path, annotations_path, image_size)
        save_segmentations(pickle_path, segmentation_objects)
    else:
        segmentation_objects = load_segmentations_pickle(pickle_path)

    # ==========================
    # Create train and val sets
    # ==========================

    train_names, val_names = create_train_val_split(images_names, val_percentage, seed)

    train_segmentations = filter_segmentations_train(segmentation_objects, train_names)
    del segmentation_objects

    train_dataset = TrainDataset(images_path, annotations_path, train_names, image_size, train_segmentations, 
        augmentation_mode, num_to_place, prob_augment, seed)

    val_dataset = Dataset(images_path, annotations_path, val_names, image_size, seed)

    train_generator = ImageGenerator(batch_size, train_dataset, seed)
    val_generator = ImageGenerator(batch_size, val_dataset, seed)

    # ==========================
    # Train the model
    # ==========================

    trainer = Trainer(results_path, model_path, model_name, image_size, num_classes, fine_tune=False)

    experiment_name_file = f'{augmentation_mode.name}_{num_to_place}'
    experiment_name_title = str(augmentation_mode.name) + f" placing {num_to_place} objects"

    trainer.train(num_epochs, train_generator, val_generator, experiment_name_file, experiment_name_title)