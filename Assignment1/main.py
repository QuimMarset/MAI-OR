import os
import matplotlib.pyplot as plt
from dataset import *
from generator import ImageGenerator, TrainBalancedImageGenerator
from train import Trainer
from load_utils import *
from segmentation_utils import *
from model_utils import mobile_name, resnet_name, inception_name


if __name__ == "__main__":

    # ==========================
    # Constants
    # ==========================

    root_path = "./Assignment1"
    train_val_path = os.path.join(root_path, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
    seg_objects_path = os.path.join(train_val_path, "SegmentationObject")
    seg_classes_path = os.path.join(train_val_path, "SegmentationClass")
    annotations_path = os.path.join(train_val_path, "Annotations")
    images_path = os.path.join(train_val_path, "JPEGImages")

    model_name = mobile_name

    results_path = os.path.join(root_path, 'results', model_name)
    model_path = os.path.join(root_path, 'models')

    train_split_path = os.path.join(root_path, 'voc_train.txt')
    val_split_path = os.path.join(root_path, 'voc_val.txt')

    pickle_path = os.path.join(root_path, "segmentations.pkl")

    train_classes_counts_path = os.path.join(root_path, "trian_classes_counts.pkl")

    seed = 1412
    image_size = 224
    num_classes = 20
    prob_augment = 0.5

    batch_size = 32
    num_epochs = 1

    num_to_place = 6
    augmentation_mode = AugmentationMode.AugmentationTransform
    overlap = True

    # ==========================
    # Read train and val split
    # ==========================

    train_names, val_names = read_train_val_split(train_split_path, val_split_path)

    # ==========================
    # Get train class proportions
    # ==========================

    if not exists_path(train_classes_counts_path):
        train_classes_counts = extract_train_classes_counts(annotations_path, train_names)
        save_dict_to_pickle(train_classes_counts_path, train_classes_counts)
    else:
        train_classes_counts = load_pickle_dict(train_classes_counts_path)

    # ==========================
    # Get segmentation objects
    # ==========================

    if not exists_path(results_path):
        create_results_folder(results_path)

    if not exists_path(pickle_path):
        train_segmentations = extract_train_segmentations(images_path, seg_objects_path, seg_classes_path, train_names, image_size)
        save_dict_to_pickle(pickle_path, train_segmentations)
    else:
        train_segmentations = load_pickle_dict(pickle_path)

    place_per_label, objects_per_label = sort_objects_to_balance(train_classes_counts, train_segmentations)

    # ==========================
    # Create train and val sets
    # ==========================

    train_dataset = TrainBalancedDataset(images_path, annotations_path, train_names, image_size, place_per_label, objects_per_label, 
        augmentation_mode, seed)

    val_dataset = Dataset(images_path, annotations_path, val_names, image_size, seed)

    train_generator = TrainBalancedImageGenerator(batch_size, train_dataset, num_classes, seed)
    val_generator = ImageGenerator(batch_size, val_dataset, seed)

    # ==========================
    # Train the model
    # ==========================

    trainer = Trainer(results_path, model_path, model_name, image_size, num_classes, fine_tune=False)

    experiment_name_file = f'{augmentation_mode.name}_{num_to_place}'
    experiment_name_title = str(augmentation_mode.name) + f" placing {num_to_place} objects"

    trainer.train(num_epochs, train_generator, val_generator, experiment_name_file, experiment_name_title)

    print("???")