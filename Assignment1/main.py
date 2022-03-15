import os
import random
from generator import ImageGenerator, TrainImageGenerator, TrainBalancedImageGenerator
from train import Trainer
from load_utils import *
from segmentation_utils import extract_train_segmentations, sort_objects_to_balance
from model_utils import mobile_name, resnet_name, inception_name
from other_utils import *


def run_experiment(model_name, results_path, models_path, train_gen, fine_tune, experiment_name, model_file):
    file_name, plot_title = experiment_name
    trainer = Trainer(results_path, models_path, model_name, image_size, num_classes, fine_tune)
    trainer.train(num_epochs, train_gen, val_generator, file_name, plot_title, model_file)



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

    train_split_path = os.path.join(root_path, 'voc_train.txt')
    val_split_path = os.path.join(root_path, 'voc_val.txt')

    segmentations_pickle_path = os.path.join(root_path, "segmentations.pkl")
    train_classes_counts_path = os.path.join(root_path, "trian_classes_counts.pkl")

    results_path = os.path.join(root_path, 'results')
    results_mobile_path = os.path.join(results_path, mobile_name)
    results_resnet_path = os.path.join(results_path, resnet_name)
    results_inception_path = os.path.join(results_path, inception_name)

    models_path = os.path.join(root_path, 'models')
    models_mobile_path = os.path.join(models_path, mobile_name)
    models_resnet_path = os.path.join(models_path, resnet_name)
    models_inception_path = os.path.join(models_path, inception_name)

    seed = 1412
    image_size = 224
    val_percentage = 0.2
    prob_augment = 0.5
    batch_size = 32
    num_epochs = 50
    num_classes = get_num_classes()

    augmentation_mode = AugmentationMode.AugmentationTransformSameProportion

    # ==========================
    # Read train and val split
    # ==========================

    train_names, val_names = read_train_val_split(train_split_path, val_split_path)

    random.Random(seed).shuffle(train_names)
    random.Random(seed).shuffle(val_names)

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

    if not exists_path(segmentations_pickle_path):
        train_segmentations = extract_train_segmentations(images_path, seg_objects_path, seg_classes_path, train_names, image_size)
        save_dict_to_pickle(segmentations_pickle_path, train_segmentations)
    else:
        train_segmentations = load_pickle_dict(segmentations_pickle_path)

    place_per_label, objects_per_label = sort_objects_to_balance(train_classes_counts, train_segmentations)

    # ==========================
    # Create train and val sets
    # ==========================

    val_generator = ImageGenerator(batch_size, images_path, annotations_path, val_names, image_size, seed)

    # ==========================
    # Train the model
    # ==========================

    fine_tune = False

    augmentation_mode = AugmentationMode.AugmentationTransformSameProportion

    train_generator = TrainBalancedImageGenerator(batch_size, images_path, annotations_path, train_names, 
        image_size, place_per_label, objects_per_label, augmentation_mode, results_path, seed)

    experiment_name_file = 'augmentation_proportions'
    experiment_name_title = 'Augmentation - Same class proportion'
    model_file = 'augmentation_proportions'

    run_experiment(mobile_name, results_mobile_path, models_mobile_path, 
        train_generator, fine_tune, (experiment_name_file, experiment_name_title), model_file)