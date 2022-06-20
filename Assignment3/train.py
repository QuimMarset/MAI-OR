import numpy as np
import os
from model import DepthEstimationModel, ModifiedLossMode
from data_generator import *
from utils.load_save_utils import *
from utils.plot_utils import *
from utils.segmentation_utils import extract_segmentations
import json


def generate_train_val_split(data_path, save_path, num_test_images=4000, seed=1412):
    file_names = np.array([os.path.splitext(file_name)[0] for file_name in os.listdir(data_path)])

    rng = np.random.default_rng(seed)
    rng.shuffle(file_names)

    val_indices = rng.choice(file_names.shape[0], num_test_images, replace=False)
    all_indices = range(file_names.shape[0])

    train_names = file_names[~np.isin(all_indices, val_indices)]
    val_names = file_names[val_indices]
    save_train_val_split(train_names, val_names, save_path)



def save_training_results(model, metrics_dict, experiment_path):
    model.save_weights(os.path.join(experiment_path, 'model_weights'))
    save_train_metrics(metrics_dict, experiment_path)


def save_hyperparameters(parameters_dict, experiment_path):
    file_path = os.path.join(experiment_path, 'parameters.json')
    file = open(file_path, "w")
    json.dump(parameters_dict, file)


def train_model(train_gen, val_gen, epochs, learning_rate, ssim_weight, l1_weight, edge_weight, image_size, experiment_path, 
                lr_decay=False, modified_loss=ModifiedLossMode.NoModified, workers=4):

    if lr_decay:
        learning_rate = keras.optimizers.schedules.ExponentialDecay(learning_rate, 1000, 0.95, staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
    model = DepthEstimationModel(ssim_weight, l1_weight, edge_weight, image_size, modified_loss)
    model.compile(optimizer, run_eagerly=True)

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, workers=workers)
    save_training_results(model, history.history, experiment_path)


def train(experiments_path, params_dict, train_gen, val_gen, lr_decay=False, modified_loss=ModifiedLossMode.NoModified):
    num_experiments = len(os.listdir(experiments_path))
    experiment_path = os.path.join(experiments_path, f'experiment_{num_experiments+1}')
    os.makedirs(experiment_path, exist_ok=True)

    train_model(train_gen, val_gen, params_dict['epochs'], params_dict['lr'], params_dict['ssim_weight'],
        params_dict['l1_weight'], params_dict['edge_weight'], params_dict['image_size'], experiment_path, lr_decay=lr_decay,
        modified_loss=modified_loss)

    save_hyperparameters(params_dict, experiment_path)