import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from utils.load_save_utils import load_train_metrics



def __plot_experiments_val_metric(experiments_metric, save_path, metric_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    num_epochs = len(experiments_metric[0])
    epochs_range = range(1, num_epochs+1)

    for (i, metric_experiment) in enumerate(experiments_metric):
        plt.plot(epochs_range, metric_experiment, label=f'Experiment {i+1}', marker='o')
    
    plt.xticks(epochs_range)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Experiments validation {metric_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'val_{metric_name}_experiments.png'))
    plt.close()


def get_experiment_metric_values(experiments_path, experiment_number, metric_name):
    experiment_path = os.path.join(experiments_path, f'experiment_{experiment_number}')
    experiment_metric = load_train_metrics(experiment_path)[metric_name]
    return experiment_metric


def plot_experiments_val_metric(experiments_path, save_path, metric_name='loss'):
    num_experiments = len(os.listdir(experiments_path))
    experiments_metric = [get_experiment_metric_values(experiments_path, index+1, metric_name) for index in range(num_experiments)]
    __plot_experiments_val_metric(experiments_metric, save_path, metric_name)


def __plot_experiment_val_metric(metric_values, save_path, metric_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    num_epochs = len(metric_values)
    epochs_range = range(1, num_epochs+1)

    plt.plot(epochs_range, metric_values, marker='o')
    
    plt.xticks(epochs_range)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Best experiment validation {metric_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'val_{metric_name}_best_experiment.png'))
    plt.close()


def plot_best_experiment_val_metrics(experiment_path, save_path):
    experiment_metrics = load_train_metrics(experiment_path)
    __plot_experiment_val_metric(experiment_metrics['val_loss'], save_path, 'val_loss')
    __plot_experiment_val_metric(experiment_metrics['val_rmse'], save_path, 'val_rmse')
    __plot_experiment_val_metric(experiment_metrics['val_acc_1.25'], save_path, 'val_acc_1.25')
    __plot_experiment_val_metric(experiment_metrics['val_acc_2_1.25'], save_path, 'val_acc_2_1.25')


def plot_depth_estimation(image, ground_truth, predicted_depth, path, image_name):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    _, axs = plt.subplots(1, 3, figsize=(8, 5))
    for ax in axs:
        ax.axis('off')

    axs[0].imshow(image)
    axs[0].set_title('Frame')
    axs[1].imshow(ground_truth, cmap=cmap)
    axs[1].set_title('Ground-truth')
    axs[2].imshow(predicted_depth, cmap=cmap)
    axs[2].set_title('Predicted')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'depth_{image_name}.jpg'))
    plt.close()