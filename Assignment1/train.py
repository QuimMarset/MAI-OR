import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from model_utils import create_model


class Trainer:

    def __init__(self, results_path, model_path, model_name, image_size, num_classes, fine_tune):
        self.results_path = results_path
        self.model_path = model_path
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.model_name = model_name
        self.model = create_model(model_name, image_size, num_classes, fine_tune)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC', self.f1_metric])
        #self.model.summary()


    def recall(self, y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-8)
        return recall


    def precision(self, y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-8)
        return precision


    def f1_metric(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-8)


    def plot_metric(self, train_metric, val_metric, experiment_file, experiment_title, metric_name):
        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label=f'Train {metric_name}')
        plt.plot(val_metric, label=f'Val {metric_name}')
        plt.legend()
        plt.title(f'{metric_name} results using {self.model_name} and {experiment_title}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{experiment_file}_{metric_name}.png'))


    def train(self, num_epochs, train_gen, val_gen, experiment_file, experiment_title):
        fit_results = self.model.fit(train_gen, epochs=num_epochs, validation_data=val_gen)
        #self.model.save_weights(os.path.join(self.model_path, self.model_name, 'model.h5'))

        history = fit_results.history
        self.plot_metric(history['loss'], history['val_loss'], experiment_file, experiment_title, 'Loss')
        self.plot_metric(history['auc'], history['val_auc'], experiment_file, experiment_title, 'AUC')
        self.plot_metric(history['f1_metric'], history['val_f1_metric'], experiment_file, experiment_title, 'F1-score')