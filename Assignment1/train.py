import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from model_utils import create_model


class Trainer:

    def __init__(self, results_path, model_path, model_name, image_size, num_classes):
        self.results_path = results_path
        self.model_path = model_path
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.model_name = model_name
        self.model = create_model(model_name, image_size, num_classes)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['AUC', self.f1_metric])


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
        plt.figure(figsize=(10, 6))
        plt.plot(train_metric, label=f'Train {metric_name}')
        plt.plot(val_metric, label=f'Val {metric_name}')
        plt.title(f'{self.model_name} - {metric_name} - {experiment_title}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{metric_name}_{experiment_file}.png'))


    def plot_metric_fine_tune(self, train_metric, val_metric, initial_epochs, experiment_file, experiment_title, metric_name):
        plt.figure(figsize=(10, 6))
        plt.plot(train_metric, label=f'Train {metric_name}')
        plt.plot(val_metric, label=f'Val {metric_name}')
        plt.title(f'{self.model_name} - {metric_name} - {experiment_title}')
        plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start fine tuning')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{metric_name}_{experiment_file}_fine_tune.png'))
    

    def train(self, num_epochs, train_gen, val_gen, experiment_file, experiment_title, model_file):
        fit_results = self.model.fit(train_gen, epochs=num_epochs, validation_data=val_gen)
        self.model.save_weights(os.path.join(self.model_path, f'{model_file}.h5'))

        history = fit_results.history
        self.plot_metric(history['loss'], history['val_loss'], experiment_file, experiment_title, 'Loss')
        self.plot_metric(history['auc'], history['val_auc'], experiment_file, experiment_title, 'AUC')
        self.plot_metric(history['f1_metric'], history['val_f1_metric'], experiment_file, experiment_title, 'F1-score')


    def train_fine_tune(self, initial_epochs, fine_tune_epochs, train_gen, val_gen, experiment_file, experiment_title, model_file, verbose=0):
        fit_results = self.model.fit(train_gen, epochs=initial_epochs, validation_data=val_gen, verbose=verbose)
        history = fit_results.history

        f1_score = history['f1_metric']
        auc = history['auc']
        loss = history['loss']

        val_f1_score = history['val_f1_metric']
        val_auc = history['val_auc']
        val_loss = history['val_loss']

        self.model.trainable = True
        mobile = self.model.layers[1]
        for (index, layer) in enumerate(mobile.layers):
            layer.trainable = (index >= 100)

        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC', self.f1_metric])

        fit_results = self.model.fit(train_gen, epochs=fine_tune_epochs, validation_data=val_gen, verbose=0)
        self.model.save_weights(os.path.join(self.model_path, f'{model_file}_fine_tune.h5'))

        history = fit_results.history

        f1_score += history['f1_metric']
        auc += history['auc']
        loss += history['loss']

        val_f1_score += history['val_f1_metric']
        val_auc += history['val_auc']
        val_loss += history['val_loss']

        self.plot_metric_fine_tune(loss, val_loss, initial_epochs, experiment_file, experiment_title, 'Loss')
        self.plot_metric_fine_tune(auc, val_auc, initial_epochs, experiment_file, experiment_title, 'AUC')
        self.plot_metric_fine_tune(f1_score, val_f1_score, initial_epochs, experiment_file, experiment_title, 'F1-score')
