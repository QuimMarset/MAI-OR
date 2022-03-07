import tensorflow as tf
import typing
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class Trainer:

    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes
        self.model = self._create_model(image_size, num_classes)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC', self.f1_metric])

    
    def _create_model(image_size, num_classes):
        input_shape = (image_size, image_size, 3)
        mobile_net = keras.applications.MobileNetV2(input_shape, include_top=False, pooling='avg')
        mobile_net.trainable = False

        input = keras.Input(input_shape)
        extracted_features = mobile_net(input)
        flatten = keras.layers.GlobalAveragePooling2D(extracted_features)(extracted_features)
        dropout = keras.layers.Dropout(0.2)(flatten)
        probabilities = keras.layers.Dense(num_classes, activation='sigmoid')(dropout)
        return keras.Model(input, probabilities)


    def recall(self, y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value.clip(y_true, 0, 1)))
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


    def train(self, num_epochs, batch_size, x_train, y_train, x_val, y_val):
        steps_per_epoch = x_train.shape[0] // batch_size

        train_gen = keras.preprocessing.image.ImageDataGenerator()
        train_set = train_gen.flow(x_train, y_train, batch_size=batch_size)

        val_gen = keras.preprocessing.image.ImageDataGenerator()
        val_set = val_gen.flow(x_val, y_val, batch_size=batch_size)

        fit_results = self.model.fit_generator(train_set, steps_per_epoch=steps_per_epoch, 
            epochs=num_epochs, verbose=1, validation_data=val_set)