from tensorflow import keras


class DownscaleBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.convA = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = keras.layers.LeakyReLU(alpha=0.2)
        self.bn2a = keras.layers.BatchNormalization()
        self.bn2b = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPool2D((2, 2), (2, 2))


    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.us = keras.layers.UpSampling2D((2, 2))
        self.convA = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = keras.layers.LeakyReLU(alpha=0.2)
        self.bn2a = keras.layers.BatchNormalization()
        self.bn2b = keras.layers.BatchNormalization()
        self.conc = keras.layers.Concatenate()


    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)
        return x


class BottleNeckBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.convA = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = keras.layers.LeakyReLU(alpha=0.2)


    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x