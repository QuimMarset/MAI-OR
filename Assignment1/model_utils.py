from tensorflow import keras


mobile_name = 'MobileNet-v2'
resnet_name = 'ResNet50'
inception_name = 'Inception-v3'


def create_mobile_model(image_size, num_classes, fine_tune):
    input_shape = (image_size, image_size, 3)
    mobile_net = keras.applications.MobileNetV2(input_shape, include_top=False, pooling='avg')
    mobile_net.trainable = fine_tune

    input = keras.Input(input_shape)
    extracted_features = mobile_net(input)
    dropout = keras.layers.Dropout(0.2)(extracted_features)
    probabilities = keras.layers.Dense(num_classes, activation='sigmoid')(dropout)
    return keras.Model(input, probabilities)


def create_resnet_model(image_size, num_classes, fine_tune):
    pass


def create_inception_model(image_size, num_classes, fine_tune):
    pass


def create_model(model_name, image_size, num_classes, fine_trune=False):
    if model_name == mobile_name:
        function = create_mobile_model
    elif model_name == resnet_name:
        function = create_resnet_model
    elif model_name == inception_name:
        function = create_inception_model
    else:
        raise KeyError(model_name)

    return function(image_size, num_classes, fine_trune)