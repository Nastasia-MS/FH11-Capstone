import tensorflow as tf
from tensorflow.keras import layers, models


def get_model(name, input_shape, num_classes, lr=0.001):
    """Return a compiled tf.keras model by name with reasonable default params."""
    if len(input_shape) == 1:
        input_shape = (input_shape[0], 1)

    if name == 'SimpleCNN':
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(16, 9, activation='relu', padding='same')(inputs)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(32, 7, activation='relu', padding='same')(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)
    elif name == 'MLP':
        inputs = layers.Input(shape=(input_shape[0],)) if len(input_shape) == 1 else layers.Input(shape=(input_shape[0]*input_shape[1],))
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)
    elif name == 'TinyConv':
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(8, 5, activation='relu', padding='same')(inputs)
        x = layers.MaxPool1D(2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)
    else:
        # default to a small conv net
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(8, 5, activation='relu', padding='same')(inputs)
        x = layers.MaxPool1D(2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
