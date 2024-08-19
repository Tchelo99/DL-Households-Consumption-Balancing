import tensorflow as tf
import os


def create_model(input_window_length):
    """Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(
        filters=30,
        kernel_size=(10, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(
        filters=30,
        kernel_size=(8, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(
        filters=40,
        kernel_size=(6, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(
        filters=50,
        kernel_size=(5, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(
        filters=50,
        kernel_size=(5, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def save_model(model, network_type, algorithm, appliance):
    """Saves a model to a specified location. Models are named using a combination of their
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.
    """
    model_path = f"./saved_models/{appliance}_{algorithm}_{network_type}_model.h5"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model in HDF5 format
    model.save(model_path, save_format="h5")
    print(f"Model saved to {model_path}")


def custom_input_layer_from_config(config):
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return InputLayer.from_config(config)


custom_objects = {
    "InputLayer": custom_input_layer_from_config,
}


def load_model(model, network_type, algorithm, appliance):
    """Loads a model from a specified location.

    Parameters:
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    Returns:
    model (tensorflow.keras.Model): The loaded Keras model.
    """
    model_path = f"G:/seq2point-nilm-master/saved_models/{appliance}_{algorithm}_{network_type}_model.h5"
    print("PATH NAME: ", model_path)

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model
