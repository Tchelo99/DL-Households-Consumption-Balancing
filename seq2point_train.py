import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Added import for pandas
import os
from data_feeder import TrainSlidingWindowGenerator
from model_structure import create_model, save_model


class Trainer:
    def __init__(
        self,
        appliance,
        batch_size,
        crop,
        network_type,
        training_directory,
        validation_directory,
        save_model_dir,
        epochs=25,
        input_window_length=599,
        validation_frequency=1,
        patience=3,
        min_delta=1e-6,
        verbose=1,
    ):
        self.__appliance = appliance
        self.__algorithm = network_type
        self.__network_type = network_type
        self.__crop = crop
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__patience = patience
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__loss = "mse"
        self.__metrics = ["mse", "msle", "mae"]
        self.__learning_rate = 0.001
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir

        self.__input_window_length = input_window_length
        self.__window_size = 2 + self.__input_window_length
        self.__window_offset = int((0.5 * self.__window_size) - 1)
        self.__max_chunk_size = 5 * 10**2
        self.__validation_frequency = validation_frequency
        self.__ram_threshold = 5 * 10**5
        self.__skip_rows_train = 10000000
        self.__validation_steps = 100
        self.__skip_rows_val = 0

        # Directories of the training and validation files
        self.__training_directory = training_directory
        self.__validation_directory = validation_directory

        self.__training_chunker = TrainSlidingWindowGenerator(
            file_name=self.__training_directory,
            chunk_size=self.__max_chunk_size,
            batch_size=self.__batch_size,
            crop=self.__crop,
            shuffle=True,
            skip_rows=self.__skip_rows_train,
            offset=self.__window_offset,
            ram_threshold=self.__ram_threshold,
        )
        self.__validation_chunker = TrainSlidingWindowGenerator(
            file_name=self.__validation_directory,
            chunk_size=self.__max_chunk_size,
            batch_size=self.__batch_size,
            crop=self.__crop,
            shuffle=True,
            skip_rows=self.__skip_rows_val,
            offset=self.__window_offset,
            ram_threshold=self.__ram_threshold,
        )

    def train_model(self):
        steps_per_training_epoch = np.round(
            int(self.__training_chunker.total_num_samples / self.__batch_size),
            decimals=0,
        )

        model = create_model(self.__input_window_length)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.__learning_rate,
                beta_1=self.__beta_1,
                beta_2=self.__beta_2,
            ),
            loss=self.__loss,
            metrics=self.__metrics,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=self.__min_delta,
            patience=self.__patience,
            verbose=self.__verbose,
            mode="auto",
        )

        callbacks = [early_stopping]

        training_history = self.default_train(
            model, callbacks, steps_per_training_epoch
        )

        training_history.history["val_loss"] = np.repeat(
            training_history.history["val_loss"], self.__validation_frequency
        )

        model.summary()
        save_model(model, self.__network_type, self.__algorithm, self.__appliance, self.__save_model_dir)

        self.plot_training_results(training_history)

    def default_train(self, model, callbacks, steps_per_training_epoch):
        training_history = model.fit(
            self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=self.__epochs,
            verbose=self.__verbose,
            callbacks=callbacks,
            validation_data=self.__validation_chunker.load_dataset(),
            validation_freq=self.__validation_frequency,
            validation_steps=self.__validation_steps,
        )

        return training_history

    def plot_training_results(self, training_history):
        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title("Training History")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()


