import tensorflow as tf
# Use this import method for VS Code IntelliSense.
keras = tf.keras

from keras import layers

from typing import AnyStr, Tuple, List

import numpy as np
import keras_tuner
 
import matplotlib.pyplot as plt

from database.db import GESTURE_LIST, VIDEO_LENGTH, engine, get_db
from database import db_models
from api.gestures import get_all_coordinates_as_array, get_classifications_as_categorical

from sklearn.model_selection import train_test_split


class TrainNeuralNetwork:
    def __init__(self, save_model_path = "keras/best_model.h5", hyperopt_max_trials = 25):
        # Store a trained model.
        self.model = None
        # Path where the best model will be saved.
        self.save_model_path = save_model_path
        
        # Set up tuner for hyperparameter optimization.
        self.tuner = keras_tuner.RandomSearch(
            hypermodel=self.__build_model,
            objective="val_accuracy",
            max_trials=hyperopt_max_trials,
            executions_per_trial=2,
            overwrite=True,
            directory="keras",
            project_name="visual_gesture_recognition",
        )

        # Print info about the seach space.
        self.tuner.search_space_summary()
        print("\n##################################################################################################\n")

        # Set up database.
        try:
            db_models.Base.metadata.create_all(bind=engine)
        except:
            raise RuntimeError("Error creating a database connection.")
        

    def __get_training_data_from_db(self) -> Tuple[List, List]:
        # Get data from database.
        try:
            gesture_data = get_all_coordinates_as_array(db=get_db().__next__(), limit=999)
            classification_data = get_classifications_as_categorical(db=get_db().__next__(), limit=999)
        except:
            raise RuntimeError("Could not fetch data. Ensure a database with valid entries exists.")

        return gesture_data, classification_data


    def __build_model(self, hp) -> keras.Sequential:
        # Create Sequential model.
        model = keras.Sequential()
        
        # Add input layer.
        # There will be VIDEO_LENGTH * 21 hand landmarks * (x, y, z) input nodes.
        model.add(layers.Input(shape=(VIDEO_LENGTH, 21, 3), name="Input"))

        # Flatten the input:
        model.add(layers.Flatten(name="Flatten"))

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        # Tune dropout layer.
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))

        # Add output layer.
        model.add(layers.Dense(len(GESTURE_LIST), activation="softmax", name="Output"))

        # Compile model.
        model.compile(
            optimizer="adam",
            # optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ])
        return model


    def __plot_history(self, history: AnyStr, num_epochs: int, filename: str = "model_info") -> None:
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history["loss"], label="Loss"
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history["accuracy"], label="Accuracy"
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history["precision"], label="Precision"
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history["recall"], label="Recall"
        )
        plt.title("Evaluation Metrics", size=16)
        plt.xlabel("Epoch", size=12)
        plt.legend()
        plt.savefig(f"plots_and_data/{filename}.png")


    def calculate_best_model(self, num_epochs: int = 10, savefig: bool = False) -> None:
        gesture_data, classification_data = self.__get_training_data_from_db()
        
        # Create a train/test split.
        X_train, X_test, y_train, y_test = train_test_split(gesture_data, classification_data, test_size=0.20)

        # Convert array data to Tensors.
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        # Run hyperparameter optimization to find the optimal parameters.
        self.tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
        
        # Get the best hyperparameters.
        best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hyperparameters.values)
        self.model = self.__build_model(best_hyperparameters)
        
        # Train the model on the complete dataset.
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        history = self.model.fit(X, y, epochs=num_epochs)
        
        # Save the model and its weights.
        self.model.save(filepath=self.save_model_path, overwrite=True)

        # Plot info about model if savefig == True.
        if savefig:
            self.__plot_history(history=history, num_epochs=num_epochs)

        self.model.summary()


    def set_save_model_path(self, new_save_model_path: str) -> None:
        self.save_model_path = new_save_model_path


    def load_model(self, model_path: str = "keras/best_model.h5") -> None:
        try:
            self.model = keras.models.load_model(model_path)
        except:
            raise RuntimeError("Error loading model. Check model_path is valid, or run TrainNeuralNetwork.calculate_best_model() if none exists.")


    def fit_model_to_db_gestures(self, num_epochs: int = 20, savefig: bool = True) -> None:
        if self.model is None:
            raise RuntimeWarning("Load a model first!")

        gesture_data, classification_data = self.__get_training_data_from_db()
        X = tf.convert_to_tensor(gesture_data, dtype=tf.float32)
        # y = classification_data

        history = self.model.fit(X, classification_data, epochs=num_epochs)
        
        # Plot info about model if savefig == True.
        if savefig:
            self.__plot_history(history=history, num_epochs=num_epochs, filename="fitted_model_info")

        self.model.summary()


if __name__ == "__main__":
    tnn = TrainNeuralNetwork(save_model_path="keras/best_model.h5", hyperopt_max_trials=20)

    # Calculate best model using hyper parameter optimization
    # tnn.calculate_best_model(num_epochs=25, savefig=False)

    # Fit the model using a larger number of epochs.
    tnn.load_model("keras/best_model.h5")
    tnn.fit_model_to_db_gestures(num_epochs=25, savefig=True)
