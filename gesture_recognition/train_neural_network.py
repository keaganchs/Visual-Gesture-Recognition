import tensorflow as tf
# Use this import method for VS Code IntelliSense
keras = tf.keras

from keras import layers, Input

import numpy as np
import keras_tuner

import json

from database.db import GESTURE_LIST, VIDEO_LENGTH, SessionLocal, engine
from database import db_models, pydantic_models
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
            hypermodel=self.build_model,
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


    def __get_db(self):
        db = SessionLocal()
        try: 
            yield db
        finally:
            db.close()


    def build_model(self, hp) -> keras.Sequential:
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
        model.add(layers.Dense(2, activation="softmax", name="Output"))

        # Compile model.
        model.compile(
            optimizer="adam",
            # optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


    def calculate_best_model(self) -> None:
        # Get data from database.
        try:
            gesture_data = get_all_coordinates_as_array(db=self.__get_db().__next__(), limit=999)
            classification_data = get_classifications_as_categorical(db=self.__get_db().__next__(), limit=999)
        except:
            raise RuntimeError("Could not fetch data. Ensure a database with valid entries exists.")
        
        # Create a train/test split.
        X_train, X_test, y_train, y_test = train_test_split(gesture_data, classification_data, test_size=0.20)

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        # Run hyperparameter optimization to find the optimal parameters.
        self.tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        
        # Get the best hyperparameters.
        best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hyperparameters.values)
        self.model = self.build_model(best_hyperparameters)
        
        # Train the model on the complete dataset.
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        self.model.fit(X, y, epochs=10)

        # Save the model and its weights.
        self.model.save(filepath=self.save_model_path, overwrite=True)

        self.model.summary()


if __name__ == "__main__":
    tnn = TrainNeuralNetwork(hyperopt_max_trials=10)
    tnn.calculate_best_model()

