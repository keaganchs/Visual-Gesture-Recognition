from tensorflow import keras
from tensorflow.keras import layers, Input

import numpy as np
import keras_tuner

import json

from database.db import GESTURE_LIST, VIDEO_LENGTH, SessionLocal, engine
from database import db_models, pydantic_models
from api.gestures import get_all_coordinates_as_array, get_classifications

from sklearn.model_selection import train_test_split

class TrainNeuralNetwork:
    def __init__(self):
        self.tuner = keras_tuner.RandomSearch(
            hypermodel=self.build_model,
            objective="val_accuracy",
            max_trials=3,
            executions_per_trial=2,
            overwrite=True,
            directory="keras",
            project_name="visual_gesture_recognition",
        )
        # Print info about the seach space.
        self.tuner.search_space_summary()
        print("#################################################")

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
        model = keras.Sequential()
        model.add(layers.Flatten())
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def prepare_data(self) -> None:
        gesture_data = get_all_coordinates_as_array(db=self.__get_db().__next__())
        classification_data = get_classifications(db=self.__get_db().__next__())

        # Convert [SWIPE_LEFT, SWIPE_RIGHT, ...] to [0, 1, ...].
        classification_map = {classification: value for value, classification in enumerate(GESTURE_LIST)}

        y = np.zeros(len(classification_data))

        for i, data in enumerate(classification_data):
            y[i] = classification_map[data]

        X_train, X_test, y_train, y_test = train_test_split(gesture_data, y, test_size=0.20)


        self.tuner.search(X_train, y_train, epochs=2, validation_data=(X_test, y_test))


if __name__ == "__main__":
    tnn = TrainNeuralNetwork()
    # tnn.build_model(keras_tuner.HyperParameters())
    tnn.prepare_data()



