from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification

import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, AnyStr, Tuple

from api.gestures import get_all_coordinates_as_array, get_classifications_as_categorical

# Import TensorFlow for keras.utils.to_categorical
import tensorflow as tf
# Use this import method for VS Code IntelliSense
keras = tf.keras

from database.db import GESTURE_LIST, VIDEO_LENGTH, SessionLocal, engine
from database import db_models, pydantic_models
from helper_functions import time_this

HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]


class FeatureImportance:
    def __init__(self):
        self.feature_names = []

        for frame_idx in range(30):
            for landmark in HAND_LANDMARKS:
                for axis in ["x", "y", "z"]:
                    self.feature_names.append(f"Frame{frame_idx}_{landmark}_{axis}")

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


    def __get_training_data_from_db(self) -> Tuple[List, List]:
        # Get data from database.
        try:
            gesture_data = get_all_coordinates_as_array(db=self.__get_db().__next__(), limit=999)
            classification_data = get_classifications_as_categorical(db=self.__get_db().__next__(), limit=999)
        except:
            raise RuntimeError("Could not fetch data. Ensure a database with valid entries exists.")

        return gesture_data, classification_data


    def __fit_forest_classifier(self, X_train: List, y_train: List) -> RandomForestClassifier:
        forest = RandomForestClassifier(n_estimators=2000)
        forest.fit(X_train, y_train)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        return forest, importances, std


    def __plot_importances(self, forest_importances: AnyStr, std_deviation: AnyStr, max_features: int = 10) -> None:
        top_feature_names = []
        top_importances = []
        top_std_devs = []

        # Get the n = max_features most important features.
        print(np.sort(forest_importances))
        sorted_indices = np.argsort(forest_importances)[-max_features:]

        for idx in sorted_indices:
            top_importances.append(forest_importances[idx])
            top_feature_names.append(self.feature_names[idx])
            top_std_devs.append(std_deviation[idx])

        plt.bar(range(max_features), top_importances, yerr=top_std_devs)
        plt.title("Feature Importance")
        plt.ylabel("Mean accuracy decrease")
        plt.xticks(range(max_features), top_feature_names, rotation=45, ha="right")
        plt.tight_layout()

        filename = "plots/plot_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        plt.savefig(filename)


    def __flatten(self, arr: List) -> npt.NDArray:
        return np.array(arr).reshape((len(arr), 1890))
    

    def start(self) -> None:
        # Get data from database and create a train/test split.
        gesture_data, classification_data = self.__get_training_data_from_db()
        X_train, X_test, y_train, y_test = train_test_split(self.__flatten(gesture_data), classification_data, test_size=0.20)


        _, importances, std = self.__fit_forest_classifier(X_train, y_train)
        self.__plot_importances(forest_importances=importances, std_deviation=std)

        # result = permutation_importance(
        #     forest, X_test, y_test, n_repeats=100, n_jobs=-1
        # )
        # print(result)
        
        # importances = result.importances
        # sorted_indices = np.argsort(importances)[::-1]
        
        # p_values = []
        # for idx in sorted_indices:
        #     X_permuted = X_train.copy()
        #     np.random.shuffle(X_permuted[:, idx])
        #     _, permuted_importances, _ = permutation_importance(forest, X_permuted, , n_repeats=5)
        #     p_value = np.mean(permuted_importances >= importances[idx])
        #     p_values.append(p_value)

        # self.__plot_importances(forest_importances, result.importances_std)


if __name__ == "__main__":
    fi = FeatureImportance()
    fi.start()
