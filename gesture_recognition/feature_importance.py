from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest

import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, AnyStr, Tuple

from api.gestures import get_all_coordinates_as_array, get_classifications_as_categorical, get_classifications

# Import TensorFlow for keras.utils.to_categorical
import tensorflow as tf
# Use this import method for VS Code IntelliSense
keras = tf.keras
from keras.wrappers.scikit_learn import KerasClassifier

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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

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


    def __flatten(self, arr: List) -> npt.NDArray:
        # TODO: Update flatten function to work for any data.
        return np.array(arr).reshape((len(arr), 1890))
    

    def __get_training_data_from_db(self) -> Tuple[List, List]:
        # Get data from database.
        try:
            gesture_data = get_all_coordinates_as_array(db=self.__get_db().__next__(), limit=999)
            classification_data = get_classifications_as_categorical(db=self.__get_db().__next__(), limit=999)
        except:
            raise RuntimeError("Could not fetch data. Ensure a database with valid entries exists.")

        return gesture_data, classification_data


    def __check_data_or_fetch_from_db(self, X_train: List, y_train: List, X_test: List = None, y_test: List = None) -> None:
        # If no input data is given, fetch from DB if no data is already saved..
        if X_train is None or y_train is None:
            if self.X_train is None or y_train is None:
                # Warn about overwriting if only one of self.X_train or self.y_train is set.
                if not (X_train is None and y_train is None):
                    print("WARNING: overwriting X_train and y_train with data from database because one of the two is None.")

                # Get data from database and create a train/test split.
                gesture_data, classification_data = self.__get_training_data_from_db()
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.__flatten(gesture_data), classification_data, test_size=0.20)
        else:
            # TODO: Add validation.
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test


    def calculate_rfc_importance(self, X_train: List = None, y_train: List = None, n_estimators: int = 1000, max_features: int = 10, savefig: bool = True) -> None:
        self.__check_data_or_fetch_from_db(X_train, y_train)

        forest = RandomForestClassifier(n_estimators=n_estimators)
        forest.fit(self.X_train, self.y_train)
        
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        top_feature_names = []
        top_importances = []
        top_std_devs = []

        sorted_indices = np.argsort(importances)[-max_features:]

        for idx in sorted_indices:
            top_importances.append(importances[idx])
            top_feature_names.append(self.feature_names[idx])
            top_std_devs.append(std[idx])

        if savefig:
            # Plot most important features.
            plt.bar(range(max_features), top_importances, yerr=top_std_devs)
            plt.title(f"Feature Importance from Mean Decrease in Impurity\n(n_estimators = {n_estimators}).")
            plt.ylabel("Mean accuracy decrease")
            plt.xticks(range(max_features), top_feature_names, rotation=45, ha="right")
            plt.tight_layout()

            # Save plot.
            filename = "plots/rfc_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
            plt.savefig(filename)
        else:
            # If not saving a plot, print top values.
            print("Top feature names:\n", top_feature_names)
            print("Top importances:\n", top_importances)
            print("Top standard deviations:\n", top_std_devs)
    

    def calculate_kbest_importance(self, max_features: int = 10, savefig: bool = True) -> None:
        skb = SelectKBest(k=max_features)
        skb.fit(X=self.__flatten(get_all_coordinates_as_array(db=self.__get_db().__next__())), y=get_classifications(db=self.__get_db().__next__()))

        top_feature_names = []
        top_scores = np.zeros(max_features)
        top_p_values = np.zeros(max_features)
        
        for idx, feature_idx in enumerate(skb.get_support(indices=True)):
            top_scores[idx] = skb.scores_[feature_idx]
            top_p_values[idx] = skb.pvalues_[feature_idx]
            top_feature_names.append(self.feature_names[feature_idx])

        if savefig:
            x_labels = []
            for idx in range(max_features):
                x_labels.append(f"{top_feature_names[idx]}, p={top_p_values[idx]:.1E}")

            # Plot most important features.
            plt.bar(range(max_features), top_scores)
            plt.title(f"Feature Importance from ANOVA F-value).")
            plt.ylabel("Score")
            plt.xticks(range(max_features), x_labels, rotation=45, ha="right")
            plt.tight_layout()

            # Save plot.
            filename = "plots/kbest_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
            plt.savefig(filename)
        else:
            # If not saving a plot, print top values.
            print("Top feature names: ", top_feature_names, "\n")
            print("Top scores: ", top_scores, "\n")
            print("Top p-values: ", top_p_values, "\n")


if __name__ == "__main__":
    fi = FeatureImportance()
    fi.calculate_rfc_importance(n_estimators=5000, max_features=20, savefig=True)
    fi.calculate_kbest_importance(max_features=20, savefig=True)
