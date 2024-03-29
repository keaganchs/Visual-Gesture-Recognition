from sqlalchemy.orm import Session
from typing import List

from database import db_models, pydantic_models

import numpy as np
import numpy.typing as npt

from database.db import GESTURE_LIST, VIDEO_LENGTH

from json import JSONEncoder
from collections import deque

import tensorflow as tf
# Use this import method for VS Code IntelliSense
keras = tf.keras


class HandHistoryEncoder(JSONEncoder):
    def default(self, obj):
        ################## JSON format of hand landmarks: ##################
        #   {
        #       "frames":
        #       [
        #           {
        #               "frame_idx":0, 
        #               "handedness":"Left",
        #               "landmarks":
        #               [
        #                   {"landmark_idx":0,  "x":0.0,  "y":0.0,  "z":0.0},
        #                   {"landmark_idx":1,  "x":0.0,  "y":0.0,  "z":0.0},
        #                                    ...
        #                   {"landmark_idx":20,  "x":0.0,  "y":0.0,  "z":0.0}
        #               ]
        #           },
        #
        #           ...
        #
        #           {
        #               "frame_idx":29, 
        #               "handedness":"Left",
        #               "landmarks":
        #               [
        #                   {"landmark_idx":0,  "x":0.0,  "y":0.0,  "z":0.0},
        #                   {"landmark_idx":1,  "x":0.0,  "y":0.0,  "z":0.0},
        #                                    ...
        #                   {"landmark_idx":20,  "x":0.0,  "y":0.0,  "z":0.0}
        #               ]
        #           }
        #       ]
        #   }
        ####################################################################

        list_obj = []
        if isinstance(obj, deque):
            list_obj = list(obj)
        else:
            list_obj = obj

        json_frames_list = {"frames":[]}
        # Loop through each frame of video
        for i, result in enumerate(list_obj):
            # If a hand was detected in the frame
            if (result is not None) and (result.multi_hand_landmarks is not None):
                temp_landmark_array = []
                # Loop through the 21 hand landmarks in each frame.
                for j, landmark in enumerate(result.multi_hand_landmarks[0].landmark):
                    if landmark is not None:
                        temp_landmark_array.append({
                            "landmark_idx": j,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    else:
                        temp_landmark_array.append({
                            "landmark_idx": j,
                            "x": None,
                            "y": None,
                            "z": None
                        })
                json_frames_list["frames"].append({
                    "frame_idx": i,
                    "handedness": result.multi_handedness[0].classification[0].label,
                    "landmarks": temp_landmark_array
                })
            # If the frame is empty:
            else: 
                json_frames_list["frames"].append({
                    "frame_idx": i,
                    "handedness": None,
                    "landmarks": None
                })

        return json_frames_list


def convert_video_to_array(video: dict) -> npt.ArrayLike:
    num_frames = VIDEO_LENGTH
    output_array = np.zeros((num_frames, 21, 3))

    for frame in video["frames"]:
        frame_idx = frame["frame_idx"]

        # If no landmark data exists, leave output_array as zero for this block.
        if frame["landmarks"] is not None:
            for landmark in frame["landmarks"]:
                landmark_idx = landmark["landmark_idx"]

                # Get coordinates.
                x, y, z = landmark["x"], landmark["y"], landmark["z"]

                output_array[frame_idx, landmark_idx] = [x, y, z]
        else:
            continue

    return output_array

def convert_list_of_videos_to_array(obj: List[dict]) -> npt.ArrayLike:
    # Initialize empty array with n frames, 21 landmarks, and 3 coordinates (x, y, z).
    num_datapoints = len(obj)
    num_frames = VIDEO_LENGTH
    output_array = np.zeros((num_datapoints, num_frames, 21, 3))

    for video_idx, video in enumerate(obj):
        output_array[video_idx] = convert_video_to_array(video)

    return output_array


# Get one gesture by ID.
def get_gesture_by_id(db: Session, gesture_id: int) -> pydantic_models.Gesture:
    return db.query(db_models.Gesture).filter(db_models.Gesture.id == gesture_id).first()


# Get all gestures of one classification.
def get_gestures_by_classification(db: Session, classification: str, skip: int = 0, limit: int = 100) -> List[pydantic_models.Gesture]:
    return db.query(db_models.Gesture).filter(db_models.Gesture.classification == classification).offset(skip).limit(limit).all()


# Get all hand coordinates, parse as a 3D Numpy array, then return.
def get_all_coordinates_as_array(db: Session, skip: int = 0, limit: int = 100) -> npt.ArrayLike:
    dict_data = [data for data, in db.query(db_models.Gesture.hand_coordinates).offset(skip).limit(limit).all()]
    return convert_list_of_videos_to_array(dict_data)


# Get all classifications.
def get_classifications(db: Session, skip: int = 0, limit: int = 100):
    return [classification for classification, in db.query(db_models.Gesture.classification).offset(skip).limit(limit).all()]


# Get all classifications as categotical data.
def get_classifications_as_categorical(db: Session, skip: int = 0, limit: int = 100):
    classification_data = [classification for classification, in db.query(db_models.Gesture.classification).offset(skip).limit(limit).all()]
    
    # Convert [SWIPE_LEFT, SWIPE_RIGHT, ...] to [0, 1, ...].
    classification_map = {classification: value for value, classification in enumerate(GESTURE_LIST)}
    num_classifications = len(classification_map)

    # Create output vector.
    y = np.zeros(len(classification_data))
    for i, data in enumerate(classification_data):
        y[i] = classification_map[data]

    # Convert values to categorical (e.g. [0, 1, 2, 3] -> [0, 0], [0, 1], [1, 0], [1, 1]).
    return keras.utils.to_categorical(y, num_classes=num_classifications)


# Get all gestures.
def get_gestures(db: Session, skip: int = 0, limit: int = 100) -> List[pydantic_models.Gesture]:
    return db.query(db_models.Gesture).offset(skip).limit(limit).all()


# Create new Gesture entry.
def create_gesture(db: Session, gesture: pydantic_models.GestureCreate):
    db_gesture = db_models.Gesture(
        sequence_length     = gesture.sequence_length,
        classification      = gesture.classification,
        hand_coordinates    = gesture.hand_coordinates
    )
    db.add(db_gesture)
    db.commit()
    db.refresh(db_gesture)
    return db_gesture