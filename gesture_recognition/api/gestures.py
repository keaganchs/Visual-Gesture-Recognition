from sqlalchemy.orm import Session
from typing import List

from database import db_models, pydantic_models

from json import JSONEncoder
from collections import deque

class HandHistoryEncoder(JSONEncoder):
    def default(self, obj):
        ################## JSON format of hand landmarks: ##################
        #   {
        #       "frames":
        #       [
        #           {
        #               "frame_idx":0, 
        #               "handedness":"Left"
        #               "landmarks":
        #               [
        #                   {"landmark_idx":0,  "x":0.0,  "y":0.0,  "z":0.0}
        #                   {"landmark_idx":1,  "x":0.0,  "y":0.0,  "z":0.0}
        #                                    ...
        #                   {"landmark_idx":20,  "x":0.0,  "y":0.0,  "z":0.0}
        #               ]
        #           },
        #
        #           ...
        #
        #           {
        #               "frame_idx":29, 
        #               "handedness":"Left"
        #               "landmarks":
        #               [
        #                   {"landmark_idx":0,  "x":0.0,  "y":0.0,  "z":0.0}
        #                   {"landmark_idx":1,  "x":0.0,  "y":0.0,  "z":0.0}
        #                                    ...
        #                   {"landmark_idx":20,  "x":0.0,  "y":0.0,  "z":0.0}
        #               ]
        #           }
        #       ]
        #   }

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

# Get one gesture by ID
def get_gesture_by_id(db: Session, gesture_id: int) -> pydantic_models.Gesture:
    return db.query(db_models.Gesture).filter(db_models.Gesture.id == gesture_id).first()

# Get all gestures of one classification
def get_gestures_by_classification(db: Session, classification: str, skip: int = 0, limit: int = 100) -> List[pydantic_models.Gesture]:
    return db.query(db_models.Gesture).filter(db_models.Gesture.classification == classification).offset(skip).limit(limit).all()

# Get all gestures
def get_gestures(db: Session, skip: int = 0, limit: int = 100) -> List[pydantic_models.Gesture]:
    return db.query(db_models.Gesture).offset(skip).limit(limit).all()

# Create new Gesture entry
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