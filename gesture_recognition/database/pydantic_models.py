from typing import Literal, NamedTuple, List
from pydantic import BaseModel, Json

from .db import GESTURE_LIST

# Gestures
########################################################

class GestureBase(BaseModel):
    sequence_length:    int
    classification:     Literal["SWIPE_LEFT", "SWIPE_RIGHT"]
    hand_coordinates:   Json

    # class Config:
    #     arbitrary_types_allowed = True
    
class GestureCreate(GestureBase):
    pass

class Gesture(GestureBase):
    id: int

    class Config:
        orm_mode = True