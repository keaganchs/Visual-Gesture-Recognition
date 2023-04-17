from pydantic import BaseModel, Json
import json

# Gestures
########################################################

class GestureBase(BaseModel):
    sequence_length:    int
    classification:     str
    hand_coordinates:   Json

    class Config:
        # arbitrary_types_allowed = True
        json_loads = json.loads
        json_dumps = json.dumps
    
class GestureCreate(GestureBase):
    pass

class Gesture(GestureBase):
    id: int

    class Config:
        orm_mode = True