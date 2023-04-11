from pydantic import BaseModel, Json

# Gestures
########################################################

class GestureBase(BaseModel):
    sequence_length:    int
    classification:     str
    hand_coordinates:   Json

    class Config:
        arbitrary_types_allowed = True
    
class GestureCreate(GestureBase):
    pass

class Gesture(GestureBase):
    id: int

    class Config:
        orm_mode = True