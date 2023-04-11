from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.orm import validates

from .db import Base, GESTURE_LIST

class Gesture(Base):
    __tablename__ = "Gesture"

    id                  = Column(Integer, primary_key=True, index=True)
    sequence_length     = Column(Integer)
    classification      = Column(String(32))
    hand_coordinates    = Column(JSON)

    # Ensure the given classification is in the list of accepted gestures.
    @validates("classification")
    def validate_name(self, key, value):
        assert value in GESTURE_LIST
        return value
