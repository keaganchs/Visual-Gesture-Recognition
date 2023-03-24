from sqlalchemy import Column, Integer, String, JSON

from .db import Base, GESTURE_LIST

class Gesture(Base):
    __tablename__ = "Gesture"

    id                  = Column(Integer, primary_key=True, index=True)
    sequence_length     = Column(Integer)
    classification      = Column(String(32))
    hand_coordinates    = Column(JSON)

