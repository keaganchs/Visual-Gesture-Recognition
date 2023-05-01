from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Literal

VIDEO_LENGTH = 30 # Number of frames
GESTURE_LIST = ["SWIPE_LEFT", "SWIPE_RIGHT", "CW_CIRCLE", "CCW_CIRCLE"]

SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
