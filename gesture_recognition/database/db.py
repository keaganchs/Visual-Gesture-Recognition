from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


"""
To allow for the recording of a new gesture, simply edit the GESTURE_LIST below.

Changing the `VIDEO_LENGTH` will require all entires in the database to have that number of frames, invalidating all current data.
If you want to change `VIDEO_LENGTH`, it may be a good idea to change the `SQLALCHEMY_DATABASE_URL` to something like "sqlite:///./database_v2.db".

TODO: Change to environment variables (import os.environ)
"""

# Number of frames
VIDEO_LENGTH = 30 

# Gestures which appear in the `gesture_annotation.py`. 
# This list should also be passed as an argument to `gesture_recognition.py` and `feature_importance.py`. 
GESTURE_LIST = ["SWIPE_LEFT", "SWIPE_RIGHT", "CW_CIRCLE", "CCW_CIRCLE"]
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"


"""
Below is some miscelaneous setup for the Sqlalchemy database. Changes will likely result in the database no longer functioning.
"""

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Return a generator object with the database.
# This may require calling Generator.__next__().
def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()