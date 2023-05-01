# Real-time Visual Gesture Recognition
Code for my bachelor thesis at Jacobs (Constructor) University Bremen.

## Quick Start

1. Clone this repo and move into the new directory.  
2. Install the requirements: `python -m pip install -r requirements.txt`
3. Move into the gesture_recognition dir: `cd gesture_recognition`. This is necessary because some files will look for `database.db` in this dir.
4. Run the gesture recognition file `python gesture_recognition.py`
5. Press spacebar to start the gesture recognition. Available gestures are shown left side of the window,
detected gestures will be printed on the screen if the output of the model is above a certain threshold. 

## File Heirarchy
The files are organized such that all files that can be run are in the `./gesture_recognition` dir.  

```
.  
├── gesture_recognition/  
│   ├── api/  
│   │   ├── __init__.py  
│   │   ├── gestures.py  
│   │   └──helper_functions.py 
│   ├── database/  
│   │   ├── __init__.py  
│   │   ├── db_models.py  
│   │   ├── db.py  
│   │   └── pydantic_models.py  
│   ├── keras/  
│   │   └── best_model.h5  
│   ├── plots/  
│   │   └── matplotlib_plot.png  
│   ├── __init__.py  
│   ├── database.db  
│   ├── feature_importance.py  
│   ├── gesture_annotation.py  
│   ├── gesture_recognition.py  
│   ├── helper_functions.py  
│   └── train_neural_network.py  
├── .gitignore  
├── CITATION.cff  
└── requirements.txt  
```

For the most part, you will only be interested in three files: `gesture_annotation.py`, `train_neural_network.py`, and `gesture_recognition.py`.  

## Adding a New Gesture

1. Edit the list `GESTURE_LIST` in `./gesture_recognition/database/db.py`.  
This will automatically allow you to add new training data to the database with this annotation via `gesture_annotation.py`.  

2. Run `gesture_annotation.py` and press the assigned key to record the following 30 frames of video.
After creating a number of entries (the exact number doesn't necessarily matter, but for reference there are around 75 entries 
for predefined gestures, at least 25 entries is recommended).  

3. Run the file `train_neural_network.py`. This will run create a train/test split of the data in the database, 
then run hyperparameter optimization to determine an ideal model. This optimal model will then be trained on the entire dataset.  

4. Now that everything is set up, just run `gesture_recognition.py`, and press the spacebar to start recognizing gestures. You should see your 
new entry on the screen with the other available gestures.
