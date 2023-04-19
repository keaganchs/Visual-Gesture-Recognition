import numpy as np
import numpy.typing as npt
from collections import deque
from typing import List

import cv2
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

from api.gestures import HandHistoryEncoder, convert_video_to_array

from database.db import GESTURE_LIST, VIDEO_LENGTH, SessionLocal, engine
from database import db_models, pydantic_models

import json

import tensorflow as tf
# Use this import method for VS Code IntelliSense
keras = tf.keras


class GestureRecognition:  
    def __init__(self, model_path: str, gesture_list: List, video_length: int, debug = False):   
        # Will store the video source, for an integrated camera this is cv2.VideoCapture(0) .  
        self.cap = None
        # Will store the current frame.
        self.frame = npt.NDArray

        # Deque for storing the last video_length hand coordinates frames.
        self.last_hand_positions = deque(maxlen=video_length) 
        for _ in range(video_length):
            self.last_hand_positions.append(None)

        # Store the machine learning model.
        self.model = None
        # Get the model from the given path.
        try:
            self.load_model(model_path)
        except:
            raise RuntimeError("Error fetching model. Double-check the path, or if no model exists, run the file gesture_recognition/train_neural_network.py.")

        # Store gesture list to draw available gestures on the screen.
        self.gesture_list = gesture_list

        # Toggle if the gesture recognition model should be called.
        self.is_trying_to_detect = False

        # Bool to stop multiple gestures from being detected at once.
        self.is_gesture_detected = False

        # Store a detected gesture.
        self.detected_gesture = None
        
        # Count frames since the gesture was detected. 
        self.detected_gesture_frame_idx = None
        

    def __del__(self):
        self.stop()


    def __start_webcam(self) -> None:
        # For webcam input:
        try:
            self.cap = cv2.VideoCapture(0)
        except:
            raise(RuntimeError("Error connecting to webcam."))


    def __draw_hand_landmarks(self, image: np.ndarray, results) -> None:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, get_default_hand_landmarks_style(), get_default_hand_connections_style())


    def __draw_gesture_list(self) -> None:
        cv2.putText(
            img=self.frame, 
            text="Available Gestures:", 
            org=(20, 20), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(0, 255, 0), 
            thickness=1, 
            lineType=cv2.LINE_AA, 
            bottomLeftOrigin=False)
        # Write available gestures from the input.
        for idx, gesture in enumerate(self.gesture_list):
            cv2.putText(
                img=self.frame, 
                text=f"{gesture}", 
                org=(20, 40 + (20 * idx)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(0, 255, 0), 
                thickness=1, 
                lineType=cv2.LINE_AA, 
                bottomLeftOrigin=False)
         
    
    def __draw_detected_gesture(self) -> None:
        if self.detected_gesture is not None:
            cv2.putText(
                img=self.frame, 
                text=f"{self.detected_gesture}", 
                org=(200, 450), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(0, 255, 0), 
                thickness=1, 
                lineType=cv2.LINE_AA, 
                bottomLeftOrigin=False)


    def __handle_key_press(self, key_press: int) -> None:
        # Do nothing on no key press
        if key_press == -1:
            pass
        # Exit on 'Esc'
        elif key_press == 27:
            self.stop()
        # For any other key press:
        else:
            print("Key press: ", key_press)
            
            # On spacebar, toggle self.is_trying_to_detect.
            if key_press == 32:
                self.is_trying_to_detect = not self.is_trying_to_detect


    # TODO: Optimize.
    def __preprocess_last_recorded_frames(self) -> npt.ArrayLike:
        # HandHistoryEncoder formats to JSON string.
        json_hand_positions=json.dumps(self.last_hand_positions, cls=HandHistoryEncoder)
        # Convert formatted data into a python dict.
        dict_hand_positions = json.loads(json_hand_positions)
        # Convert to array.
        array_hand_positions = convert_video_to_array(dict_hand_positions)
        # Convert to Tensor and return.
        # tensor = tf.convert_to_tensor(array_hand_positions, dtype=tf.float32)
        array_hand_positions = tf.expand_dims(array_hand_positions, axis=0)
        return array_hand_positions


    def __check_for_gesture(self) -> (str | None):
        if self.model is not None:
            input = self.__preprocess_last_recorded_frames()
            print(input)
            output = self.model(input)
            print(output)
            return output
        

    def load_model(self, model_path: str) -> None:
        # Load model from given path.
        try:
            new_model = keras.models.load_model(model_path, compile=False)
        except: 
            # Raise a warning if model can not be found.
            raise RuntimeWarning("New model could not be loaded.")
        
        # Compile model.
        try:
            new_model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
        except:
            raise RuntimeWarning("Issue compiling model. No changes to the current model have been made.")
        
        self.model = new_model

    
    def start(self) -> None:
        self.__start_webcam()

        with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
            while self.cap.isOpened():
                # Read frame from video feed.
                success, self.frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Flip the image horizontally for a selfie-view display:
                self.frame = cv2.flip(self.frame, 1)

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                self.frame.flags.writeable = False
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                
                # Get hand landmarks from Mediapipe.
                current_hand_landmarks = hands.process(self.frame)

                # Record the last hand coordinates for active gesture processing.
                self.last_hand_positions.popleft()
                self.last_hand_positions.append(current_hand_landmarks)

                # Draw hand annotations on the image.
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                
                # Mark image as writable again to draw on the frame.
                self.frame.flags.writeable = True

                # Draw hand landmarks and index fingertip history.
                self.__draw_hand_landmarks(self.frame, current_hand_landmarks)

                # Draw available gestures on the flipped image.
                self.__draw_gesture_list()
                self.__draw_detected_gesture()

                if self.is_trying_to_detect:
                    # Only detect one gesture at a time
                    if not self.is_gesture_detected:
                        output = self.__check_for_gesture()
                        print("Output of model: ", output)

                # Get keyboard input.
                key_press = cv2.waitKey(10)
                self.__handle_key_press(key_press)

                cv2.imshow('Gesture Recognition', self.frame)


    def stop(self) -> int:
        try: 
            self.cap.release()
            return 0
        except:
            print("Error destructing.")
            return 1


if __name__ == "__main__":
    gr = GestureRecognition(model_path="keras/best_model.h5", gesture_list=GESTURE_LIST, video_length=VIDEO_LENGTH)
    gr.start()

