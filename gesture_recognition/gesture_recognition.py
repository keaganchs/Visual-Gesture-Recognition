import numpy as np
import numpy.typing as npt
from collections import deque
from typing import List

import cv2
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

from api.gestures import create_gesture, HandHistoryEncoder

from database.db import GESTURE_LIST, VIDEO_LENGTH, SessionLocal, engine
from database import db_models, pydantic_models

import json


class GestureRecognition:  
    def __init__(self, gesture_list: List, video_length: int, debug = False):   
        # Will store the video source, for an integrated camera this is cv2.VideoCapture(0) .  
        self.cap = None
        # Will store the current frame.
        self.frame = npt.NDArray
        # Store gestures in a dict:
        self.gesture_dict = {}

        # Deque for storing the last video_length hand coordinates frames.
        self.last_hand_positions = deque(maxlen=video_length) 
        for _ in range(video_length):
            self.last_hand_positions.append(None)

        # Bool to stop multiple gestures from being detected at once.
        self.is_gesture_detected = False
        # Store a detected gesture.
        self.detected_gesture = None
        # Count frames since the gesture was detected. 
        self.detected_gesture_frame_idx = None
        

    def __del__(self):
        self.stop()


    def __start_webcam(self):
        # For webcam input:
        try:
            self.cap = cv2.VideoCapture(0)
        except:
            raise(RuntimeError("Error connecting to webcam."))


    def __draw_hand_landmarks(self, image: np.ndarray, results) -> None:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, get_default_hand_landmarks_style(), get_default_hand_connections_style())


    def __draw_index_fingertip_landmarks_history(self) -> None:
        assert self.frame.flags.writeable
        
        points = []
        image_rows, image_cols, _ = self.frame.shape

        # Get index finger coordinates.
        for result in self.last_hand_positions:
            if result and result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    points.append((
                        # Pixel conversion from mediapipe.solutions.drawing_utils._normalized_to_pixel_coordinates():
                        int(min(np.floor(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_cols), image_cols - 1)),
                        int(min(np.floor(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_rows), image_rows - 1))
                    ))
            else:
                points.append(None)

        for i in range(1, len(points)):
            if points[i] and points[i-1]:
                cv2.line(self.frame, points[i], points[i-1], (0, 0, 255), 3)


    def __draw_gesture_list(self) -> None:
        for idx, key in enumerate(self.gesture_dict):
            cv2.putText(
                img=self.frame, 
                text=f"{key}: {self.gesture_dict[key]}", 
                org=(20, 20 + (20 * idx)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(0, 255, 0), 
                thickness=1, 
                lineType=cv2.LINE_AA, 
                bottomLeftOrigin=False)
         
    
    def __draw_detected_gesture(self) -> None:
        pass

    def __handle_key_press(self, key_press: int) -> None:
        # Do nothing on no key press
        if key_press == -1:
            pass
        # Exit on 'Esc'
        elif key_press == 27:
            self.stop()
        # For any other key press:
        else:
            # If a gesture is currently detected, ignore other key presses.
            if self.is_gesture_detected:
                return
            # Store this key press
            self.is_showing_key_press = True
            self.last_useful_key_press = key_press
            self.key_press_frame_idx = 0


    def __get_trained_model(self) -> None:
        # Get a trained model, if one exists.
        pass

    
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
                self.__draw_index_fingertip_landmarks_history()

                # Draw available gestures on the flipped image.
                self.__draw_gesture_list()

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
    ga = GestureRecognition(gesture_list=GESTURE_LIST, video_length=VIDEO_LENGTH)
    ga.start()

