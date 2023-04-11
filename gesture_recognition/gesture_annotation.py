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


class GestureAnnotation:  
    def __init__(self, gesture_list: List, video_length: int, debug = False):   
        self.debug = debug
        # self.db = db

        # Will store the video source, for an integrated camera this is cv2.VideoCapture(0) .  
        self.cap = None
        # Will store the current frame.
        self.frame = npt.NDArray
        # Store gestures in a dict:
        self.gesture_dict = {}

        # Store the last pressed key
        self.is_showing_key_press = False
        self.last_useful_key_press = None
        self.key_press_frame_idx = 0

        # Store the last recorded gesture.
        self.is_recording = False
        self.saved_gesture = npt.NDArray
        self.record_gesture_frame_idx = 0

        # Deque for storing the last video_length hand coordinates frames.
        self.last_hand_positions = deque(maxlen=video_length) 
        for _ in range(video_length):
            self.last_hand_positions.append(None)
        
        self.__assign_keys_to_gestures(gesture_list=gesture_list)
        
        # Set up database.
        # db_models.Base.metadata.create_all(bind=engine)


    def __del__(self):
        self.stop()


    def __get_db(self):
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


    def __start_webcam(self):
        # For webcam input:
        try:
            self.cap = cv2.VideoCapture(0)
        except:
            raise(RuntimeError("Error connecting to webcam."))


    def __assign_keys_to_gestures(self, gesture_list) -> None:
        # Assign gestures to the keys 1-9 then a-z.
        for idx, gesture in enumerate(gesture_list):
            if idx < 9:
                self.gesture_dict[str(idx + 1)] = gesture
            else:
                self.gesture_dict[chr((idx - 8) + ord('a'))] = gesture


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


    def __draw_red_recording_box(self) -> None:
        image_rows, image_cols, _ = self.frame.shape
        cv2.rectangle(self.frame, (0, 0), (image_cols-1, image_rows-1), color=(0,0,255), thickness=2)


    def __draw_key_press(self) -> None:
        # Draw key press on the image for VIDEO_LENGTH frames.
        if self.last_useful_key_press is not None:
            # Count frames since the last key press.
            self.key_press_frame_idx += 1

            if (self.key_press_frame_idx <= VIDEO_LENGTH):
                if (self.is_showing_key_press):
                    cv2.putText(
                        img=self.frame, 
                        text=f"Key press: { chr(self.last_useful_key_press) }", 
                        org=(20, 450), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=(0, 255, 0), 
                        thickness=1, 
                        lineType=cv2.LINE_AA, 
                        bottomLeftOrigin=False)
            else: # key_press_frame_idx is greater than the video length. Stop showing the pressed key.
                self.is_showing_key_press = False
            
    
    def __handle_key_press(self, key_press: int) -> None:
        # Do nothing on no key press
        if key_press == -1:
            pass
        # Exit on 'Esc'
        elif key_press == 27:
            self.stop()
        # For any other key press:
        else:
            # If a gesture being recorded, ignore the key press. 
            # Note that last_key_press will hold the same value until recording has finished.
            if self.is_recording:
                return
            # Store this key press
            self.is_showing_key_press = True
            self.last_useful_key_press = key_press
            self.key_press_frame_idx = 0

        self.__draw_key_press()


    def __handle_gesture_recording(self) -> None:
            # Check if recording should be started
            if not self.is_recording:
                if (self.last_useful_key_press is not None) and (chr(self.last_useful_key_press) in self.gesture_dict.keys()):
                    self.is_recording = True
                    self.record_gesture_frame_idx = 0
                else:
                    # Not recording, key press does not correspond to a gesture annotation: return.
                    return
            
            # Currently recording. Incriment frame counter
            self.record_gesture_frame_idx += 1
            
            # Draw red border around image when recording
            self.__draw_red_recording_box()

            if self.record_gesture_frame_idx >= VIDEO_LENGTH:
                # Save the last hand positions to the database:
                # print("Recorded data:\n", json.dumps(self.last_hand_positions, indent=2, cls=HandHistoryEncoder))
                
                # Write JSON of last hand coordinates to file for debugging. 
                if self.debug:
                    with open('coordinates.txt', 'w') as f:
                        f.write(json.dumps(self.last_hand_positions, indent=2, cls=HandHistoryEncoder))

                # Create new gesture entry in the database.
                create_gesture(
                    db=self.__get_db().__next__(),
                    gesture=pydantic_models.GestureCreate(
                        sequence_length=VIDEO_LENGTH,
                        classification=self.gesture_dict[chr(self.last_useful_key_press)],
                        hand_coordinates=json.dumps(self.last_hand_positions, cls=HandHistoryEncoder)
                    )
                )
                try:
                    pass
                except Exception as err:
                    print("Error creating gesture entry in database.", err)

                # Reset recording flag
                self.is_recording = False
                # Clear last key press to avoid overflowing the key_press_frame_idx counter.
                self.last_useful_key_press = None 

    
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
                self.frame.flags.writeable = True
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                self.__draw_hand_landmarks(self.frame, current_hand_landmarks)
                self.__draw_index_fingertip_landmarks_history()

                # Draw available gestures on the flipped image.
                self.__draw_gesture_list()

                # Get keyboard input
                key_press = cv2.waitKey(10)
                self.__handle_key_press(key_press)
                self.__handle_gesture_recording()

                cv2.imshow('Gesture Annotation', self.frame)


    def stop(self) -> int:
        try: 
            self.cap.release()
            return 0
        except:
            print("Error destructing.")
            return 1


if __name__ == "__main__":
    # Set up database.
    db_models.Base.metadata.create_all(bind=engine)

    ga = GestureAnnotation(gesture_list=GESTURE_LIST, video_length=VIDEO_LENGTH)
    ga.start()

