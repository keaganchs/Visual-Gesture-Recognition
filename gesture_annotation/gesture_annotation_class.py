import time
import numpy as np
import numpy.typing as npt
from collections import deque
from typing import Union, Tuple, List, NamedTuple, Literal

import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

from helper_functions import time_this
# from ..database.db import GESTURE_LIST
from gesture_annotation import VIDEO_LENGTH

class GestureAnnotation:  
    def __init__(self, debug = False):     
        self.debug = debug

        # Will store the video source, for a webcam: cv2.VideoCapture(0)   
        self.cap = None
        # Will store the current frame
        self.frame = npt.NDArray

        # Store the last pressed key
        self.is_showing_key_press = False
        self.last_key_press = None
        self.key_press_frame_idx = 0

        # Store the last recorded gesture 
        self.is_recording = False
        self.saved_gesture = npt.NDArray
        self.record_gesture_frame_idx = 0

        # Deque for storing the last VIDEO_LENGTH hand coordinates frames
        self.last_hand_positions = deque(maxlen=VIDEO_LENGTH) 
        for _ in range(VIDEO_LENGTH):
            self.last_hand_positions.append(None)




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

        # Get index finger coordinates
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
            if not self.is_recording:
                self.is_showing_key_press = True
                self.last_key_press = key_press
                self.key_press_frame_idx = 0

        # Draw key press on the image for VIDEO_LENGTH frames.
        if self.last_key_press is not None:
            # Increase counter for frames since the 
            self.key_press_frame_idx += 1

            if (self.key_press_frame_idx <= VIDEO_LENGTH):
                if (self.is_showing_key_press):
                    cv2.putText(
                        img=self.frame, 
                        text=f"Key press: { chr(self.last_key_press) }", 
                        org=(50, 450), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(255, 0, 0), 
                        thickness=2, 
                        lineType=cv2.LINE_AA, 
                        bottomLeftOrigin=False)
            else:
                # Clear last key press to avoid overflowing the key_press_frame_idx counter.
                self.last_key_press = None
                self.is_showing_key_press = False


    # def __record_gesture(self) -> None:
    #         for gesture in GESTURE_LIST:


    #         else:
    #             self.is_recording = False
                


    def start(self) -> None:
        self.__start_webcam()

        with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
            while self.cap.isOpened():
                # Read frame from video feed
                success, self.frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                self.frame.flags.writeable = False
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                
                # Get hand landmarks from Mediapipe
                current_hand_landmarks = hands.process(self.frame)

                # Record the last hand coordinates for active gesture processing.
                self.last_hand_positions.popleft()
                self.last_hand_positions.append(current_hand_landmarks)

                # Draw hand annotations on the image.
                self.frame.flags.writeable = True
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                self.__draw_hand_landmarks(self.frame, current_hand_landmarks)
                self.__draw_index_fingertip_landmarks_history()
                
                # Flip the image horizontally for a selfie-view display:
                self.frame = cv2.flip(self.frame, 1)

                key_press = cv2.waitKey(5)
                self.__handle_key_press(key_press)

                cv2.imshow('Gesture Annotation', self.frame)

    def stop(self) -> int:
        try: 
            self.cap.release()
            return 0
        except:
            print("Error destructing. I'm sure glad this was written in a language with a garbage collector.")
            return 1
        

if __name__ == "__main__":
    ga = GestureAnnotation()
    ga.start()