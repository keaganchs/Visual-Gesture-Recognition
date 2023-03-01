from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import math

from typing import Union, Tuple

# Set up parameters
VIDEO_LENGTH = 30 # Number of frames

from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

import mediapipe.python.solutions.hands as mp_hands


def start_webcam() -> cv2.VideoCapture:
    # For webcam input:
    cap = cv2.VideoCapture(0)
    return cap

# def _normalized_to_pixel_coordinates(
#     normalized_x: float, normalized_y: float, image_width: int,
#     image_height: int) -> Union[None, Tuple[int, int]]:
#   """Converts normalized value pair to pixel coordinates."""

#   # Checks if the float value is between 0 and 1.
#   def is_valid_normalized_value(value: float) -> bool:
#     return (value > 0 or math.isclose(0, value)) and (value < 1 or
#                                                       math.isclose(1, value))

#   if not (is_valid_normalized_value(normalized_x) and
#           is_valid_normalized_value(normalized_y)):
#     # TODO: Draw coordinates even if it's outside of the image bounds.
#     return None
#   x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#   y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#   return x_px, y_px

def _draw_hand_landmarks(image: np.ndarray, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, get_default_hand_landmarks_style(), get_default_hand_connections_style())

def _draw_index_fingertip_landmarks_history(image: np.ndarray, history): # history: [iterable] of results
    assert image.flags.writeable
    
    points = []
    image_rows, image_cols, _ = image.shape # TODO: check image width height and other normalizing tasks in mediapipe.solutions.drawing_utils.draw_landmarks()
    
    # See mediapipe.solutions.drawing_utils._normalized_to_pixel_coordinates, pixel values come from
    # x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    # y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    

    # Get index finger coordinates
    for result in history:
        if result and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                points.append((
                    int(min(np.floor(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_rows), image_rows - 1)),
                    int(min(np.floor(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_cols), image_cols - 1))
                    # int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_rows),
                    # int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_cols)
                ))
        else: 
            points.append(None)

    for i in range(0, len(points)):
        cv2.circle(image, points[i], 1, (0, 0, 255), 5)

    for i in range(1, len(points)):
        if points[i] and points[i-1]:
            cv2.line(image, points[i], points[i-1], (0, 0, 255), 5)
            

def loop(cap: cv2.VideoCapture):
    with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
        
        # TODO: Test if using a list is faster than a deque as all elements must be accessed when drawing the history 
        last_hand_positions = deque(maxlen=VIDEO_LENGTH) # Deque for storing the last n hand coordinates frames
        for _ in range(VIDEO_LENGTH):
            last_hand_positions.append(None)

        # i = 0

        while cap.isOpened():
            success, image = cap.read()
            image = cv2.resize(image, (500, 500)) 

            if not success:
                print("Ignoring empty camera frame.")
                continue # If loading a video, use 'break' instead of 'continue'.

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image)

            # Record the last hand coordinates for active gesture processing
            last_hand_positions.popleft()
            last_hand_positions.append(results)
            
            # if results.multi_hand_landmarks and i >= 30:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         print('hand_landmarks:', hand_landmarks)
            #         print(
            #             f'Index finger tip coordinates: (',
            #             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, '
            #             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y} )'
            #         )
            #     i = 0
            # else: 
            #     i += 1
            

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _draw_hand_landmarks(image, results) # Add hand landmarks to image
            _draw_index_fingertip_landmarks_history(image, last_hand_positions) # Add index finger tip history to image

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            # Exit on 'Esc' 
            if cv2.waitKey(5) & 0xFF == 27:
                break


    cap.release()


if __name__ == "__main__":
    cap = start_webcam()
    loop(cap)

