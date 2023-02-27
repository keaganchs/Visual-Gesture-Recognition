from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# Set up parameters
VIDEO_LENGTH = 30 # Number of frames

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def start_webcam() -> cv2.VideoCapture:
    # For webcam input:
    cap = cv2.VideoCapture(0)
    return cap


def draw_hand_landmarks(image: cv2.VideoCapture, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

def draw_index_fingertip_landmarks_history(image: cv2.VideoCapture, history): # history: [iterable] of results
    points = []
    image_rows, image_cols, _ = image.shape # TODO: check image width height and other normalizing tasks in mediapipe.solutions.drawing_utils.draw_landmarks()
    assert image.flags.writeable

    # Get index finger coordinates
    for result in history:
        if result and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                points.append((
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        
        last_hand_positions = deque(maxlen=VIDEO_LENGTH) # Deque for storing the last n hand coordinates frames
        for _ in range(VIDEO_LENGTH):
            last_hand_positions.append(None)

        # i = 0

        while cap.isOpened():
            success, image = cap.read()
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
            #             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, '
            #             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * cap.get(cv2.CAP_PROP_FRAME_WIDTH)} )'
            #         )
            #     i = 0
            # else: 
            #     i += 1
            

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_hand_landmarks(image, results) # Add hand landmarks to image
            draw_index_fingertip_landmarks_history(image, last_hand_positions)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            # Exit on 'Esc' 
            if cv2.waitKey(5) & 0xFF == 27:
                break


    cap.release()


if __name__ == "__main__":
    cap = start_webcam()
    loop(cap)

