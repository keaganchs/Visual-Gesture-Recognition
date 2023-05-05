import numpy as np
import numpy.typing as npt
from collections import deque
from typing import List

import cv2
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

from api.gestures import HandHistoryEncoder, convert_video_to_array

from database.db import GESTURE_LIST, VIDEO_LENGTH

import json

import tensorflow as tf
# Use this import method for VS Code IntelliSense.
keras = tf.keras


class GestureRecognition:  
    def __init__(self, model_path: str, gesture_list: List, video_length: int, detection_threshold: float = 0.96, min_num_frames_before_detecting_again: int = -1, print_output: bool = False):   
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

        # Toggle if the raw output of the model should be printed.
        self.is_printing_output = print_output

        # Confidence threshold for detecting a gesture.
        self.threshold = detection_threshold

        # Bool to stop multiple gestures from being detected at once.
        self.is_waiting_to_detect = False

        # Store a detected gesture and the confidence in that prediction.
        self.prediction = None
        self.prediction_confidence = None
        
        # Count frames since the gesture was detected. 
        self.detected_gesture_frame_idx = 0
        # Do not detect another gesture for __ frames after a gesture has been detected.
        self.min_num_frames_before_detecting_again = min_num_frames_before_detecting_again
        

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


    def __draw_toggle_detection(self) -> None:
        # Show "Enable/Disable Detection".
        cv2.putText(
            img=self.frame, 
            text=("[spacebar]: Disable Detection" if self.is_trying_to_detect else "[spacebar]: Enable Detection"), 
            org=(200, 470), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(0, 255, 0), 
            thickness=1, 
            lineType=cv2.LINE_AA, 
            bottomLeftOrigin=False)


    def __draw_gesture_list(self) -> None:
        # Show "Available Gestures" on the image.
        cv2.putText(
            img=self.frame, 
            text="Available Gestures:", 
            org=(20, 20), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(0, 255, 0), 
            thickness=1, 
            lineType=cv2.LINE_AA, 
            bottomLeftOrigin=False)
        # Show available gestures from the given input.
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


    def __draw_detected_gesture(self) -> None:
        # Show the detected gesture.
        if (self.prediction is not None) and (self.prediction_confidence is not None):
            cv2.putText(
                img=self.frame, 
                text=f"Output of model: {self.prediction} with {self.prediction_confidence:0.2F}% confidence.", 
                org=(200, 450), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(0, 255, 0), 
                thickness=1, 
                lineType=cv2.LINE_AA, 
                bottomLeftOrigin=False)
            
            if self.prediction == "POINTING":
                self.__draw_index_fingertip_landmarks_history()


    def __handle_key_press(self, key_press: int) -> None:
        # Do nothing on no key press.
        if key_press == -1:
            pass
        # Exit on "Esc".
        elif key_press == 27:
            self.stop()
        # For any other key press:
        else:
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
        # Expand dimensions so the array can be processed as a Tensor.
        array_hand_positions = tf.expand_dims(array_hand_positions, axis=0)
        
        return array_hand_positions


    def __check_for_gesture(self) -> None:
        if self.model is not None:
            if not self.is_waiting_to_detect:
                # Format the last recorded frames (input).
                input = self.__preprocess_last_recorded_frames()
                # Run the last recorded frames through the model.
                output = self.model(input).numpy()[0]

                # If printing is enabled, print the raw value of each output node.
                if self.is_printing_output:
                    print(["{0:0.4f}".format(weight) for weight in output])

                # The values in the output nodes are confidences of each gesture being the input.
                # The prediction is the highest confidence. 
                prediction_idx = np.argmax(output)
                confidence = output[prediction_idx]
                
                # Return the name of the gesture if the confidence is above the threshold.
                if confidence > self.threshold:
                    self.prediction = self.gesture_list[prediction_idx]
                    self.prediction_confidence = confidence
                    self.is_waiting_to_detect = True
                    self.detected_gesture_frame_idx = 0

                    # Print output of model upon detection.
                    # if self.is_printing_output:
                        # print(f"Output of model: {self.prediction} with {self.prediction_confidence:0.4F}% confidence.")
                else:
                    self.prediction = None
                    self.prediction_confidence = None

            # Check if a gesture should be searched for on the next cycle.        
            if self.detected_gesture_frame_idx > self.min_num_frames_before_detecting_again:
                # If a gesture was detected a number of frames over the limit, reset the counters. 
                self.is_waiting_to_detect = False
                self.detected_gesture_frame_idx = 0
            else:    
                # If a gesture is currently detected for a number of frames under the frame limit, incriment the index.
                self.detected_gesture_frame_idx += 1



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

                # Draw hand landmarks.
                self.__draw_hand_landmarks(self.frame, current_hand_landmarks)

                # Draw available gestures.
                self.__draw_gesture_list()

                # Draw toggle option.
                self.__draw_toggle_detection()

                if self.is_trying_to_detect:
                    # Only detect one gesture at a time.
                    self.__check_for_gesture()
                    self.__draw_detected_gesture()
                else:
                    # Clear predicitons when stopping detection.
                    self.prediction = None
                    self.prediction_confidence = None

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
    gr = GestureRecognition(model_path="keras/best_model.h5", gesture_list=GESTURE_LIST, video_length=VIDEO_LENGTH, detection_threshold=0.97, min_num_frames_before_detecting_again=15, print_output=True)
    gr.start()

