import cv2
import mediapipe as mp

# Initialize the Mediapipe hand detection module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the OpenCV video capture
cap = cv2.VideoCapture(0)

# Set up the output window
cv2.namedWindow('Hand Gestures')

with mp_hands.Hands(
    max_num_hands=1,  # Set to 1 for single hand tracking
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the image
        results = hands.process(image)

        # Draw landmarks and connections on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the gesture label from the Mediapipe output
                gesture = None
                if results.multi_handedness:
                    handedness = results.multi_handedness[0]
                    gesture = handedness.classification[0].label

                # Add the gesture label to the output image
                if gesture:
                    cv2.putText(frame, gesture, (50, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the output image
        cv2.imshow('Hand Gestures', frame)

        # Wait for key press and exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
