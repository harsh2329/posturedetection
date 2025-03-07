import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
from posture_analysis import calculate_angle, check_posture  # Ensure this module exists

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Accuracy tracking
total_frames = 0
correct_predictions = 0

# Simulated ground truth (for demonstration) - Replace with actual data collection
# Format: [(angle, 'Good'/'Bad'), ...]
ground_truth = [(160, 'Good'), (145, 'Bad'), (170, 'Good')]

# Open webcam
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect for better usability
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract key points
            landmarks = results.pose_landmarks.landmark

            # Get specific joint coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, hip, knee)

            # Check posture
            posture_status = check_posture(angle)
            cv2.putText(frame, f"Posture: {posture_status}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if posture_status == "Good" else (0, 0, 255), 2)

            # Accuracy check (using simulated ground truth)
            if total_frames < len(ground_truth):
                true_angle, true_status = ground_truth[total_frames]
                if posture_status == true_status:
                    correct_predictions += 1
                total_frames += 1

        # Display accuracy
        if total_frames > 0:
            accuracy = (correct_predictions / total_frames) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
