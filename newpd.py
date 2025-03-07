import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
import time
import argparse
import pygame # type: ignore
import os
from datetime import datetime
from scipy.io.wavfile import write # type: ignore

class PostureDetectionSystem:
    def __init__(self, show_landmarks=True):
        # Initialize MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize variables for tracking posture state
        self.bad_posture_counter = 0
        self.good_posture_counter = 0
        self.alert_active = False
        self.last_alert_time = time.time()
        self.show_landmarks = show_landmarks
        
        # Initialize sound for alerts
        pygame.mixer.init()
        # Check if sound file exists, create if it doesn't
        self.sound_file = 'alert.wav'
        if not os.path.exists(self.sound_file):
            self.create_alert_sound()
        pygame.mixer.music.load(self.sound_file)
        
        # Define posture thresholds
        self.shoulder_threshold = 0.15  # Max acceptable shoulder tilt
        self.head_forward_threshold = 0.3  # Max acceptable head forward position
        self.neck_angle_threshold = 15.0  # Max acceptable neck angle deviation
        
        # Posture tracking stats
        self.posture_history = []
        self.session_start_time = datetime.now()
        
        # Visualization parameters
        self.GOOD_POSTURE_COLOR = (0, 255, 0)  # Green
        self.BAD_POSTURE_COLOR = (0, 0, 255)   # Red
        self.TEXT_COLOR = (255, 255, 255)      # White
    
    def create_alert_sound(self):
        """Create a basic alert sound file"""
        # Generate a simple alert tone
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate a tone that alternates between two frequencies
        tone1 = np.sin(2 * np.pi * 800 * t)
        tone2 = np.sin(2 * np.pi * 1000 * t)
        tone = np.concatenate([tone1, tone2])
        
        # Normalize to 16-bit range
        tone = np.int16(tone / np.max(np.abs(tone)) * 32767)
        
        # Save as WAV
        write(self.sound_file, sample_rate, tone)
        print(f"Created {self.sound_file}")
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_landmarks(self, frame):
        """Process frame and detect pose landmarks"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get pose landmarks
        results = self.pose.process(frame_rgb)
        
        return results
    
    def check_posture(self, results, frame):
        """Check if posture is correct based on landmarks"""
        frame_height, frame_width, _ = frame.shape
        issues = []
        
        if not results.pose_landmarks:
            return False, ["No pose detected"], frame, 0
        
        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Check shoulder alignment (horizontal)
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height]
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height]
        
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / frame_height
        if shoulder_diff > self.shoulder_threshold:
            issues.append("Uneven shoulders")
            # Highlight problematic shoulders
            cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 15, self.BAD_POSTURE_COLOR, -1)
            cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 15, self.BAD_POSTURE_COLOR, -1)
        
        # Check neck angle
        nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * frame_width,
               landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * frame_height]
        left_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * frame_width,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * frame_height]
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                       (left_shoulder[1] + right_shoulder[1]) / 2]
        
        # Calculate head forward angle
        neck_angle = self.calculate_angle(left_ear, mid_shoulder, [mid_shoulder[0] + 100, mid_shoulder[1]])
        if abs(90 - neck_angle) > self.neck_angle_threshold:
            issues.append("Head too far forward")
            # Highlight problematic neck position
            cv2.line(frame, (int(nose[0]), int(nose[1])), (int(mid_shoulder[0]), int(mid_shoulder[1])), 
                    self.BAD_POSTURE_COLOR, 4)
        
        # Check for slouching
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height]
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_height]
        
        mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        
        # Calculate back angle (vertical alignment)
        back_angle = self.calculate_angle([mid_shoulder[0], 0], mid_shoulder, mid_hip)
        if back_angle < 160:  # Less than 160 degrees indicates slouching
            issues.append("Slouching")
            # Highlight problematic back position
            cv2.line(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 
                    (int(mid_hip[0]), int(mid_hip[1])), self.BAD_POSTURE_COLOR, 4)
        
        # Draw landmarks if enabled
        if self.show_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Calculate posture score (0-100)
        posture_score = 100
        if issues:
            posture_score -= 25 * len(issues)
            posture_score = max(posture_score, 0)
        
        good_posture = len(issues) == 0
        
        return good_posture, issues, frame, posture_score
    
    def run(self):
        """Main loop for the posture detection system"""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture video")
                break
            
            # Detect pose landmarks
            results = self.detect_landmarks(frame)
            
            # Check posture
            good_posture, issues, frame, posture_score = self.check_posture(results, frame)
            
            # Update counters
            if good_posture:
                self.good_posture_counter += 1
                self.bad_posture_counter = 0
                border_color = self.GOOD_POSTURE_COLOR
            else:
                self.bad_posture_counter += 1
                self.good_posture_counter = 0
                border_color = self.BAD_POSTURE_COLOR
            
            # Add visual border based on posture
            frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
            
            # Display posture status
            status_text = "Good Posture" if good_posture else "Bad Posture"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
            
            # Display posture score
            cv2.putText(frame, f"Posture Score: {posture_score}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 2)
            
            # Display issues if any
            if issues:
                for i, issue in enumerate(issues):
                    cv2.putText(frame, f"• {issue}", (10, 110 + i * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BAD_POSTURE_COLOR, 2)
            
            # Sound alert for bad posture that persists
            current_time = time.time()
            if self.bad_posture_counter > 30 and not self.alert_active and (current_time - self.last_alert_time) > 5:
                try:
                    pygame.mixer.music.play()
                    self.alert_active = True
                    self.last_alert_time = current_time
                except Exception as e:
                    print(f"Sound alert error: {e}")
            
            if self.good_posture_counter > 5:
                self.alert_active = False
            
            # Display session stats
            session_duration = (datetime.now() - self.session_start_time).seconds
            cv2.putText(frame, f"Session: {session_duration//60}m {session_duration%60}s", 
                       (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
            
            # Track posture history (every second)
            if time.time() % 1 < 0.1:
                self.posture_history.append(good_posture)
            
            # Calculate and display overall posture quality
            if self.posture_history:
                posture_quality = sum(self.posture_history) / len(self.posture_history) * 100
                cv2.putText(frame, f"Overall: {posture_quality:.1f}% good", 
                           (frame.shape[1] - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
            
            # Display the result
            cv2.imshow('Posture Detection System', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.pose.close()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        if self.posture_history:
            session_duration = (datetime.now() - self.session_start_time).seconds
            print(f"\nSession Summary:")
            print(f"Duration: {session_duration//60} minutes {session_duration%60} seconds")
            print(f"Overall posture quality: {sum(self.posture_history) / len(self.posture_history) * 100:.1f}% good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Posture Detection System')
    parser.add_argument('--no-landmarks', action='store_false', dest='show_landmarks',
                       help='Disable visualization of pose landmarks')
    args = parser.parse_args()
    
    # Run the system
    system = PostureDetectionSystem(show_landmarks=args.show_landmarks)
    system.run()