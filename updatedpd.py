import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
import time
import argparse
import pygame # type: ignore
import os
from datetime import datetime, timedelta
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
        # Check if sound files exist, create if they don't
        self.posture_alert_sound = 'posture_alert.wav'
        self.movement_alert_sound = 'movement_alert.wav'
        if not os.path.exists(self.posture_alert_sound):
            self.create_alert_sound(self.posture_alert_sound, 800, 1000)
        if not os.path.exists(self.movement_alert_sound):
            self.create_alert_sound(self.movement_alert_sound, 600, 700, duration=0.3, repeats=3)
        
        # Define posture thresholds
        self.shoulder_threshold = 0.15  # Max acceptable shoulder tilt
        self.head_forward_threshold = 0.3  # Max acceptable head forward position
        self.neck_angle_threshold = 15.0  # Max acceptable neck angle deviation
        
        # Posture tracking stats
        self.posture_history = []
        self.session_start_time = datetime.now()
        
        # Movement tracking variables
        self.last_position = None
        self.movement_threshold = 0.05  # Minimum movement to be considered as changed position
        self.last_movement_time = time.time()
        self.movement_check_interval = 15 * 60  # 15 minutes in seconds (default)
        self.movement_alert_active = False
        self.movement_positions = []  # Track positions for movement analysis
        
        # Visualization parameters
        self.GOOD_POSTURE_COLOR = (0, 255, 0)  # Green
        self.BAD_POSTURE_COLOR = (0, 0, 255)   # Red
        self.WARNING_COLOR = (0, 165, 255)     # Orange
        self.TEXT_COLOR = (255, 255, 255)      # White
        self.BACKGROUND_COLOR = (50, 50, 50)   # Dark gray
        
        # UX parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.panel_alpha = 0.7  # Transparency of info panels
    
    def create_alert_sound(self, filename, freq1, freq2, duration=0.5, repeats=1):
        """Create a basic alert sound file with gentler tones"""
        # Generate a simple alert tone
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate a tone with a soft sine wave envelope
        envelope = np.sin(np.pi * t / duration)
        tone1 = np.sin(2 * np.pi * freq1 * t) * envelope
        tone2 = np.sin(2 * np.pi * freq2 * t) * envelope
        
        # Combine tones with short pauses if multiple repeats
        if repeats > 1:
            silence = np.zeros(int(sample_rate * 0.1))  # 0.1 second silence
            tone = np.array([])
            for _ in range(repeats):
                tone = np.concatenate([tone, tone1, silence, tone2, silence])
        else:
            tone = np.concatenate([tone1, tone2])
        
        # Normalize to 16-bit range with reduced volume (70%)
        tone = np.int16(tone / np.max(np.abs(tone)) * 32767 * 0.7)
        
        # Save as WAV
        write(filename, sample_rate, tone)
        print(f"Created {filename}")
        
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
        
        # Store current position for movement tracking (using mid_shoulder as reference)
        current_position = [mid_shoulder[0] / frame_width, mid_shoulder[1] / frame_height]
        if len(self.movement_positions) >= 10:
            self.movement_positions.pop(0)
        self.movement_positions.append(current_position)
        
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
        
        # Check for movement (if we have enough position history)
        if len(self.movement_positions) >= 5:
            self.check_movement()
        
        return good_posture, issues, frame, posture_score
    
    def check_movement(self):
        """Check if user has moved significantly in the specified time interval"""
        # Calculate average positions to reduce noise
        avg_pos = np.mean(self.movement_positions, axis=0)
        
        # If this is our first position check, initialize
        if self.last_position is None:
            self.last_position = avg_pos
            self.last_movement_time = time.time()
            return
        
        # Check if there's significant movement
        movement_detected = np.linalg.norm(np.array(avg_pos) - np.array(self.last_position)) > self.movement_threshold
        
        if movement_detected:
            self.last_position = avg_pos
            self.last_movement_time = time.time()
            self.movement_alert_active = False
    
    def format_time(self, seconds):
        """Format seconds into hours:minutes:seconds"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def create_info_panel(self, frame, x, y, width, height, alpha=0.7):
        """Create a semi-transparent panel for information display"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.BACKGROUND_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return x + 10, y + 30  # Return starting position for text
    
    def run(self):
        """Main loop for the posture detection system"""
        # Parse movement check interval from arguments (default is 15 minutes)
        parser = argparse.ArgumentParser(description='Posture Detection System')
        parser.add_argument('--movement-interval', type=int, default=15,
                          help='Interval in minutes to check for movement')
        parser.add_argument('--no-landmarks', action='store_false', dest='show_landmarks',
                          help='Disable visualization of pose landmarks')
        args = parser.parse_args()
        
        self.movement_check_interval = args.movement_interval * 60  # Convert to seconds
        self.show_landmarks = args.show_landmarks
        
        print(f"Movement check interval set to {args.movement_interval} minutes")
        
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
            
            # Create main info panel
            frame_height, frame_width, _ = frame.shape
            panel_width = 280
            panel_height = 130 if issues else 80
            x, y = self.create_info_panel(frame, 20, 20, panel_width, panel_height, self.panel_alpha)
            
            # Display posture status with enhanced styling
            status_text = "Good Posture" if good_posture else "Bad Posture"
            cv2.putText(frame, status_text, (x, y), 
                       self.font, 1, border_color, 2)
            
            # Display posture score with gauge-like visualization
            score_x = x
            score_y = y + 35
            cv2.putText(frame, f"Posture Score:", (score_x, score_y),
                       self.font, 0.7, self.TEXT_COLOR, 1)
            
            # Draw score bar
            bar_length = 200
            filled_length = int(bar_length * posture_score / 100)
            bar_height = 15
            bar_x = score_x
            bar_y = score_y + 10
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Score color based on value
            if posture_score > 75:
                score_color = self.GOOD_POSTURE_COLOR
            elif posture_score > 40:
                score_color = self.WARNING_COLOR
            else:
                score_color = self.BAD_POSTURE_COLOR
                
            # Filled bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), 
                         score_color, -1)
            
            # Score text
            cv2.putText(frame, f"{posture_score}%", (bar_x + bar_length + 10, bar_y + bar_height - 2),
                       self.font, 0.6, self.TEXT_COLOR, 1)
            
            # Display issues if any in a separate panel
            if issues:
                issues_y = y + 75
                for i, issue in enumerate(issues):
                    cv2.putText(frame, f"• {issue}", (x, issues_y + i * 25),
                              self.font, 0.6, self.BAD_POSTURE_COLOR, 1)
            
            # Create session info panel
            session_panel_width = 280
            session_panel_height = 80
            session_x, session_y = self.create_info_panel(frame, frame_width - session_panel_width - 20, 
                                                        20, session_panel_width, session_panel_height, 
                                                        self.panel_alpha)
            
            # Display session stats
            session_duration = (datetime.now() - self.session_start_time).seconds
            cv2.putText(frame, f"Session Duration:", (session_x, session_y),
                       self.font, 0.7, self.TEXT_COLOR, 1)
            cv2.putText(frame, self.format_time(session_duration), 
                       (session_x, session_y + 25), self.font, 0.7, self.TEXT_COLOR, 1)
            
            # Calculate and display overall posture quality
            if self.posture_history:
                posture_quality = sum(self.posture_history) / len(self.posture_history) * 100
                cv2.putText(frame, f"Overall Quality: {posture_quality:.1f}%", 
                           (session_x, session_y + 55), self.font, 0.7, self.TEXT_COLOR, 1)
            
            # Movement tracking panel
            if self.last_position is not None:
                movement_panel_height = 80
                movement_x, movement_y = self.create_info_panel(
                    frame, 20, frame_height - movement_panel_height - 20,
                    panel_width, movement_panel_height, self.panel_alpha)
                
                time_since_movement = time.time() - self.last_movement_time
                movement_remaining = max(0, self.movement_check_interval - time_since_movement)
                
                if movement_remaining > 0:
                    # Normal state - countdown to next required movement
                    cv2.putText(frame, "Movement Timer:", (movement_x, movement_y),
                              self.font, 0.7, self.TEXT_COLOR, 1)
                    
                    # Determine color based on remaining time
                    if movement_remaining < 60:  # Less than a minute
                        timer_color = self.WARNING_COLOR
                    else:
                        timer_color = self.TEXT_COLOR
                    
                    cv2.putText(frame, self.format_time(int(movement_remaining)), 
                              (movement_x, movement_y + 30), self.font, 0.8, timer_color, 1)
                else:
                    # Warning - need to move
                    cv2.putText(frame, "Time to Move!", (movement_x, movement_y),
                              self.font, 0.8, self.BAD_POSTURE_COLOR, 2)
                    cv2.putText(frame, f"Stationary for {self.format_time(int(time_since_movement))}", 
                              (movement_x, movement_y + 30), self.font, 0.7, self.BAD_POSTURE_COLOR, 1)
                    
                    # Check if we should trigger the movement alert
                    current_time = time.time()
                    if not self.movement_alert_active and (current_time - self.last_movement_time) > self.movement_check_interval:
                        try:
                            # Load and play the movement alert sound
                            pygame.mixer.music.load(self.movement_alert_sound)
                            pygame.mixer.music.play()
                            self.movement_alert_active = True
                        except Exception as e:
                            print(f"Movement alert sound error: {e}")
            
            # Sound alert for bad posture that persists
            current_time = time.time()
            if self.bad_posture_counter > 30 and not self.alert_active and (current_time - self.last_alert_time) > 5:
                try:
                    pygame.mixer.music.load(self.posture_alert_sound)
                    pygame.mixer.music.play()
                    self.alert_active = True
                    self.last_alert_time = current_time
                except Exception as e:
                    print(f"Posture alert sound error: {e}")
            
            if self.good_posture_counter > 5:
                self.alert_active = False
            
            # Track posture history (every second)
            if time.time() % 1 < 0.1:
                self.posture_history.append(good_posture)
                if len(self.posture_history) > 3600:  # Limit history to last hour
                    self.posture_history.pop(0)
            
            # Display controls help
            help_x, help_y = self.create_info_panel(
                frame, frame_width - 180, frame_height - 40, 160, 30, self.panel_alpha)
            cv2.putText(frame, "Press 'q' to quit", (help_x, help_y - 8),
                       self.font, 0.5, self.TEXT_COLOR, 1)
            
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
    parser.add_argument('--movement-interval', type=int, default=15,
                       help='Interval in minutes to check for movement (default: 15)')
    args = parser.parse_args()
    
    # Run the system
    system = PostureDetectionSystem(show_landmarks=args.show_landmarks)
    system.movement_check_interval = args.movement_interval * 60  # Convert to seconds
    system.run()