import cv2 #type: ignore 
import numpy as np #type: ignore 
from tensorflow.keras.models import load_model #type: ignore 
from sklearn.ensemble import RandomForestClassifier #type: ignore 
import winsound #type: ignore 
import joblib #type: ignore 
import os #type: ignore 

# Set environment variable to turn off OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check if the CNN model file exists
cnn_model_path = 'path/to/your/cnn_posture_model.h5'
if not os.path.exists(cnn_model_path):
    print(f"Error: CNN model file '{cnn_model_path}' not found.")
    exit()

# Load the trained CNN model
try:
    cnn_model = load_model(cnn_model_path)
except Exception as e:
    print(f"Error loading CNN model: {e}")
    exit()

# Check if the Random Forest model file exists
rf_model_path = 'path/to/your/rf_posture_model.pkl'
if not os.path.exists(rf_model_path):
    print(f"Error: Random Forest model file '{rf_model_path}' not found.")
    exit()

# Load the trained Random Forest model
try:
    rf_model = joblib.load(rf_model_path)
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to predict posture and calculate accuracy
def predict_posture(frame):
    try:
        # Preprocess frame
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # CNN prediction
        cnn_pred = cnn_model.predict(frame_expanded)
        cnn_label = np.argmax(cnn_pred)

        # Flatten frame for Random Forest
        frame_flattened = frame_resized.flatten().reshape(1, -1)
        rf_label = rf_model.predict(frame_flattened)

        # Combine predictions (simple averaging)
        combined_label = int(round((cnn_label + rf_label[0]) / 2))

        # Accuracy calculation
        cnn_accuracy = max(cnn_pred[0]) * 100
        rf_accuracy = rf_model.predict_proba(frame_flattened).max() * 100
        accuracy = (cnn_accuracy + rf_accuracy) / 2

        return combined_label, accuracy

    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0

# Alert system
bad_posture_count = 0
GOOD_POSTURE_COLOR = (0, 255, 0)
BAD_POSTURE_COLOR = (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    label, accuracy = predict_posture(frame)

    if label == -1:
        cv2.putText(frame, 'Error in Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display accuracy on webcam
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Posture classification and visual alerts
        if label == 0:  # Good posture
            cv2.putText(frame, 'Good Posture', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, GOOD_POSTURE_COLOR, 2)
            frame_color = GOOD_POSTURE_COLOR
            bad_posture_count = 0
        else:  # Bad posture
            cv2.putText(frame, 'Bad Posture', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, BAD_POSTURE_COLOR, 2)
            frame_color = BAD_POSTURE_COLOR
            bad_posture_count += 1

            # Sound alert if bad posture persists
            if bad_posture_count > 5:
                winsound.Beep(1000, 500)

        # Visual alert by coloring frame border
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=frame_color)

    cv2.imshow('Posture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
