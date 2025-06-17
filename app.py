import cv2
import torch
import numpy as np
from scipy.spatial import distance
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import pickle
from uuid import uuid4
import sys
import os
import time
import mimetypes
import logging
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, Response, request, render_template, make_response, send_file, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
import io
import tempfile
import random
import bcrypt
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER.setLevel(logging.ERROR)

# Flask app initialization
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', str(uuid4()))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(CURR_PATH, 'Uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB limit
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Twilio configuration
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
ADMIN_PHONE_NUMBER = os.getenv('ADMIN_PHONE_NUMBER')
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN) if TWILIO_SID and TWILIO_AUTH_TOKEN else None

# Language dictionaries
LANGUAGES = {
    'en': {
        'title': 'Factory Worker Behavior Recognition',
        'header': 'AI Model for Recognizing Factory Worker Behavior',
        'dashboard_tab': 'Dashboard',
        'camera_tab': 'Camera Input',
        'video_tab': 'Video Input',
        'settings_tab': 'Settings',
        'logout': 'Logout',
        'daily_chart_title': 'Hazardous Behaviors Per Hour (Today)',
        'monthly_chart_title': 'Hazardous Behaviors Per Day (This Month)',
        'detection_log': 'Detection Log',
        'video_detection_log': 'Video Detection Log',
        'start_button': 'Start',
        'stop_button': 'Stop',
        'shutdown_button': 'Shutdown',
        'save_log_button': 'Save Log',
        'clear_log_button': 'Clear Log',
        'upload_video_button': 'Upload Video',
        'predict_actions_button': 'Predict Actions',
        'clear_video_log_button': 'Clear Log',
        'save_results_button': 'Save Results',
        'admin_phone_label': 'Admin Phone Number for SMS Alerts',
        'alert_sensitivity_label': 'Alert Sensitivity',
        'save_settings_button': 'Save Settings',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High',
        'language_label': 'Language',
        'frame_size_label': 'Frame Size',
        'hazard_alert': 'Hazardous Behavior Detected: {action} by Person {id} at {time}'
    },
    'ko': {
        'title': '공장 근로자 행동 인식',
        'header': '공장 근로자 행동 인식을 위한 AI 모델',
        'dashboard_tab': '대시보드',
        'camera_tab': '카메라 입력',
        'video_tab': '비디오 입력',
        'settings_tab': '설정',
        'logout': '로그아웃',
        'daily_chart_title': '시간별 위험 행동 (오늘)',
        'monthly_chart_title': '일별 위험 행동 (이번 달)',
        'detection_log': '탐지 로그',
        'video_detection_log': '비디오 탐지 로그',
        'start_button': '시작',
        'stop_button': '중지',
        'shutdown_button': '종료',
        'save_log_button': '로그 저장',
        'clear_log_button': '로그 지우기',
        'upload_video_button': '비디오 업로드',
        'predict_actions_button': '행동 예측',
        'clear_video_log_button': '로그 지우기',
        'save_results_button': '결과 저장',
        'admin_phone_label': 'SMS 알림용 관리자 전화번호',
        'alert_sensitivity_label': '알림 민감도',
        'save_settings_button': '설정 저장',
        'low': '낮음',
        'medium': '중간',
        'high': '높음',
        'language_label': '언어',
        'frame_size_label': '프레임 크기',
        'hazard_alert': '위험 행동 탐지됨: {id}번 사람이 {time}에 {action} 수행'
    }
}

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global variables
camera_on = False
cap = None
webcam_index = 0
frame_width = 640
frame_height = 480
detections = []
video_detections = []
last_time = time.time()
video_file_path = None
video_predict_writer = None
video_predict_temp = None
video_predict_progress = 0
last_alert_time = 0
alert_cooldown = 300
sequence_buffer = []
sequence_length = 30
num_keypoints = 17
next_id = 1
prev_centroids = []
action_chart_data = []

# Model configuration
MODEL_PATH = "models/convlstm_model.pth"
LABEL_MAP_PATH = "data/label_mapping.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8l.pt", task='detect').to('cpu')
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(DEVICE)
pose_model.eval()

import torch
import torch.nn as nn

# -- Ánh xạ tên hành động
ACTION_NAME_MAP = {
    (0, 4): "Walking",
    (0, 21): "Transition state",
    (0, 28): "Lay down",
    (0, 31): "Standing",
    (0, 33): "Check-in",
    (0, 34): "Check-out",
    (0, 35): "Carry object",
    (0, 36): "Work in progress",
    (0, 37): "Falling down",
    (22, 31): "Pick up a tool",
    (23, 31): "Put down a tool",
    (24, 31): "setup machine"
}

class ConvLSTM_Model(nn.Module):
    def __init__(self, in_channels=2, num_class=12, sequence_length=30, num_keypoints=17):
        super(ConvLSTM_Model, self).__init__()
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints

        # 1D Convolutional layers with residual connections
        self.conv1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.3)

        # Residual adapters
        self.res_adapter1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=1) if in_channels * num_keypoints != 64 else nn.Identity()
        self.res_adapter2 = nn.Conv1d(64, 128, kernel_size=1) if 64 != 128 else nn.Identity()

        # LSTM input size
        lstm_in_size = 128  # Output channels from conv4

        # LSTM layer
        self.lstm = nn.LSTM(lstm_in_size, 256, num_layers=2, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Input x: [B, T, V*C] -> Reshape to [B, V*C, T]
        B, T, VC = x.shape
        x = x.view(B, T, self.num_keypoints, -1)  # [B, T, V, C]
        x = x.permute(0, 3, 2, 1).reshape(B, VC, T)  # [B, V*C, T]

        # Convolutional layers with residual connections
        residual = self.res_adapter1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 64, T//2]
        x = self.dropout_conv(x)

        residual = self.res_adapter2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 128, T//4]
        x = self.dropout_conv(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # [B, T//4, 128]

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [B, T//4, 256]
        x = lstm_out[:, -1, :]  # Last time step [B, 256]

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

# Load ConvLSTM model and label mapping
model = ConvLSTM_Model(
    in_channels=2,
    num_class=len(ACTION_NAME_MAP),
    sequence_length=sequence_length,
    num_keypoints=num_keypoints
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

action_to_name = {action_label: ACTION_NAME_MAP.get(pair, "Unknown") for pair, action_label in label_map.items()}

# -- Cấu trúc skeleton (COCO format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (3, 5), (4, 6),  # Tiếp tục phần Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (5, 11), (6, 12)  # Torso
]

# -- Tách các kết nối thuộc phần Head
HEAD_CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

def resize_with_padding(frame, target_width, target_height):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def normalize_keypoints(keypoints, bbox):
    logging.info(f"Input keypoints shape: {keypoints.shape}")
    x_min, y_min, width, height = bbox
    normalized = []
    for i in range(0, len(keypoints.flatten()), 3):
        x, y = keypoints.flatten()[i], keypoints.flatten()[i + 1]
        x_norm = (x - x_min) / width if width > 1 else x / frame_width
        y_norm = (y - y_min) / height if height > 1 else y / frame_height
        normalized.extend([x_norm, y_norm])
    normalized = np.array(normalized).reshape(num_keypoints, 2)
    logging.info(f"Normalized keypoints shape: {normalized.shape}")
    return normalized

def assign_id(bbox, prev_centroids, max_distance=100):
    global next_id
    centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if not prev_centroids:
        prev_centroids.append((next_id, centroid))
        return str(next_id), prev_centroids
    distances = [distance.euclidean(centroid, prev[1]) for prev in prev_centroids]
    if min(distances) < max_distance:
        idx = np.argmin(distances)
        prev_centroids[idx] = (prev_centroids[idx][0], centroid)
        return str(prev_centroids[idx][0]), prev_centroids
    prev_centroids.append((next_id, centroid))
    next_id += 1
    return str(next_id - 1), prev_centroids

def draw_skeleton(frame, keypoints, connections, bbox=None, person_id="1", action_label="Unknown"):
    keypoints = keypoints.reshape(-1, 3)  # [x, y, confidence]
    
    # Vẽ các kết nối skeleton với màu sắc khác nhau
    for start_idx, end_idx in connections:
        if keypoints[start_idx, 2] > 0.1 and keypoints[end_idx, 2] > 0.1:
            start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
            end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
            # Nếu kết nối thuộc phần đầu (Head), vẽ màu cam
            if (start_idx, end_idx) in HEAD_CONNECTIONS or (end_idx, start_idx) in HEAD_CONNECTIONS:
                color = (14, 107, 255)  # Màu cam
            else:
                color = (14, 255, 30)  # Màu xanh nước biển
            cv2.line(frame, start_point, end_point, color, 2)

    # Vẽ các điểm keypoints (màu đỏ)
    for i in range(len(keypoints)):
        if keypoints[i, 2] > 0.1:
            cv2.circle(frame, (int(keypoints[i, 0]), int(keypoints[i, 1])), 5, (0, 0, 255), -1)  # Điểm đỏ

    if bbox is not None:
        x_min, y_min, width, height = bbox
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min + width), int(y_min + height)), (255, 0, 0), 2)  # Blue bbox
        
    return frame

def send_sms_alert(message):
    global last_alert_time, alert_cooldown
    current_time = time.time()
    if twilio_client and ADMIN_PHONE_NUMBER:
        if current_time - last_alert_time >= alert_cooldown:
            try:
                twilio_client.messages.create(
                    body=message,
                    from_=TWILIO_PHONE_NUMBER,
                    to=ADMIN_PHONE_NUMBER
                )
                last_alert_time = current_time
                logging.info(f"SMS sent: {message}")
            except Exception as e:
                logging.error(f"Failed to send SMS: {str(e)}")

def generate_simulated_detections():
    simulated_detections = []
    today = datetime.now()
    month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for hour in range(24):
        num_incidents = random.randint(0, 8)
        for _ in range(num_incidents):
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            action = random.choice(["Falling down", "Lay down"])
            timestamp = today.replace(hour=hour, minute=minute, second=second)
            detection = {
                "person_id": random.randint(1, 10),
                "action": action,
                "time": timestamp.strftime("%H:%M:%S"),
                "full_time": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            simulated_detections.append(detection)
    days_in_month = (today - month_start).days + 1
    for day in range(1, days_in_month + 1):
        num_incidents = random.randint(0, 10)
        for _ in range(num_incidents):
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            action = random.choice(["Falling down", "Lay down"])
            timestamp = month_start + timedelta(days=day-1, hours=hour, minutes=minute, seconds=second)
            detection = {
                "person_id": random.randint(1, 10),
                "action": action,
                "time": timestamp.strftime("%H:%M:%S"),
                "full_time": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            simulated_detections.append(detection)
    return simulated_detections

detections = generate_simulated_detections()

def generate_frames():
    global cap, camera_on, frame_width, frame_height, detections, last_time, sequence_buffer, prev_centroids, next_id
    ith_img = 0
    while True:
        if not camera_on or cap is None or not cap.isOpened():
            placeholder = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            text = "Connect Camera"
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 1.0
            text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.putText(
                placeholder, text, (text_x, text_y),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            logging.warning("Cannot read frame from webcam")
            continue

        frame = resize_with_padding(frame, frame_width, frame_height)
        ith_img += 1
        img = frame.copy()

        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time != last_time else 0
        last_time = current_time
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            rgb = img[:, :, ::-1]
            results = yolo_model(rgb)[0]
            person_boxes_xyxy = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if int(cls.item()) == 0:
                    person_boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])

            if not person_boxes_xyxy:
                logging.info("No persons detected in frame")
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            person_boxes_xywh = np.array(person_boxes_xyxy)
            person_boxes_xywh[:, 2] -= person_boxes_xywh[:, 0]
            person_boxes_xywh[:, 3] -= person_boxes_xywh[:, 1]

            inputs = pose_processor(rgb, boxes=[person_boxes_xywh], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = pose_model(**inputs)
                results_pose = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_xywh])[0]

            for i, pose_result in enumerate(results_pose):
                keypoints = pose_result["keypoints"].cpu().numpy()
                scores = pose_result["scores"].cpu().numpy()
                keypoints_with_conf = np.concatenate([keypoints, scores[..., None]], axis=-1)
                bbox = person_boxes_xywh[i]
                person_id, prev_centroids = assign_id(bbox, prev_centroids)

                norm_keypoints = normalize_keypoints(keypoints_with_conf, bbox)
                sequence_buffer.append(norm_keypoints.flatten())

                action_label = "Unknown"
                if len(sequence_buffer) >= sequence_length:
                    input_data = np.array(sequence_buffer[-sequence_length:]).reshape(1, sequence_length, num_keypoints * 2)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        prediction = model(input_tensor)
                        predicted_class = torch.argmax(prediction, dim=1).item()
                        action_xx = sorted(list(set(label_map.values())))[predicted_class]
                        action_label = action_to_name.get(action_xx, "Unknown")
                    sequence_buffer.pop(0)

                img = draw_skeleton(img, keypoints_with_conf, SKELETON_CONNECTIONS, bbox, person_id, action_label)

                cv2.putText(frame, f"Action: {action_label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if action_label in ["Falling down", "Lay down"]:
                    current_time_str = time.strftime("%H:%M:%S")
                    detection_entry = {
                        "person_id": person_id,
                        "action": action_label,
                        "time": current_time_str,
                        "full_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if not detections or detections[-1] != detection_entry:
                        detections.append(detection_entry)
                        logging.info(f"Detection recorded: {detection_entry}")
                        language = session.get('language', 'en')
                        sms_message = LANGUAGES[language]['hazard_alert'].format(
                            action=action_label, id=person_id, time=current_time_str
                        )
                        send_sms_alert(sms_message)
                        socketio.emit('hazard_alert', {
                            'message': sms_message,
                            'time': detection_entry['full_time']
                        })

            if len(detections) > 10:
                detections.pop(0)

        except Exception as e:
            logging.error(f"Error in action recognition: {str(e)}")

        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_video_predict_frames():
    global video_file_path, video_detections, video_predict_writer, video_predict_temp, video_predict_progress, frame_width, frame_height, prev_centroids, next_id, action_chart_data
    if not video_file_path or not os.path.exists(video_file_path):
        placeholder = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        text = "No Prediction"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.0
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(
            placeholder, text, (text_x, text_y),
            font, font_scale, (255, 255, 255), 2, cv2.LINE_AA
        )
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    cap_video = cv2.VideoCapture(video_file_path)
    if not cap_video.isOpened():
        logging.error("Cannot open video file")
        return

    orig_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Original video size: {orig_width}x{orig_height}")
    logging.info(f"Output video size: {frame_width}x{frame_height}")

    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_video.get(cv2.CAP_PROP_FPS) or 30

    if video_predict_temp is None:
        video_predict_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_predict_writer = cv2.VideoWriter(video_predict_temp.name, fourcc, fps, (frame_width, frame_height))
        logging.info(f"Video writer initialized with size: {frame_width}x{frame_height}")

    ith_img = -1
    video_predict_progress = 0
    video_sequence_buffer = []
    prev_centroids = []
    next_id = 1
    action_chart_data = []

    while cap_video.isOpened():
        success, frame = cap_video.read()
        if not success:
            break

        frame = resize_with_padding(frame, frame_width, frame_height)
        ith_img += 1
        img = frame.copy()

        video_predict_progress = (ith_img / total_frames) * 100

        try:
            rgb = img[:, :, ::-1]
            results = yolo_model(rgb)[0]
            person_boxes_xyxy = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if int(cls.item()) == 0:
                    person_boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])

            if not person_boxes_xyxy:
                action_chart_data.append({"frame": ith_img, "action": "No Person"})
                if video_predict_writer is not None:
                    video_predict_writer.write(img)
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            person_boxes_xywh = np.array(person_boxes_xyxy)
            person_boxes_xywh[:, 2] -= person_boxes_xywh[:, 0]
            person_boxes_xywh[:, 3] -= person_boxes_xywh[:, 1]

            inputs = pose_processor(rgb, boxes=[person_boxes_xywh], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = pose_model(**inputs)
                results_pose = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_xywh])[0]

            for i, pose_result in enumerate(results_pose):
                keypoints = pose_result["keypoints"].cpu().numpy()
                scores = pose_result["scores"].cpu().numpy()
                keypoints_with_conf = np.concatenate([keypoints, scores[..., None]], axis=-1)
                bbox = person_boxes_xywh[i]
                person_id, prev_centroids = assign_id(bbox, prev_centroids)

                norm_keypoints = normalize_keypoints(keypoints_with_conf, bbox)
                video_sequence_buffer.append(norm_keypoints.flatten())

                action_label = "Unknown"
                if len(video_sequence_buffer) >= sequence_length:
                    input_data = np.array(video_sequence_buffer[-sequence_length:]).reshape(1, sequence_length, num_keypoints * 2)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        prediction = model(input_tensor)
                        predicted_class = torch.argmax(prediction, dim=1).item()
                        action_xx = sorted(list(set(label_map.values())))[predicted_class]
                        action_label = action_to_name.get(action_xx, "Unknown")
                    video_sequence_buffer.pop(0)
                    if not action_chart_data or action_chart_data[-1]["frame"] != ith_img:
                        action_chart_data.append({"frame": ith_img, "action": action_label})

                img = draw_skeleton(img, keypoints_with_conf, SKELETON_CONNECTIONS, bbox, person_id, action_label)
                
                cv2.putText(frame, f"Action: {action_label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if action_label in ["Falling down", "Lay down"]:
                    current_time = ith_img / fps
                    detection_entry = {
                        "person_id": person_id,
                        "action": action_label,
                        "time": f"{current_time:.2f}s",
                        "full_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if not video_detections or video_detections[-1] != detection_entry:
                        video_detections.append(detection_entry)
                        logging.info(f"Video detection recorded: {detection_entry}")
                        language = session.get('language', 'en')
                        sms_message = LANGUAGES[language]['hazard_alert'].format(
                            action=action_label, id=person_id, time=f"{current_time:.2f}s"
                        )
                        send_sms_alert(sms_message)
                        socketio.emit('hazard_alert', {
                            'message': sms_message,
                            'time': detection_entry['full_time']
                        })

            if len(video_detections) > 10:
                video_detections.pop(0)

            if video_predict_writer is not None:
                video_predict_writer.write(img)

        except Exception as e:
            logging.error(f"Error in video action recognition: {str(e)}")

        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap_video.release()
    if video_predict_writer is not None:
        video_predict_writer.release()
        video_predict_writer = None
    video_predict_progress = 100

with app.app_context():
    db.create_all()

@app.route('/register', methods=['GET', 'POST'])
def register():
    language = session.get('language', 'en')
    translations = LANGUAGES[language]
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not username or not password or not confirm_password:
            flash('All fields are required.', 'error')
            return render_template('register.html', translations=translations)
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html', translations=translations)
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('register.html', translations=translations)
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', translations=translations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    language = session.get('language', 'en')
    translations = LANGUAGES[language]
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            session['language'] = language
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html', translations=translations)
    return render_template('login.html', translations=translations)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('language', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/set_language', methods=['POST'])
@login_required
def set_language():
    language = request.form.get('language')
    if language in LANGUAGES:
        session['language'] = language
    return redirect(url_for('index'))

@app.route('/')
@login_required
def index():
    language = session.get('language', 'en')
    translations = LANGUAGES[language]
    return render_template('index.html', detections=detections, video_detections=video_detections, translations=translations, username=current_user.username, ADMIN_PHONE_NUMBER=ADMIN_PHONE_NUMBER)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_predict_feed')
@login_required
def video_predict_feed():
    return Response(generate_video_predict_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_predict_progress')
@login_required
def get_video_predict_progress():
    global video_predict_progress
    return {"progress": video_predict_progress}

@app.route('/get_action_chart_data')
@login_required
def get_action_chart_data():
    global action_chart_data
    return {"data": action_chart_data}

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    global video_file_path, video_predict_writer, video_predict_temp, video_predict_progress
    if 'file' not in request.files:
        logging.error("No file part in request")
        return {"status": "error", "message": "No file part in request"}, 400

    file = request.files['file']
    chunk_index = int(request.form.get('chunk', 0))
    total_chunks = int(request.form.get('total_chunks', 1))
    filename = request.form.get('name', '')
    file_id = request.form.get('file_id', str(uuid4()))

    if not filename:
        logging.error("No file name provided")
        return {"status": "error", "message": "No file name provided"}, 400

    allowed_extensions = {'.mp4', '.avi'}
    allowed_mime_types = {'video/mp4', 'video/x-msvideo'}
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in allowed_extensions:
        logging.error(f"Invalid file extension: {file_ext}")
        return {"status": "error", "message": f"Invalid file extension. Expected .mp4 or .avi, got {file_ext}"}, 400

    upload_dir = app.config['UPLOAD_FOLDER']
    temp_file_path = os.path.join(upload_dir, f"{file_id}{file_ext}")

    mode = 'ab' if chunk_index > 0 else 'wb'
    with open(temp_file_path, mode) as f:
        file.save(f)

    if chunk_index + 1 == total_chunks:
        mime_type, _ = mimetypes.guess_type(temp_file_path)
        if mime_type not in allowed_mime_types:
            os.unlink(temp_file_path)
            logging.error(f"Invalid file MIME type: {mime_type}")
            return {"status": "error", "message": f"Invalid file format. Expected video/mp4 or video/avi, got {mime_type}"}, 400

        file_size = os.path.getsize(temp_file_path)
        max_size = 1024 * 1024 * 1024  # 1GB
        if file_size > max_size:
            os.unlink(temp_file_path)
            logging.error(f"File size too large: {file_size} bytes")
            return {"status": "error", "message": "File size exceeds 1GB limit"}, 400

        if video_predict_writer is not None:
            video_predict_writer.release()
            video_predict_writer = None
        if video_predict_temp is not None:
            try:
                os.unlink(video_predict_temp.name)
            except:
                pass
            video_predict_temp = None
        video_predict_progress = 0
        video_file_path = temp_file_path
        logging.info(f"Video uploaded successfully: {video_file_path}")
        return {"status": "success", "file_id": file_id}
    else:
        return {"status": "chunk_received", "chunk_index": chunk_index}

@app.route('/start_webcam', methods=['POST'])
@login_required
def start_webcam():
    global camera_on, cap, webcam_index, sequence_buffer, prev_centroids, next_id
    data = request.get_json()
    webcam_index = int(data.get('index', 0))
    logging.info(f"Attempting to open webcam ID {webcam_index}")

    if not camera_on:
        if cap is not None:
            cap.release()
        for attempt in range(3):
            cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera_on = True
                sequence_buffer = []
                prev_centroids = []
                next_id = 1
                logging.info(f"Webcam {webcam_index} opened successfully on attempt {attempt+1}")
                break
            logging.warning(f"Cannot open webcam ID {webcam_index} on attempt {attempt+1}")
            time.sleep(1)
        else:
            logging.error(f"Failed to open webcam {webcam_index} after 3 attempts")
            camera_on = False
    return {"status": camera_on, "message": f"Webcam ID {webcam_index} not available" if not camera_on else ""}

@app.route('/stop_webcam', methods=['POST'])
@login_required
def stop_webcam():
    global camera_on, cap, sequence_buffer, prev_centroids, next_id
    if camera_on:
        if cap is not None:
            cap.release()
            cap = None
        camera_on = False
        sequence_buffer = []
        prev_centroids = []
        next_id = 1
        logging.info("Webcam stopped")
    return {"status": camera_on}

@app.route('/set_frame_size', methods=['POST'])
@login_required
def set_frame_size():
    global frame_width, frame_height
    data = request.get_json()
    frame_width = int(data.get('width', 640))
    frame_height = int(data.get('height', 480))
    logging.info(f"Frame size set to {frame_width}x{frame_height} for {data.get('type', 'camera')}")
    return {"status": "success"}

@app.route('/get_detections')
@login_required
def get_detections():
    return {"detections": detections}

@app.route('/get_video_detections')
@login_required
def get_video_detections():
    return {"detections": video_detections}

@app.route('/save_detections', methods=['POST'])
@login_required
def save_detections():
    global detections
    try:
        df = pd.DataFrame({
            "Time": [d["time"] for d in detections],
            "Person ID": [d["person_id"] for d in detections],
            "Hazardous Behaviors": [d["action"] for d in detections]
        })
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        logging.info(f"Detections prepared for download: {filename}")
        return response
    except Exception as e:
        logging.error(f"Error saving detections: {str(e)}")
        return {"status": "error", "message": str(e)}, 500

@app.route('/clear_detections', methods=['POST'])
@login_required
def clear_detections():
    global detections
    detections.clear()
    logging.info("Detections cleared")
    return {"status": "success"}

@app.route('/predict_video', methods=['POST'])
@login_required
def predict_video():
    global video_file_path
    if not video_file_path or not os.path.exists(video_file_path):
        logging.error("No video uploaded for prediction")
        return {"status": "error", "message": "No video uploaded"}, 400
    return {"status": "success"}

@app.route('/clear_video_detections', methods=['POST'])
@login_required
def clear_video_detections():
    global video_detections, video_predict_writer, video_predict_temp, video_predict_progress, video_file_path, action_chart_data
    video_detections.clear()
    if video_predict_writer is not None:
        video_predict_writer.release()
        video_predict_writer = None
    if video_predict_temp is not None:
        try:
            os.unlink(video_predict_temp.name)
        except:
            pass
        video_predict_temp = None
    if video_file_path is not None:
        try:
            os.unlink(video_file_path)
        except:
            pass
        video_file_path = None
    video_predict_progress = 0
    action_chart_data = []
    logging.info("Video detections cleared")
    return {"status": "success"}

@app.route('/save_video_result', methods=['GET'])
@login_required
def save_video_result():
    global video_predict_temp
    if video_predict_temp is None or not os.path.exists(video_predict_temp.name):
        logging.error("No predicted video available")
        return {"status": "error", "message": "No predicted video available"}, 400
    filename = f"predicted_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    return send_file(video_predict_temp.name, as_attachment=True, download_name=filename)

@app.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    global alert_cooldown
    data = request.get_json()
    sensitivity = data.get('sensitivity')
    time = data.get('time')
    if sensitivity in ['low', 'medium', 'high'] and isinstance(time, int) and time >= 1:
        alert_cooldown = time * 60
        logging.info(f"Settings updated: sensitivity={sensitivity}, cooldown={time} minutes")
        return {"status": "success"}
    return {"status": "error", "message": "Invalid settings"}, 400

if __name__ == '__main__':
    socketio.run(app, debug=True)