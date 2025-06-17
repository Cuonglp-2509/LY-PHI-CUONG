import cv2
import torch
import numpy as np
from torch_geometric.nn import GCNConv
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
from flask_uploads import UploadSet, configure_uploads, ALL
import io
import tempfile
import random
import boto3
import ffmpeg
import bcrypt
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thêm đường dẫn dự án
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(ROOT)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER.setLevel(logging.ERROR)

# Khởi tạo Flask và extensions
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', str(uuid4()))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOADED_VIDEOS_DEST'] = os.path.join(CURR_PATH, 'uploads')
app.config['UPLOADED_VIDEOS_ALLOW'] = ['mp4', 'avi']
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

# Cấu hình Flask-Uploads
videos = UploadSet('videos', ALL)
configure_uploads(app, videos)

# Cấu hình AWS S3
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Cấu hình Twilio
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
        'view_video_button': 'View Predicted Video',
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
        'view_video_button': '예측 비디오 보기',
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

# Model User cho database
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
    return db.session.get(User, int(user_id))

# Biến toàn cục
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
sequence_length = 20
num_keypoints = 17
next_id = 1
prev_centroids = []

# Cấu hình model
MODEL_PATH = "models/gcn_lstm_model_0531.pth"
LABEL_MAP_PATH = "data/label_mapping.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8l.pt", task='detect').to(DEVICE)
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(DEVICE)
pose_model.eval()

class GCN_LSTM_Model(torch.nn.Module):
    def __init__(self, sequence_length=20, num_keypoints=17, num_classes=12):
        super(GCN_LSTM_Model, self).__init__()
        self.gcn1 = GCNConv(3, 64)
        self.gcn2 = GCNConv(64, 128)
        self.lstm = torch.nn.LSTM(128 * num_keypoints, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x, edge_index):
        batch_size, seq_len, features = x.size()
        x = x.view(batch_size, seq_len, num_keypoints, 3)
        gcn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :]
            frame = self.gcn1(frame, edge_index).relu()
            frame = self.gcn2(frame, edge_index).relu()
            gcn_features.append(frame.view(batch_size, -1))
        gcn_features = torch.stack(gcn_features, dim=1)
        lstm_out, _ = self.lstm(gcn_features)
        out = lstm_out[:, -1, :]
        out = self.fc1(out).relu()
        out = self.fc2(out)
        return out

# Load GCN-LSTM model và label mapping
model = GCN_LSTM_Model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

ACTION_NAME_MAP = {
    (0, 4): "Walking",
    (0, 21): "Transition state",
    (0, 28): "Lay down",
    (0, 31): "Using the tool",
    (0, 33): "Enter working area",
    (0, 34): "Leave working area",
    (0, 35): "Carry object",
    (0, 36): "Work in progress",
    (0, 37): "Falling down",
    (22, 31): "Pick up object",
    (23, 31): "Put down object",
    (24, 31): "Setup machine"
}
action_to_name = {action_label: ACTION_NAME_MAP.get(pair, "Unknown") for pair, action_label in label_map.items()}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]
adj_matrix = np.zeros((num_keypoints, num_keypoints))
for edge in SKELETON_CONNECTIONS:
    i, j = edge
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1
edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long).to(DEVICE)

def normalize_keypoints(keypoints, bbox):
    x_min, y_min, width, height = bbox
    normalized = []
    for x, y, c in keypoints:
        x_norm = (x - x_min) / width if width > 1 else x / frame_width
        y_norm = (y - y_min) / height if height > 1 else y / frame_height
        normalized.append([x_norm, y_norm, c])
    return np.array(normalized)

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
    for start_idx, end_idx in connections:
        if keypoints[start_idx, 2] > 0.1 and keypoints[end_idx, 2] > 0.1:
            p1 = tuple(map(int, keypoints[start_idx, :2]))
            p2 = tuple(map(int, keypoints[end_idx, :2]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for x, y, c in keypoints:
        if c > 0.1:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    if bbox is not None:
        x_min, y_min, w, h = bbox
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min + w), int(y_min + h)), (255, 0, 0), 2)
        label = f"ID: {person_id}, Action: {action_label}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (int(x_min), int(y_min - 30)), (int(x_min + text_size[0]), int(y_min)), (0, 0, 0), -1)
        cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    logo_path = os.path.join(CURR_PATH, "static/Imgs/CNU_logo.png")
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path)
        logo = cv2.resize(logo, (100, 100))
        h, w = logo.shape[:2]
        frame[0:h, 0:w] = logo
    company_logo_path = os.path.join(CURR_PATH, "static/Imgs/safemotion_logo.png")
    if os.path.exists(company_logo_path):
        company_logo = cv2.imread(company_logo_path)
        company_logo = cv2.resize(company_logo, (100, 100))
        h, w = company_logo.shape[:2]
        frame[0:h, w+10:w*2+10] = company_logo
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
        num_incidents = random.randint(0, 5)
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
                "full_time": timestamp.strftime("%Y-%m-d %H:%M:%S")
            }
            simulated_detections.append(detection)
    return simulated_detections

detections = generate_simulated_detections()

def convert_to_h264(input_path, output_path):
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='h264', acodec='aac', format='mp4')
        ffmpeg.run(stream)
        logging.info(f"Video converted to H.264: {output_path}")
    except Exception as e:
        logging.error(f"Error converting video: {str(e)}")

def generate_frames():
    global cap, camera_on, frame_width, frame_height, detections, last_time, sequence_buffer, prev_centroids, next_id
    ith_img = -1
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
                font, font_scale, (255, 255, 255), 2, cv2.LINE_AA
            )
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            logging.warning("Cannot read frame from webcam")
            continue

        frame = cv2.resize(frame, (frame_width, frame_height))
        ith_img += 1
        img_disp = frame.copy()

        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time != last_time else 0
        last_time = current_time
        cv2.putText(img_disp, f"FPS: {fps:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            rgb = img_disp[:, :, ::-1]
            results = yolo_model(rgb)[0]
            person_boxes_xyxy = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if int(cls.item()) == 0:
                    person_boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])

            if not person_boxes_xyxy:
                ret, buffer = cv2.imencode('.jpg', img_disp)
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
                if len(sequence_buffer) == sequence_length:
                    input_data = np.array(sequence_buffer).reshape(1, sequence_length, num_keypoints * 3)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        prediction = model(input_tensor, edge_index)
                        predicted_class = torch.argmax(prediction, dim=1).item()
                        action_xx = sorted(list(set(label_map.values())))[predicted_class]
                        action_label = action_to_name.get(action_xx, "Unknown")
                    sequence_buffer.pop(0)

                img_disp = draw_skeleton(img_disp, keypoints_with_conf, SKELETON_CONNECTIONS, bbox, person_id, action_label)

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

        ret, buffer = cv2.imencode('.jpg', img_disp)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_video_predict_frames():
    global video_file_path, video_detections, video_predict_writer, video_predict_temp, video_predict_progress, frame_width, frame_height, prev_centroids, next_id
    if not video_file_path:
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

    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_video.get(cv2.CAP_PROP_FPS) or 30

    if video_predict_temp is None:
        video_predict_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_predict_writer = cv2.VideoWriter(video_predict_temp.name, fourcc, fps, (frame_width, frame_height))

    ith_img = -1
    video_predict_progress = 0
    video_sequence_buffer = []
    prev_centroids = []
    next_id = 1

    while cap_video.isOpened():
        success, frame = cap_video.read()
        if not success:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        ith_img += 1
        img_disp = frame.copy()

        video_predict_progress = (ith_img / total_frames) * 100

        try:
            rgb = img_disp[:, :, ::-1]
            results = yolo_model(rgb)[0]
            person_boxes_xyxy = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if int(cls.item()) == 0:
                    person_boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])

            if not person_boxes_xyxy:
                if video_predict_writer is not None:
                    video_predict_writer.write(img_disp)
                ret, buffer = cv2.imencode('.jpg', img_disp)
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
                if len(video_sequence_buffer) == sequence_length:
                    input_data = np.array(video_sequence_buffer).reshape(1, sequence_length, num_keypoints * 3)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        prediction = model(input_tensor, edge_index)
                        predicted_class = torch.argmax(prediction, dim=1).item()
                        action_xx = sorted(list(set(label_map.values())))[predicted_class]
                        action_label = action_to_name.get(action_xx, "Unknown")
                    video_sequence_buffer.pop(0)

                img_disp = draw_skeleton(img_disp, keypoints_with_conf, SKELETON_CONNECTIONS, bbox, person_id, action_label)

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
                        sms_message = LANGUAGES[language]['hazard'].format(
                            action=action_label, id=person_id, time=f"{current_time:.2f}s"
                        )

            if len(video_detections) > 10:
                video_detections.pop(0)

            if video_predict_writer is not None:
                video_predict_writer.write(img_disp)

        except Exception as e:
            logging.error(f"Error in video action recognition: {str(e)}")

        ret, buffer = cv2.imencode('.jpg', img_disp)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap_video.release()
    if video_predict_writer is not None:
        video_predict_writer.release()
        video_predict_writer = None

    # Convert video to H.264
    converted_path = os.path.join(CURR_PATH, 'Uploads_video', f"converted_{os.path.basename(video_predict_temp.name)}")
    convert_to_h264(video_predict_temp.name, converted_path)
    os.unlink(video_predict_temp.name)
    video_predict_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    os.rename(converted_path, video_predict_temp.name)
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
    return render_template('index.html', detections=detections, video_detections=video_detections, translations=translations, ADMIN_PHONE_NUMBER=ADMIN_PHONE_NUMBER)

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

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    global video_file_path, video_predict_writer, video_predict_temp, video_predict_progress
    if 'file' not in request.files:
        logging.error("No video file uploaded")
        return {"status": "error", "message": "No video file uploaded"}, 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No video file selected")
        return {"status": "error", "message": "No video file selected"}, 400

    if videos_allowed(file.filename):
        try:
            filename = videos.save(file)
            local_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
            s3_key = f"videos/{filename}"
            s3_client.upload_file(local_path, AWS_S3_BUCKET, s3_key)
            os.unlink(local_path)  # Xóa file cục bộ
            video_file_path = local_path

            if video_predict_writer is not None:
                video_predict_writer.release()
                video_predict_writer = None
            if video_predict_temp is not None:
                try:
                    os.unlink(video_predict_temp.name)
                except OSError:
                    pass
                video_predict_temp = None
            video_predict_progress = 0

            logging.info(f"Video uploaded to S3: s3://{AWS_S3_BUCKET}/{s3_key}")
            return {"status": "success", "s3_key": s3_key}
        except Exception as e:
            logging.error(f"Error uploading video: {str(e)}")
            return {"status": "error", "message": str(e)}, 500
    else:
        logging.error(f"Invalid file type: {file.filename}")
        return {"status": "error", "message": "Only .mp4 and .avi files are allowed"}, 400

@app.route('/view_video_result', methods=['GET'])
@login_required
def view_video_result():
    global video_predict_temp
    if video_predict_temp is None or not os.path.exists(video_predict_temp.name):
        logging.error("No predicted video available")
        return {"status": "error", "message": "No predicted video available"}, 400

    s3_key = f"predicted_videos/{os.path.basename(video_predict_temp.name)}"
    try:
        s3_client.upload_file(video_predict_temp.name, AWS_S3_BUCKET, s3_key)
        video_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600
        )
        logging.info(f"Generated presigned URL for video: {s3_key}")
        return {"status": "success", "video_url": video_url}
    except Exception as e:
        logging.error(f"Error generating video URL: {str(e)}")
        return {"status": "error", "message": str(e)}, 500

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
            time.sleep(0.5)
        else:
            logging.error(f"Failed to open webcam ID {webcam_index} after 3 attempts")
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
    if not video_file_path:
        logging.error("No video uploaded for prediction")
        return {"status": "error", "message": "No video uploaded"}, 400
    return {"status": "success"}

@app.route('/clear_video_detections', methods=['POST'])
@login_required
def clear_video_detections():
    global video_detections, video_predict_writer, video_predict_temp, video_predict_progress, video_file_path
    video_detections.clear()
    if video_predict_writer is not None:
        video_predict_writer.release()
        video_predict_writer = None
    if video_predict_temp is not None:
        try:
            os.unlink(video_predict_temp.name)
        except OSError:
            pass
        video_predict_temp = None
    if video_file_path:
        try:
            os.unlink(video_file_path)
        except OSError:
            pass
        video_file_path = None
    video_predict_progress = 0
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