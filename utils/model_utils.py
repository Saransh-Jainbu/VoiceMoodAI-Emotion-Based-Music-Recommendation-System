"""
Model handling utilities for VoiceMood AI
"""
import pickle
import torch
import torch.nn as nn
import librosa
import numpy as np
from config import *


class EmotionCNN(nn.Module):
    """CNN model for emotion detection"""
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 75, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout2(x)
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = torch.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model():
    """Load the trained emotion detection model and preprocessing objects"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        model.eval()
        print(f"✓ Model loaded on {device}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {MODEL_PATH}")
        return None, None, None
    
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        print("✓ Preprocessing objects loaded")
    except FileNotFoundError as e:
        print(f"✗ Preprocessing file not found: {e}")
        return None, None, None
    
    return model, scaler, device


def zcr(data, frame_length, hop_length):
    """Zero Crossing Rate feature extraction"""
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))


def rmse(data, frame_length=2048, hop_length=512):
    """Root Mean Square Energy feature extraction"""
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    """MFCC feature extraction"""
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    return np.ravel(mfcc_features.T) if flatten else np.squeeze(mfcc_features.T)


def extract_features(data, sr=SAMPLE_RATE, frame_length=2048, hop_length=512):
    """Extract all features from audio data"""
    target_length = int(sr * DURATION)
    data = data[:target_length] if len(data) > target_length else np.pad(data, (0, max(0, target_length - len(data))))
    
    result = np.hstack((
        np.array([]),
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    
    if len(result) < FEATURE_SIZE:
        result = np.pad(result, (0, FEATURE_SIZE - len(result)))
    elif len(result) > FEATURE_SIZE:
        result = result[:FEATURE_SIZE]
    
    return result


def get_predict_feat(path, scaler, device):
    """Prepare features for prediction"""
    d, s_rate = librosa.load(path, duration=DURATION, offset=OFFSET)
    res = extract_features(d, sr=s_rate)
    result = np.reshape(np.array(res), newshape=(1, FEATURE_SIZE))
    i_result = scaler.transform(result)
    final_result = np.reshape(np.expand_dims(i_result, axis=0), (1, 1, FEATURE_SIZE))
    return torch.FloatTensor(final_result).to(device)


def predict_emotion(audio_path, model, scaler, device):
    """
    Predict emotion from audio file
    
    Returns:
        emotion (str): Predicted emotion label
        confidence_scores (np.array): Confidence scores for all emotions
    """
    try:
        features = get_predict_feat(audio_path, scaler, device)
        
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = predicted.item()
        
        emotion = EMOTION_LABELS[predicted_class] if predicted_class < len(EMOTION_LABELS) else 'unknown'
        confidence_scores = probabilities.cpu().numpy()
        
        return emotion, confidence_scores
    except Exception as e:
        print(f"Prediction error: {e}")
        return 'error', None
