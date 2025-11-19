import pickle
from io import BytesIO
import librosa
import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import warnings
import sys
import streamlit as st
import tempfile
import os

# Set ffmpeg path for pydub before importing
ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg_temp', 'ffmpeg-8.0-essentials_build', 'bin', 'ffmpeg.exe')
if os.path.exists(ffmpeg_path):
    os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
    # Set converter path for pydub
    import pydub
    pydub.AudioSegment.converter = ffmpeg_path

from pydub import AudioSegment

# Import Spotify integration
from spotify_integration import get_spotify_recommendations, spotify_recommender

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the CNN model (same architecture as training)
class EmotionCNN(nn.Module):
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
        self.fc1 = nn.Linear(128 * 75, 512)  # Corrected input size after conv layers
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

# Load the model
model = EmotionCNN().to(device)

# Try to load the best model first, then fallback to regular model
try:
    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
    print("Loaded best model from training")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
        print("Loaded final model from training")
    except FileNotFoundError:
        print("No trained model found! Please run Emotion_detection_pytorch.py first")
        sys.exit(1)

model.eval()

# Load scaler and encoder
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Model and preprocessing objects loaded successfully!")

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    # Extract 20 MFCCs
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    if flatten:
        return np.ravel(mfcc_features.T)
    return np.squeeze(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    # Ensure consistent length by padding or truncating
    target_length = int(sr * 2.5)  # 2.5 seconds of audio
    if len(data) > target_length:
        data = data[:target_length]
    else:
        data = np.pad(data, (0, max(0, target_length - len(data))))

    # Extract features
    result = np.array([])

    # Zero crossing rate
    zcr_feat = zcr(data, frame_length, hop_length)
    # RMS energy
    rmse_feat = rmse(data, frame_length, hop_length)
    # MFCC
    mfcc_feat = mfcc(data, sr, frame_length, hop_length)

    # Combine all features
    result = np.hstack((result, zcr_feat, rmse_feat, mfcc_feat))

    # Ensure consistent size by padding if necessary
    if len(result) < 2376:
        result = np.pad(result, (0, 2376 - len(result)))
    elif len(result) > 2376:
        result = result[:2376]

    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=0)  # Add batch dimension: (1, 2376)
    final_result = np.reshape(final_result, (1, 1, 2376))  # Reshape to (batch, channels, features)
    return torch.FloatTensor(final_result).to(device)

emotions1 = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def prediction(path1):
    res = get_predict_feat(path1)
    with torch.no_grad():
        outputs = model(res)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()

    # Convert prediction to emotion label
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    if predicted_class < len(emotion_labels):
        return emotion_labels[predicted_class], probabilities.cpu().numpy()
    else:
        return 'unknown', probabilities.cpu().numpy()

def recommendations(s):
    filename = 'songs.csv'
    emotion = ""
    fields = []
    rows = []
    songs = []
    songs1 = []

    if s == 'happy':
        emotion = 'Happy'
    elif s == 'sad':
        emotion = 'Sad'
    elif s == 'neutral':
        emotion = 'Neutral'
    elif s == 'calm':
        emotion = 'Calm'  # Keep calm mapping for backward compatibility
    elif s == 'angry':
        emotion = 'Angry'
    elif s == 'fear':
        emotion = 'Fear'
    elif s == 'disgust':
        emotion = 'Disgust'
    elif s == 'surprise':
        emotion = 'Surprise'
    else:
        print("No emotion matched")
        return pd.DataFrame()

    final = []
    ind_pos = [1, 2]  # Artist and Song columns

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)

        # Define synonyms / related mood tags for emotions that may not be exact matches in songs.csv
        synonyms = {
            'happy': ['happy', 'cheerful', 'joyous', 'uplifting', 'euphoric'],
            'sad': ['sad', 'melancholy', 'melancholic', 'plaintive', 'poignant'],
            'neutral': ['neutral', 'calm', 'laid-back', 'reserved'],
            'calm': ['calm', 'relaxed', 'mellow', 'soft', 'gentle'],
            'angry': ['angry', 'aggressive', 'outraged', 'hostile', 'fierce'],
            'fear': ['anxious', 'tense', 'ominous', 'menacing', 'atmospheric', 'paranoid', 'suspense', 'nervous'],
            'disgust': ['disgust', 'bitter', 'harsh', 'hostile'],
            'surprise': ['surprise', 'surprising', 'unexpected']
        }

        for i in range(len(rows)):
            if i == 0:  # Skip header
                continue
            # Normalize mood tags from CSV
            words = [w.strip().lower() for w in rows[i][3].split(",")]
            # If exact emotion tag exists in the song's mood tags, add it
            if emotion.lower() in words:
                songs.append(rows[i])
                continue
            # Otherwise check synonyms mapping (if any)
            for syn in synonyms.get(emotion.lower(), []):
                if syn in words:
                    songs.append(rows[i])
                    break

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        for j in range(len(songs)):
            final.append([songs[j][i] for i in ind_pos])

        if final:
            pd_dataframe = pd.DataFrame(final, columns=['Artist', 'Song'])
            return pd_dataframe
        else:
            return pd.DataFrame(columns=['Artist', 'Song'])

def convert_to_wav(input_file):
    try:
        audio = AudioSegment.from_file(input_file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            return temp_wav_file.name
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# Custom CSS for ultra-modern, sleek design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark theme main container */
    .main {
        background: #0a0e27;
        padding: 0;
    }
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1200px;
    }
    
    /* Header section */
    .header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #1e1b4b 100%);
        border-radius: 24px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.1), transparent 50%);
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .logo {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }
    
    .tagline {
        font-size: 1rem;
        color: #a5b4fc;
        font-weight: 400;
        margin-top: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Stats bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1.5rem;
        background: rgba(99, 102, 241, 0.05);
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6366f1;
        display: block;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }
    
    /* Upload section */
    .upload-container {
        background: #1e293b;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #6366f1;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.1);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Emotion result card */
    .result-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 24px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .result-content {
        position: relative;
        z-index: 1;
    }
    
    .result-label {
        font-size: 0.875rem;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.75rem;
    }
    
    .result-emotion {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    /* Confidence bars */
    .confidence-container {
        background: #1e293b;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #334155;
    }
    
    .confidence-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
    }
    
    .confidence-item {
        margin-bottom: 1rem;
    }
    
    .confidence-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .confidence-label {
        font-size: 0.875rem;
        color: #cbd5e1;
        text-transform: capitalize;
        font-weight: 500;
    }
    
    .confidence-value {
        font-size: 0.875rem;
        color: #6366f1;
        font-weight: 600;
    }
    
    .confidence-bar-bg {
        background: #0f172a;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    
    /* Music recommendations */
    .music-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .music-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    
    .music-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.5rem 0;
    }
    
    .music-artist {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    
    .music-album {
        font-size: 0.75rem;
        color: #64748b;
    }
    
    /* Info alert */
    .info-alert {
        background: rgba(251, 191, 36, 0.1);
        border-left: 3px solid #fbbf24;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #fbbf24;
        font-size: 0.875rem;
        margin: 1.5rem 0;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 12px;
        border: 1px solid #334155;
        color: #f1f5f9 !important;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: #0f172a;
        border: 1px solid #334155;
        border-top: none;
        border-radius: 0 0 12px 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <div class="header-content">
        <h1 class="logo">VOICEMOOD</h1>
        <p class="tagline">AI-Powered Emotion Recognition & Music Discovery</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-value">97.09%</span>
        <span class="stat-label">Accuracy</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">7</span>
        <span class="stat-label">Emotions</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">GPU</span>
        <span class="stat-label">Accelerated</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">PyTorch</span>
        <span class="stat-label">CNN Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Info alert
st.markdown("""
<div class="info-alert">
    <strong>Best Results:</strong> Model trained on professional acted speech. Use clear recordings with exaggerated emotions for accurate predictions.
</div>
""", unsafe_allow_html=True)

st.divider()

# Upload section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Audio File</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Supported formats: WAV, MP3, FLAC, OGG</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"], label_visibility="collapsed")
if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Convert to WAV if necessary
    if not uploaded_file.name.lower().endswith('.wav'):
        converted_path = convert_to_wav(tmp_path)
        if converted_path:
            tmp_path = converted_path
        else:
            st.error("Failed to convert audio file to WAV format")
            os.unlink(tmp_path)
            st.stop()

    # Display the audio
    st.audio(uploaded_file, format='audio/wav')
    
    # Processing indicator
    with st.spinner('Analyzing emotion...'):
        # Make prediction using the temporary file path
        s, confidence_scores = prediction(tmp_path)
    
    # Display result with modern styling
    st.markdown(f"""
    <div class="result-card">
        <div class="result-content">
            <div class="result-label">Detected Emotion</div>
            <h1 class="result-emotion">{s.upper()}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence scores
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
    st.markdown('<div class="confidence-title">Confidence Scores</div>', unsafe_allow_html=True)
    
    # Sort by confidence (highest first)
    sorted_indices = confidence_scores.argsort()[::-1]
    
    for idx in sorted_indices:
        confidence_percent = confidence_scores[idx] * 100
        st.markdown(f"""
        <div class="confidence-item">
            <div class="confidence-header">
                <span class="confidence-label">{emotion_labels[idx]}</span>
                <span class="confidence-value">{confidence_percent:.2f}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width: {confidence_percent}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close upload-container div

    # Create two columns for recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Music Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Personalized selections based on your detected emotion</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title" style="font-size: 1rem; margin-top: 1rem;">Local Library</div>', unsafe_allow_html=True)
        dataframe = recommendations(s)
        if not dataframe.empty:
            st.dataframe(dataframe, use_container_width=True, hide_index=True)
            st.caption(f"Found {len(dataframe)} tracks")
        else:
            st.info("No local matches found")

    with col2:
        st.markdown('<div class="section-title" style="font-size: 1rem; margin-top: 1rem;">Spotify</div>', unsafe_allow_html=True)
        try:
            spotify_recs = get_spotify_recommendations(s, 5)
            if spotify_recs:
                for i, rec in enumerate(spotify_recs, 1):
                    st.markdown(f"""
                    <div class="music-card">
                        <div class="music-title">{rec['name']}</div>
                        <div class="music-artist">{rec['artist']}</div>
                        <div class="music-album">{rec.get('album', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Spotify link
                    if rec.get('spotify_url'):
                        st.link_button("Listen on Spotify", rec['spotify_url'], use_container_width=True, type="primary")

                    # Embed player (optional)
                    if st.checkbox(f"Show player", key=f"embed_{i}"):
                        if rec.get('embed_url'):
                            st.components.v1.html(rec['embed_url'], height=152)

            else:
                st.info("Spotify recommendations unavailable")
                with st.expander("Setup Instructions"):
                    st.markdown("""
                    **Enable Spotify Integration:**
                    1. Visit [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
                    2. Create app and obtain credentials
                    3. Set environment variables:
                       - SPOTIFY_CLIENT_ID
                       - SPOTIFY_CLIENT_SECRET
                    """)
        except Exception as e:
            st.error(f"Spotify error: {str(e)}")
            st.info("Local recommendations available in left column")

    # Clean up the temporary file
    import os
    os.unlink(tmp_path)

# Footer
if uploaded_file is None:
    st.markdown("</div>", unsafe_allow_html=True)  # Close upload-container div
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 2rem; font-size: 0.875rem;">
        <p style="margin: 0;">Powered by PyTorch • CUDA • Spotify API</p>
    </div>
    """, unsafe_allow_html=True)