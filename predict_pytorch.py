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
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()

    # Convert prediction to emotion label
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    if predicted_class < len(emotion_labels):
        return emotion_labels[predicted_class]
    else:
        return 'unknown'

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

st.title("🎵 VoiceMood AI - Emotion-Based Music Discovery")
st.markdown("""
**Discover music through your voice with AI-powered emotion detection!**

🎭 **Advanced Features:**
- Real-time emotion detection from voice (7 emotions)
- 97.09% accuracy with PyTorch CNN
- GPU acceleration for lightning-fast processing
- Dual recommendation system: Local + Spotify
- Privacy-first: All processing happens locally

🎵 **How it works:**
1. Upload any voice recording (WAV/MP3/FLAC/OGG)
2. AI analyzes your emotional state
3. Get personalized music recommendations from both local database and Spotify
4. Listen instantly with Spotify integration!

⚠️ **Important Note:** This model is trained on acted emotional speech datasets. For best results, use clear voice recordings with exaggerated emotional expressions, similar to acting performances. Real conversations or music may produce unexpected results.
""")

st.divider()

uploaded_file = st.file_uploader("Upload Your Audio File", type=["wav", "mp3", "flac", "ogg"])
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

    # Make prediction using the temporary file path
    s = prediction(tmp_path)
    st.write(f"Your mood is: **{s.capitalize()}**")

    # Create two columns for recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.write("🎵 **Local Music Recommendations:**")
        dataframe = recommendations(s)
        if not dataframe.empty:
            st.dataframe(dataframe)
        else:
            st.write("😔 No songs found for this emotion in our database")

    with col2:
        st.write("🎵 **Spotify Recommendations:**")
        try:
            spotify_recs = get_spotify_recommendations(s, 5)
            if spotify_recs:
                for i, rec in enumerate(spotify_recs, 1):
                    with st.container():
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            if rec.get('image_url'):
                                st.image(rec['image_url'], width=60)
                            else:
                                st.write("🎵")
                        with col_b:
                            st.write(f"**{i}. {rec['name']}**")
                            st.write(f"*{rec['artist']}*")
                            st.write(f"Album: {rec.get('album', 'N/A')}")

                        # Spotify link
                        if rec.get('spotify_url'):
                            st.markdown(f"[🔗 Listen on Spotify]({rec['spotify_url']})")

                        # Embed player (optional - can be toggled)
                        if st.checkbox(f"Show player for {rec['name']}", key=f"embed_{i}"):
                            if rec.get('embed_url'):
                                st.components.v1.html(rec['embed_url'], height=152)

                        st.divider()
            else:
                st.write("🎵 Spotify recommendations not available")
                st.info("💡 **To enable Spotify recommendations:**\n"
                       "1. Go to https://developer.spotify.com/dashboard\n"
                       "2. Create an app and get Client ID & Secret\n"
                       "3. Set environment variables:\n"
                       "   - SPOTIFY_CLIENT_ID\n"
                       "   - SPOTIFY_CLIENT_SECRET")
        except Exception as e:
            st.error(f"Spotify integration error: {e}")
            st.info("Local recommendations are still available above! 🎵")

    # Clean up the temporary file
    import os
    os.unlink(tmp_path)