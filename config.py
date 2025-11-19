"""
Configuration file for VoiceMood AI
"""

# Model Configuration
MODEL_PATH = 'best_emotion_model.pth'
SCALER_PATH = 'scaler2.pickle'
ENCODER_PATH = 'encoder2.pickle'
SONGS_CSV_PATH = 'songs.csv'

# Audio Configuration
SAMPLE_RATE = 22050
DURATION = 2.5
OFFSET = 0.6
FEATURE_SIZE = 2376

# Emotion Labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': '#ef4444',
    'disgust': '#84cc16',
    'fear': '#8b5cf6',
    'happy': '#f59e0b',
    'neutral': '#6b7280',
    'sad': '#3b82f6',
    'surprise': '#ec4899'
}

# UI Configuration
APP_TITLE = "VOICEMOOD"
APP_TAGLINE = "AI-Powered Emotion Recognition & Music Discovery"
MODEL_ACCURACY = "97.09%"
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg"]

# Spotify Configuration
SPOTIFY_RECOMMENDATION_LIMIT = 5

# Theme Colors
THEME = {
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'background': '#0a0e27',
    'surface': '#1e293b',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8'
}
