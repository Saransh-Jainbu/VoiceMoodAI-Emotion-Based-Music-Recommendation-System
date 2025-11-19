# 🎵 VoiceMood AI - Emotion-Based Music Recommendation System

<div align="center">

![VoiceMood AI](https://img.shields.io/badge/VoiceMood-AI-blue?style=for-the-badge&logo=microphone&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=for-the-badge&logo=nvidia)
![Accuracy](https://img.shields.io/badge/Accuracy-97.09%25-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web--App-FF4B4B?style=for-the-badge&logo=streamlit)

**Discover music through your voice with AI-powered emotion detection**


</div>

---

## 🌟 Overview

VoiceMood AI is a revolutionary music recommendation system that uses advanced artificial intelligence to detect emotions from voice recordings and recommend personalized music playlists. Built with PyTorch and accelerated by NVIDIA GPU technology, it achieves **97.09% accuracy** in real-time emotion classification.

### 🎯 Key Highlights
- **🎭 7 Emotion Classes**: Detects neutral, calm, happy, sad, angry, fear, and disgust
- **🚀 GPU Acceleration**: 10-50x faster processing with RTX 3060
- **🧠 Advanced AI**: 7.1M parameter CNN trained on 48,532+ audio samples
- **🔒 Privacy-First**: All processing happens locally on your device
- **🎵 Smart Recommendations**: Curated music database with mood-based matching
- **🌐 Web Interface**: Beautiful Streamlit app for easy interaction

---

## ✨ Features

### 🎭 Emotion Detection
- **Real-time Analysis**: Process audio files in seconds
- **Multi-format Support**: WAV, MP3, FLAC, OGG files
- **Advanced Features**: MFCC, ZCR, RMSE extraction with data augmentation
- **Confidence Scoring**: Detailed emotion probability distributions

### 🎵 Music Recommendations
- **Mood-Based Matching**: Intelligent playlist generation
- **Curated Database**: 1000+ songs across all emotion categories
- **Artist & Song Info**: Complete metadata for each recommendation
- **Dynamic Playlists**: Context-aware music suggestions
- **Spotify Integration**: Direct links to Spotify with personalized recommendations

### 🚀 Performance
- **GPU Acceleration**: CUDA 12.1 with PyTorch optimization
- **High Accuracy**: 97.09% validation accuracy
- **Fast Inference**: Sub-second emotion detection
- **Scalable Architecture**: Batch processing capabilities

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│ Feature Extract │───▶│   CNN Model     │
│  (WAV/MP3/...)  │    │   MFCC+ZCR+RMSE │    │ 7.1M Parameters │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             ▼
│Preprocessing    │───▶│   Prediction    │◀────────────┘
│ StandardScaler  │    │   7 Emotions    │
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   Music DB      │    │ Recommendations │
│   1000+ Songs   │    │   Playlists     │
└─────────────────┘    └─────────────────┘
```

### 🧠 Model Architecture
- **Input**: 2376-dimensional audio features
- **Layers**: 5 Conv1D + BatchNorm + Dropout + Dense
- **Output**: 7 emotion classes with softmax probabilities
- **Training**: Adam optimizer with early stopping
- **Regularization**: Dropout (0.2) and Batch Normalization

---

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 97.09% | Best validation accuracy |
| **Training Samples** | 48,532 | Processed audio samples |
| **Model Parameters** | 7.1M | Neural network size |
| **GPU Speed Boost** | 10-50x | CUDA acceleration |
| **Inference Time** | <1s | Real-time processing |
| **Emotion Classes** | 7 | Comprehensive coverage |

---

## ⚠️ Model Limitations & Best Practices

### 🎭 Training Data Characteristics
The emotion detection model is trained exclusively on **acted emotional speech datasets** (RAVDESS, CREMA-D, TESS, SAVEE). These datasets contain:
- Professional actors performing scripted emotional expressions
- Controlled recording environments
- Clear, exaggerated emotional cues
- Standardized 2-3 second audio clips

### 🎯 When Predictions Work Best
- **Acted performances**: Audio from movies, theater, or deliberate emotional expressions
- **Clear speech**: Well-articulated vocal recordings
- **Similar duration**: 2-5 second clips (model processes 2.5 seconds starting at 0.6s offset)
- **Speech content**: Vocal expressions rather than music, sound effects, or noise

### � When Predictions May Be Inaccurate
- **Real conversations**: Natural, spontaneous speech may not match acted patterns
- **Music or singing**: Model expects speech features, not musical elements
- **Background noise**: Environmental sounds can interfere with feature extraction
- **Non-speech audio**: Whispers, cries, or non-verbal sounds
- **Different accents/languages**: Model trained primarily on English speech
- **Short/long audio**: Very brief (<1s) or very long (>10s) recordings

### 💡 Recommendations for Best Results
1. **Use clear voice recordings** with minimal background noise
2. **Speak with exaggerated emotion** similar to acting performances
3. **Keep recordings 2-5 seconds** for optimal processing
4. **Try different emotional expressions** if results seem off
5. **Consider the context**: Model detects acted emotions, not clinical psychological states

### 🔄 Emotion-to-Mood Mapping
The music recommendation system uses intelligent mood mapping since song metadata often uses descriptive terms rather than exact emotion labels:
- **Happy** → happy, cheerful, joyous, uplifting, euphoric
- **Sad** → sad, melancholy, melancholic, plaintive, poignant
- **Neutral** → neutral, calm, laid-back, reserved
- **Angry** → angry, aggressive, outraged, hostile, fierce
- **Fear** → anxious, tense, ominous, menacing, atmospheric, paranoid, suspense, nervous
- **Disgust** → disgust, bitter, harsh, hostile
- **Surprise** → surprise, surprising, unexpected

---

### Prerequisites
- Python 3.11+
- NVIDIA GPU (RTX 30-series recommended)
- CUDA 12.1 compatible drivers
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HitenDhamija/ai_project.git
   cd ai_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install librosa soundfile pydub pandas numpy scikit-learn streamlit
   ```

4. **Download datasets** (see [Datasets](#datasets) section)

5. **Set up Spotify integration** (optional but recommended)
   ```bash
   python spotify_setup.py
   ```

6. **Run data preprocessing**
   ```bash
   python Data_Preprocessing.py
   ```

7. **Train the model** (optional - pretrained model included)
   ```bash
   python Emotion_detection_pytorch.py
   ```

8. **Launch the web app**
   ```bash
   streamlit run predict_pytorch.py
   ```

### 🎯 Usage

1. **Open the app**: Navigate to `http://localhost:8501`
2. **Upload audio**: Choose a voice recording (WAV/MP3/FLAC/OGG)
3. **Get emotion**: AI analyzes your voice in real-time
4. **Enjoy music**: Receive personalized recommendations from both local database and Spotify
5. **Listen instantly**: Click Spotify links to listen immediately on Spotify

---

## 🎵 Spotify Integration

VoiceMood AI includes powerful Spotify integration for enhanced music discovery:

### Features
- **Personalized Recommendations**: Spotify's algorithm + emotion mapping
- **Audio Feature Matching**: Maps emotions to valence, energy, danceability, etc.
- **Direct Streaming**: Click links to listen instantly on Spotify
- **Embed Players**: Optional in-app Spotify players
- **Fallback System**: Works even without Spotify API access
- **Playlist Creation**: Create Spotify playlists from emotion recommendations (requires OAuth)
- **User Authentication**: Access personalized Spotify features

### Setup
1. **Get Spotify Credentials**:
   ```bash
   python spotify_setup.py
   ```
   This will guide you through getting Client ID and Client Secret from Spotify Developer Dashboard.

2. **Set Redirect URI** (for advanced features):
   - Go to your [Spotify App Dashboard](https://developer.spotify.com/dashboard)
   - Edit your app settings
   - Add `http://127.0.0.1:8501` to "Redirect URIs" (use explicit IP, not localhost)
   - **Note**: Spotify requires explicit IPv4/IPv6 addresses for loopback redirects
   - This enables playlist creation and user-specific features

3. **Environment Variables** (alternative):
   ```bash
   export SPOTIFY_CLIENT_ID="your_client_id"
   export SPOTIFY_CLIENT_SECRET="your_client_secret"
   export SPOTIFY_REDIRECT_URI="http://127.0.0.1:8501"  # Spotify-compliant loopback address
   ```

### Emotion-to-Music Mapping
- **Happy** → Upbeat pop, dance, party music
- **Sad** → Acoustic, indie, folk ballads
- **Angry** → Rock, metal, high-energy tracks
- **Calm** → Ambient, classical, lo-fi beats
- **Fear** → Atmospheric, soundtrack music
- **Surprise** → Electronic, upbeat discoveries

### Advanced Features (OAuth)
For playlist creation and user-specific features, use OAuth authentication:

```python
from spotify_integration import SpotifyRecommender

# Initialize with OAuth
recommender = SpotifyRecommender(use_oauth=True)

# Get authorization URL
auth_url = recommender.get_auth_url()
print(f"Visit: {auth_url}")

# After user authorizes, use the code from redirect URL
# recommender.set_oauth_token(authorization_code)

# Create emotion-based playlist
track_ids = ["spotify_track_id_1", "spotify_track_id_2"]
playlist_url = recommender.create_emotion_playlist("happy", track_ids)
```

---

## 📁 Project Structure

```
ai_project/
├── 📊 Data_Preprocessing.py      # Audio feature extraction pipeline
├── 🧠 Emotion_detection_pytorch.py # PyTorch CNN training script
├── 🌐 predict_pytorch.py         # Streamlit web application
├── 🎨 landing_page.html          # Professional landing page
├── 📄 songs.csv                  # Music database with moods
├── 🔧 scaler2.pickle            # Feature preprocessing scaler
├── 🏷️ encoder2.pickle           # Emotion label encoder
├── 🤖 best_emotion_model.pth     # Trained PyTorch model (97.09% acc)
├── 📋 README.md                  # Project documentation
├── 🎵 audio_speech_actors_01-24/ # RAVDESS dataset
├── 🎵 AudioWAV/                  # CREMA-D dataset
├── 🎵 TESS Toronto emotional.../ # TESS dataset
└── 🎵 ALL/                       # SAVEE dataset
```

---

## 📊 Datasets

### 🎭 Toronto Emotional Speech Set (TESS)
- **Source**: [Kaggle TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Content**: 2,800 audio files from 2 actresses
- **Emotions**: anger, disgust, fear, happiness, neutral, sadness, surprise
- **Format**: WAV files, ~3 seconds each

### 🎭 RAVDESS Emotional Speech Audio
- **Source**: [Kaggle RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Content**: 1,440 audio files from 24 actors
- **Emotions**: calm, happy, sad, angry, fearful, surprise, disgust
- **Format**: WAV files, professional recordings

### 🎭 CREMA-D Audio
- **Source**: [Kaggle CREMA-D Dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
- **Content**: 7,442 audio files from 91 actors
- **Emotions**: anger, disgust, fear, happy, neutral, sad
- **Format**: WAV files, diverse speaker demographics

### 🎭 Surrey Audio-Visual Expressed Emotion (SAVEE)
- **Source**: [Kaggle SAVEE Dataset](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- **Content**: 480 audio files from 4 male actors
- **Emotions**: anger, disgust, fear, happiness, sadness, surprise
- **Format**: WAV files, British English accents

---

## 🧠 Model Details

### Architecture Specifications
```python
EmotionCNN(
  (conv1): Conv1d(1, 512, kernel_size=5, stride=1, padding=2)
  (bn1): BatchNorm1d(512)
  (pool1): MaxPool1d(kernel_size=5, stride=2, padding=2)
  (conv2): Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
  (bn2): BatchNorm1d(512)
  (pool2): MaxPool1d(kernel_size=5, stride=2, padding=2)
  (dropout1): Dropout(p=0.2)
  (conv3): Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
  (bn3): BatchNorm1d(256)
  (pool3): MaxPool1d(kernel_size=5, stride=2, padding=2)
  (conv4): Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
  (bn4): BatchNorm1d(256)
  (pool4): MaxPool1d(kernel_size=5, stride=2, padding=2)
  (dropout2): Dropout(p=0.2)
  (conv5): Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
  (bn5): BatchNorm1d(128)
  (pool5): MaxPool1d(kernel_size=3, stride=2, padding=1)
  (dropout3): Dropout(p=0.2)
  (flatten): Flatten()
  (fc1): Linear(in_features=9600, out_features=512, bias=True)
  (bn6): BatchNorm1d(512)
  (fc2): Linear(in_features=512, out_features=7, bias=True)
)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: Up to 50 with early stopping
- **Data Split**: 80% train, 20% validation
- **Augmentation**: Noise, pitch shift, time stretch


## 🔧 Development

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Training Custom Model
```bash
# Preprocess data
python Data_Preprocessing.py

# Train model
python Emotion_detection_pytorch.py

# Test predictions
python predict_pytorch.py
```

### Web App Development
```bash
# Run in development mode
streamlit run predict_pytorch.py --server.headless true

# Access at http://localhost:8501
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Write tests for new features
- Update documentation
- Ensure GPU compatibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets**: Thanks to the creators of TESS, RAVDESS, CREMA-D, and SAVEE datasets
- **Libraries**: PyTorch, Librosa, Streamlit, scikit-learn
- **Community**: Open-source AI and music recommendation research


<div align="center">

**Made with ❤️ and AI for music lovers everywhere**

⭐ Star this repo if you found it helpful!

[⬆️ Back to Top](#-voicemood-ai---emotion-based-music-recommendation-system)

</div>


