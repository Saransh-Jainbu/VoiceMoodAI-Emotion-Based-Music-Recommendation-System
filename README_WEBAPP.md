# VoiceMood AI - Modern Web Application

A professional emotion detection and music recommendation system built with **Next.js** and **FastAPI**.

## 🚀 Tech Stack

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Axios** - HTTP client
- **Lucide React** - Icons

### Backend
- **FastAPI** - Python web framework
- **PyTorch** - Deep learning
- **librosa** - Audio processing
- **CUDA** - GPU acceleration

## 📦 Installation

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+
- **CUDA Toolkit** 12.1 (for GPU acceleration)

### Backend Setup

1. **Navigate to backend directory:**
```powershell
cd backend
```

2. **Create virtual environment (optional but recommended):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install PyTorch with CUDA:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install other dependencies:**
```powershell
pip install fastapi uvicorn python-multipart librosa soundfile numpy pandas scikit-learn joblib
```

5. **Start the FastAPI server:**
```powershell
python main.py
```

Server will run at: `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
```powershell
cd frontend
```

2. **Install dependencies:**
```powershell
npm install
```

3. **Start the development server:**
```powershell
npm run dev
```

Frontend will run at: `http://localhost:3000`

## 🎯 Usage

1. **Start the backend** (`http://localhost:8000`)
2. **Start the frontend** (`http://localhost:3000`)
3. **Open browser** and go to `http://localhost:3000`
4. **Upload a .wav file** (drag & drop or click to browse)
5. **View results**: emotion detection + music recommendations

## 📁 Project Structure

```
ai_project/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── api/                 # API routes (if expanded)
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main page
│   │   ├── layout.tsx       # Root layout
│   │   └── globals.css      # Global styles
│   ├── components/
│   │   ├── Header.tsx       # Header component
│   │   ├── AudioUploader.tsx # File upload
│   │   ├── EmotionResult.tsx # Results display
│   │   └── MusicRecommendations.tsx # Music cards
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   └── utils.ts         # Utilities
│   ├── package.json
│   └── .env.local           # Environment variables
│
├── utils/
│   ├── model_utils.py       # PyTorch model logic
│   └── music_utils.py       # Music recommendations
│
├── config.py                # Configuration
├── best_emotion_model.pth   # Trained model
├── scaler2.pickle           # Feature scaler
├── encoder2.pickle          # Label encoder
└── songs.csv                # Music database
```

## 🔌 API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /api/emotions` - List of emotions
- `POST /api/detect` - Detect emotion from audio
- `GET /api/stats` - Model statistics

## 🎨 Features

- ✨ **Modern UI** - Beautiful dark theme with animations
- 🎵 **Drag & Drop** - Easy file upload
- 📊 **Confidence Scores** - Visual confidence bars
- 🎧 **Music Recommendations** - Personalized song suggestions
- ⚡ **Real-time Processing** - Fast emotion detection
- 📱 **Responsive Design** - Works on all devices
- 🚀 **GPU Accelerated** - CUDA support for faster inference

## 🧪 Model Details

- **Architecture:** CNN (7.1M parameters)
- **Accuracy:** 97.09%
- **Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Datasets:** RAVDESS, CREMA-D, TESS, SAVEE
- **Input:** .wav audio files (2.5s, 22050Hz)
- **Features:** MFCC, ZCR, RMSE

## ⚠️ Important Notes

- Model trained on **professional acted speech**
- Best results with **clear, exaggerated emotional expressions**
- Use **.wav format** for audio files
- GPU recommended for faster processing

## 📝 Development Commands

### Backend
```powershell
# Run server with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```powershell
# Development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

## 🚀 Deployment

### Frontend (Vercel)
1. Push code to GitHub
2. Connect to Vercel
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy automatically

### Backend (Railway/Render)
1. Push code to GitHub
2. Connect to Railway/Render
3. Select `backend` directory
4. Deploy with Python buildpack

## 📄 License

MIT License - Feel free to use for your projects!

## 🙏 Acknowledgments

- PyTorch for deep learning framework
- Next.js for the amazing React framework
- FastAPI for the modern Python web framework
- RAVDESS, CREMA-D, TESS, SAVEE datasets

---

**Built with ❤️ using PyTorch, FastAPI, and Next.js**
