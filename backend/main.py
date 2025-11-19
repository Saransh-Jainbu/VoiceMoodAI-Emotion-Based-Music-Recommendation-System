"""
FastAPI Backend for VoiceMood AI
Emotion Detection & Music Recommendation API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Change working directory to parent for file access
os.chdir(parent_dir)

from utils.model_utils import load_model, predict_emotion
from utils.music_utils import get_recommendations
from config import SONGS_CSV_PATH, EMOTION_LABELS

# Initialize FastAPI app
app = FastAPI(
    title="VoiceMood API",
    description="AI-Powered Emotion Detection & Music Recommendation",
    version="2.0.0"
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
scaler = None
device = None

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    global model, scaler, device
    try:
        model, scaler, device = load_model()
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "VoiceMood API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "detect": "/api/detect",
            "emotions": "/api/emotions"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/api/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {
        "emotions": EMOTION_LABELS,
        "count": len(EMOTION_LABELS)
    }


@app.post("/api/detect")
async def detect_emotion(file: UploadFile = File(...)):
    """
    Detect emotion from uploaded audio file
    
    Returns:
        - emotion: detected emotion label
        - confidence: confidence scores for all emotions
        - recommendations: music recommendations
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate file type
    if not file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only .wav files are supported"
        )
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    
    try:
        # Save file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"Processing file: {temp_path}")
        
        # Predict emotion
        emotion, confidence_scores = predict_emotion(temp_path, model, scaler, device)
        
        if emotion == 'error' or confidence_scores is None:
            raise Exception("Prediction failed")
        
        print(f"Detected emotion: {emotion}")
        
        # Get music recommendations
        recommendations = get_recommendations(emotion, SONGS_CSV_PATH)
        
        # Format confidence scores
        confidence_dict = {
            label: float(score) 
            for label, score in zip(EMOTION_LABELS, confidence_scores)
        }
        
        # Sort by confidence
        sorted_confidence = dict(
            sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Format recommendations
        recommendations_list = []
        if not recommendations.empty:
            for _, row in recommendations.head(10).iterrows():
                recommendations_list.append({
                    'name': row['name'],
                    'artist': row['artist'],
                    'album': row.get('album', row.get('song_id', 'Unknown'))
                })
        
        return JSONResponse({
            "success": True,
            "data": {
                "emotion": emotion,
                "confidence": sorted_confidence,
                "top_confidence": float(confidence_scores.max()),
                "recommendations": recommendations_list
            }
        })
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/stats")
async def get_stats():
    """Get model statistics"""
    return {
        "model": {
            "accuracy": "97.09%",
            "parameters": "7.1M",
            "emotions": len(EMOTION_LABELS),
            "datasets": ["RAVDESS", "CREMA-D", "TESS", "SAVEE"]
        },
        "system": {
            "device": str(device) if device else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
