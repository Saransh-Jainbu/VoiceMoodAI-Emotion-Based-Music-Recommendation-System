"""
Music recommendation utilities
"""
import csv
import pandas as pd
from config import EMOTION_LABELS


def get_recommendations(emotion, songs_csv_path='songs.csv'):
    """Get music recommendations based on emotion"""
    filename = songs_csv_path
    
    # Emotion mapping
    emotion_map = {
        'happy': 'Happy',
        'sad': 'Sad',
        'neutral': 'Neutral',
        'calm': 'Calm',
        'angry': 'Angry',
        'fear': 'Fear',
        'disgust': 'Disgust',
        'surprise': 'Surprise'
    }
    
    # Synonyms for better matching
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
    
    emotion_label = emotion_map.get(emotion.lower(), 'Neutral')
    songs = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            rows = list(csvreader)
            
            for i in range(1, len(rows)):  # Skip header
                if len(rows[i]) < 4:
                    continue
                    
                words = [w.strip().lower() for w in rows[i][3].split(",")]
                
                # Check exact match
                if emotion_label.lower() in words:
                    songs.append(rows[i])
                    continue
                
                # Check synonyms
                for syn in synonyms.get(emotion.lower(), []):
                    if syn in words:
                        songs.append(rows[i])
                        break
        
        if songs:
            # Return DataFrame with name, artist, album columns to match API expectations
            # CSV format: Song, Artist, Title, MoodsStrSplit, SampleURL
            df = pd.DataFrame(songs, columns=['song_id', 'artist', 'name', 'mood', 'url'])
            return df[['name', 'artist', 'song_id']]
        else:
            return pd.DataFrame(columns=['name', 'artist', 'album'])
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['name', 'artist', 'album'])
