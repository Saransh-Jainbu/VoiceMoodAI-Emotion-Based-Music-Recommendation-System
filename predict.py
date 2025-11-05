import pickle
from io import BytesIO
import librosa
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
import warnings
import sys
from pydub import AudioSegment
import streamlit as st
import tempfile
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Load the model architecture from JSON
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

print("Loaded model architecture from disk")

with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Done")


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
    final_result = np.expand_dims(i_result, axis=2)

    return final_result

emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

def recommendations(s):
    filename = 'songs.csv'
    emotion = ""
    fields = []
    rows = []
    songs = []
    songs1=[]
    if s == 'happy':
        emotion ='Happy'
    elif s == 'sad':
        emotion ='Sad'
    elif s == 'neutral':
        emotion ='Neutral'
    elif s == 'calm':
        emotion ='Calm'
    elif s == 'angry':
        emotion ='Angry'
    else:
        print("no emotion")
    final=[]
    ind_pos = [1,2]


    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)

        for i in range(csvreader.line_num-1):
            words = rows[i][3].split(",")
            for word in words:
                if word.strip() == emotion:
                    songs.append(rows[i])
            # get total number of rows
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        for j in range(len(songs)):
            final.append([songs[j][i] for i in ind_pos])
        pd_dataframe = pd.DataFrame(final, columns=['Artist', 'Song'])
        st.write(" The songs that match your mood are : ")
        return pd_dataframe


def convert_to_wav(input_file):
    try:
        audio = AudioSegment.from_file(input_file)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            return temp_wav_file.name
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

st.title("Music Recommendation using Voice Emotion")
uploaded_file = st.file_uploader("Upload Your Audio File", type=["wav"])
if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        # Write the uploaded file content to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Display the audio
    st.audio(uploaded_file, format='audio/wav')
    
    # Make prediction using the temporary file path
    s = prediction(tmp_path)
    st.write(f"Your mood is {s}")
    dataframe = recommendations(s)
    st.write(dataframe)
    
    # Clean up the temporary file
    import os
    os.unlink(tmp_path)
