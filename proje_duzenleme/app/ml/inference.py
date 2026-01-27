import numpy as np
import librosa
from app.ml.model_loader import model, label_encoder

MAX_LEN = 259
N_MFCC = 40
SR = 22050

def extract_mfcc_for_dl(file_path):
    y, _ = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc

def predict_audio(file_path):
    mfcc = extract_mfcc_for_dl(file_path)
    mfcc = mfcc.reshape(1, MAX_LEN, N_MFCC)

    probs = model.predict(mfcc, verbose=0)
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = (
        label_encoder.inverse_transform([idx])[0]
        if label_encoder
        else str(idx)
    )

    return label, confidence