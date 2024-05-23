import numpy as np
import pandas as pd
import os
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Шлях до папки з аудіофайлами
audio_parent_folder = 'data'
data = []

# Перебір усіх папок з аудіофайлами
for foldername in os.listdir(audio_parent_folder):
    current_folder = os.path.join(audio_parent_folder, foldername)
    if os.path.isdir(current_folder):
        for filename in os.listdir(current_folder):
            audio_path = os.path.join(current_folder, filename)
            y, sr = librosa.load(audio_path)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            mfccs_mean = np.mean(mfccs, axis=1)
            digit = int(filename[0])
            features = np.append(mfccs_mean, digit)
            data.append(features)

# Формування DataFrame
cols = ['mean_' + str(x+1) for x in range(20)] + ['Digit']
df = pd.DataFrame(data, columns=cols)

# Стандартизація фіч
X = df.drop("Digit", axis=1)
y = df["Digit"]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

# Розподіл даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.3, random_state=42)

# Навчання моделі (Random Forest в даному випадку)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Збереження моделі та скейлера
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Прогнозування та оцінка моделі
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

def process_audio(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)
    features = mfccs_mean.reshape(1, -1)

    # Завантаження моделі та скейлера
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Стандартизація фіч
    features_scaled = scaler.transform(features)

    # Прогнозування числа
    number = model.predict(features_scaled)[0]
    return number
