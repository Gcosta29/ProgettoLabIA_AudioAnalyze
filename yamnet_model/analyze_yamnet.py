import os
import librosa
from collections import Counter
import numpy as np
import tensorflow_hub as hub
import csv

# Carica modello YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Carica mappa delle classi
def load_class_map(path='yamnet_class_map.csv'):
    import requests
    import os
    url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, 'w') as f:
            f.write(r.text)
    class_names = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_names = load_class_map()

def analyze_audio(file_path):
    THRESHOLD = 0.3

    print(f"\n🎧 Analizzando: {file_path}")
    try:
        waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet richiede 16kHz
        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores = scores.numpy()

        print(f"Segmenti in {file_path}:")
        window_labels = []
        for i, frame_scores in enumerate(scores):
            start_time = i * 0.48  # Ogni frame ha uno step di ~0.48s
           
            active_indices = np.where(frame_scores > THRESHOLD)[0]  #raccolgo gli indici delle classi che superano la soia di treshold
            active_labels = [class_names[j] for j in active_indices] #raccolgo le specefiche etichette associate agli indici attivi
            window_labels.extend(active_labels)
            
            
        counts = Counter(window_labels).most_common(3)  # le 3 più frequenti
        label_string = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in counts])
        print(f"⏱ Most common labels over frames → {label_string}")

    except Exception as e:
        print(f"⚠️ Errore con il file {file_path}: {e}")

# === Scansiona tutti gli audio nella cartella ===
AUDIO_FOLDER = "/app/UrbanSound8K"  #Volume montato dal docker_compose

for root, dirs, files in os.walk(AUDIO_FOLDER):    #analizza cartelle e sottocartelle     
    for filename in files:
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            filepath = os.path.join(root, filename)
            analyze_audio(filepath)