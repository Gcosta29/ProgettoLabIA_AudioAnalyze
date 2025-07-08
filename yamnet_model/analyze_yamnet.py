import os
import librosa
from collections import Counter
import numpy as np
import tensorflow_hub as hub
import csv
import time

# Carica modello YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
MULTICLASS = 5
AUDIO_FOLDER = "/app/UrbanSound8K"  #Volume montato dal docker_compose
ANALYSIS = 1   #Per il Debug = per saltare la fase di analisi audio

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


def analyze_audio(file_path):
    THRESHOLD = 0.5
    try:
        waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet richiede 16kHz
        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores = scores.numpy()

        window_labels = []
        for i, frame_scores in enumerate(scores):
            active_indices = np.where(frame_scores > THRESHOLD)[0]
            active_labels = [class_names[j] for j in active_indices]
            window_labels.extend(active_labels)

        # Le 3 etichette più frequenti
        counts = Counter(window_labels).most_common(MULTICLASS)

        duration = len(waveform) / sr
        num_frames = len(scores)

        return {
            "file_path": file_path,
            "top_labels": counts,
            "all_labels": window_labels,
            "duration": duration,
            "num_frames": num_frames
        }

    except Exception as e:
        return {
            "file_path": file_path,
            "error": str(e)
        }
    



# === Carica il mapping: AudioSet → UrbanSound8K ===
label_mapping = {}
with open("urbansound8k_audioset_mapping.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        audioset_label = row["AudioSet_label_name"].strip()
        urbansound_label = row["UrbanSound8K_class"].strip()
        label_mapping[audioset_label] = urbansound_label


output_csv = "/app/output/risultati_analisi.csv"
class_names = load_class_map()

# === Prepara il file per la srittura dei risultati ===
# Se il file esiste già, lo sovrascriviamo solo una volta
file_exists = os.path.exists(output_csv)        

# === Analizza tutti gli audio ===
if(ANALYSIS):
    results = []
    start_time = time.time()
    nFolder = 0
    nFile = 0
    for root, dirs, files in os.walk(AUDIO_FOLDER):  
        nFolder += 1   
        for filename in files:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                nFile += 1
                filepath = os.path.join(root, filename)
                result = analyze_audio(filepath)
                
                if "error" in result:
                    print(f"⚠️ Errore su {filepath}: {result['error']}")
                    continue
                elapsed = time.time() - start_time
                print(f"\n📊 Analisi Filefile: {filepath}")
                print(f"\nFile {nFile}/8.732\nFolder: {nFolder},{root}\nTempo trascorso")
                print(f"Tempo: {round(elapsed, 2)} secondi")
                # Mappa le top label YAMNet verso le corrispondenti UrbanSound8K (se esistono)
                mapped_labels = []
                for label, count in result["top_labels"]:
                    mapped = label_mapping.get(label)
                # if not mapped:          #Solo per aggiungere nuovi mapping
                #     mapped = check_or_confirm_mapping(label, result["file_path"])
                    if mapped:
                        mapped_labels.append((mapped, count))
                    else:
                        mapped_labels.append((f"[Nessuna Corrispondenza: {label}]", count))
                # Aggiungi ai risultati
                results.append({
                    "file_path": result["file_path"],
                    "duration": result["duration"],
                    "num_frames": result["num_frames"],
                    "yamnet_top_labels": result["top_labels"],
                    "mapped_labels": mapped_labels
                })

    # == Preparazione File CSV e salvataggio dei risultati
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_path',
            'duration',
            'num_frames',
        ]

        # Aggiungi dinamicamente le colonne yamnet_top_N_label e yamnet_top_N_count
        for i in range(1, MULTICLASS + 1):
            fieldnames.append(f'yamnet_top_{i}_label')
            fieldnames.append(f'yamnet_top_{i}_count')

        # Aggiungi dinamicamente le colonne mapped_label_N e mapped_count_N
        for i in range(1, MULTICLASS + 1):
            fieldnames.append(f'mapped_label_{i}')
            fieldnames.append(f'mapped_count_{i}')

        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in results:
                row = {
                    'file_path': entry['file_path'],
                    'duration': round(entry['duration'], 2),
                    'num_frames': entry['num_frames'],
                }

                # Aggiungi fino a MULTICLASS top label YAMNet
                for i, (label, count) in enumerate(entry["yamnet_top_labels"][:MULTICLASS], start=1):
                    row[f'yamnet_top_{i}_label'] = label
                    row[f'yamnet_top_{i}_count'] = count

                # Aggiungi fino a MULTICLASS mapped label UrbanSound8K
                for i, (label, count) in enumerate(entry["mapped_labels"][:MULTICLASS], start=1):
                    row[f'mapped_label_{i}'] = label
                    row[f'mapped_count_{i}'] = count

                writer.writerow(row)

        print(f"\n✅ Analisi completata. File analizzati: {nFile}")
        print(f"📄 Risultati salvati in: {output_csv}")
        elapsed = time.time() - start_time
        print(f"⏱️ Tempo totale: {round(elapsed, 2)} secondi")






# == Caricamento mapping filename → UrbanSound8K correct label
urbansound_metadata = {}
with open('/app/UrbanSound8K/metadata/UrbanSound8K.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row['slice_file_name'].strip()
        class_name = row['class'].strip()
        urbansound_metadata[filename] = class_name

match_count = 0
mismatch_count = 0
total = 0


with open(output_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_path = row['file_path'].strip()
        filename = os.path.basename(file_path)

        if filename not in urbansound_metadata:
            continue  # file non trovato nel metadata

        expected_label = urbansound_metadata[filename]

        # Raccogli mapped_label e count, ignorando quelli con "[Nessuna Corrispondenza:"
        mapped = []
        for i in range(1, 4):
            label_key = f'mapped_label_{i}'
            count_key = f'mapped_count_{i}'

            label = row[label_key].strip()
            count = int(row[count_key]) if row[count_key].strip().isdigit() else 0

            if not label.startswith('[Nessuna Corrispondenza'):
                mapped.append((label, count))

        if not mapped:
            # Nessuna mappatura valida → mismatch
            mismatch_count += 1
        else:
            # Controlla se almeno una mapped_label corrisponde all'expected_label
            labels = [label for label, _ in mapped]
            if expected_label in labels:
                match_count += 1
            else:
                mismatch_count += 1

        total += 1

# Risultati
print(f'Totale file analizzati: {total}')
print(f'Match: {match_count}')
print(f'Mismatch: {mismatch_count}')
print(f'Accuracy: {match_count / total:.2%}' if total > 0 else 'Accuracy: N/A')
