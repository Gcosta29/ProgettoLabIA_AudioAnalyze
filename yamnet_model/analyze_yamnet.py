import os
import librosa
from collections import Counter
import numpy as np
import tensorflow_hub as hub
import csv
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import matplotlib.pyplot as plt


MULTICLASS = 3
AUDIO_FOLDER = "/app/UrbanSound8K"  #Volume montato dal docker_compose
ANALYSIS = 0 #Per il Debug = per saltare la fase di analisi audio
MIN_DURATION_SECONDS = 0.0  # audio più brevi di MIN_DURATION_SECONDS secondi verranno ignorati
THRESHOLD = 0.3
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


def analyze_audio(file_path, class_names):
    # Carica modello YAMNet
    try:
        waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet richiede 16kHz
        duration = len(waveform) / sr
         
        if duration < MIN_DURATION_SECONDS:
            return {
                "file_path": file_path,
                "error": f"Audio troppo corto ({duration:.2f} sec)"
            }
        
        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores = scores.numpy()

        window_labels = []
        for i, frame_scores in enumerate(scores):
            active_indices = np.where(frame_scores > THRESHOLD)[0]
            active_labels = [class_names[j] for j in active_indices]
            window_labels.extend(active_labels)

        # Le 3 etichette più frequenti
        counts = Counter(window_labels).most_common(MULTICLASS)

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
    

def run_batch_analysis():
    # === Carica il mapping: AudioSet → UrbanSound8K ===
    label_mapping = {}
    csv_path = os.path.join(os.path.dirname(__file__), "urbansound8k_audioset_mapping.csv")
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audioset_label = row["AudioSet_label_name"].strip()
            urbansound_label = row["UrbanSound8K_class"].strip()
            label_mapping[audioset_label] = urbansound_label

    output_csv = "/app/output/risultati_analisi.csv"

    # === Prepara il file per la srittura dei risultati ===
    # Se il file esiste già, lo sovrascriviamo solo una volta
    file_exists = os.path.exists(output_csv)        

    # === Analizza tutti gli audio ===
    if(ANALYSIS):
        class_names = load_class_map()
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
                    result = analyze_audio(filepath, class_names)
                    
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
                        if mapped:
                            mapped_labels.append((mapped, count))
                        else:
                            mapped_labels.append((f"Nessuna Corrispondenza: {label}", count))
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
    correct_labels_num = [0,0,0,0,0,0,0,0,0,0]
    urbansound_metadata = {}
    class_name_to_id = {}
    with open('/app/UrbanSound8K/metadata/UrbanSound8K.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_name = row['class'].strip()
            class_id = int(row['classID'])
            class_name_to_id[class_name] = class_id
            correct_labels_num[class_id] += 1
            urbansound_metadata[row['slice_file_name'].strip()] = class_name

    
    # == Controllo etichette risultato con etichette corrette in UrbanSound8K/metadata 
    match_count = 0
    Unidentified_count = 0
    mismatch_count = 0
    total = 0

    y_true = []  # Etichetta reale (UrbanSound8K)
    y_pred = []  # Etichetta predetta (dalla mappatura YAMNet)

    all_labels_set = set()  # Per costruire lista completa delle etichette

    with open(output_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row['file_path'].strip()
            filename = os.path.basename(file_path)

            if filename not in urbansound_metadata:
                continue  # file non trovato nel metadata

            expected_label = urbansound_metadata[filename]
            if expected_label.strip() != '':
                all_labels_set.add(expected_label)

            # Raccogli mapped_label e count, ignorando quelli con "Nessuna Corrispondenza"
            mapped = []
            for i in range(1, MULTICLASS + 1):
                label_key = f'mapped_label_{i}'
                count_key = f'mapped_count_{i}'

                label = row[label_key].strip()
                count = int(row[count_key]) if row[count_key].strip().isdigit() else 0

                if not label.startswith('Nessuna Corrispondenza'):
                    mapped.append((label, count))

            if not mapped:
                predicted_label = "Nessuna Corrispondenza"
            else:
                # Predici la più frequente tra le label valide
                from collections import defaultdict
                label_counts = defaultdict(int)
                for label, count in mapped:
                    label_counts[label] += count

                if label_counts:
                    max_count = max(label_counts.values())
                    top_labels = [label for label, count in label_counts.items() if count == max_count]

                    # Se c'è un pareggio e la label corretta è tra quelle top, usala
                    if expected_label in top_labels:
                        predicted_label = expected_label
                    else:
                        # Altrimenti scegli una a caso (o la prima)
                        predicted_label = top_labels[0]
                else:
                    predicted_label = "Nessuna Corrispondenza"

            if predicted_label.strip() != '':
                all_labels_set.add(predicted_label)
        

            y_true.append(expected_label)
            y_pred.append(predicted_label)

            if predicted_label == expected_label:
                match_count += 1
            elif predicted_label == '':
                Unidentified_count += 1
            else:
                mismatch_count += 1

            total += 1
   
    labels_order = sorted(all_labels_set)  # Ordine fisso per classi
    # Risultati
    # 1. Accuracy generale
    acc = accuracy_score(y_true, y_pred)

    print(f"\nOverall Accuracy: {acc:.2%}")

    # 2. Report compatto
    print("\nClassification report:\n", classification_report(y_true, y_pred, labels=labels_order, zero_division=0))

    print(f'Totale file analizzati: {total}')
    print(f'File non classificati: {Unidentified_count}')
    print(f'Match: {match_count}')
    print(f'Mismatch: {mismatch_count}')
    print(f'Accuracy: {match_count / (total - Unidentified_count):.2%}' if total > 0 else 'Accuracy: N/A')

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=labels_order, normalize='true')*100


    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    im = disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=90, values_format=".0f")
    #plt.colorbar(im.im_, ax=ax, label='Percentuale (%)', fraction=0.046, pad=0.04)

    # Forza le etichette manualmente
    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=90)
    yticklabels_with_counts = []
    for label in labels_order:
        class_id = class_name_to_id.get(label, -1)
        count = correct_labels_num[class_id] if class_id >= 0 else 0
        yticklabels_with_counts.append(f"{label} ({count})")

    ax.set_yticklabels(yticklabels_with_counts)

    plt.title("Confusion Matrix - UrbanSound8K vs YAMNet Mapped Labels")
    plt.tight_layout()
    plt.savefig("/app/output/confusion_matrix.png")
    print("✅ Confusion matrix salvata in: /app/output/confusion_matrix.png")

if __name__ == "__main__":
    run_batch_analysis()
