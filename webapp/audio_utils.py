import os
import sys
import tempfile

import librosa
import soundfile as sf
import json

# Aggiunta al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yamnet_model')))

from analyze_yamnet import analyze_audio, load_class_map

def save_results_to_json(results, output_path):
    """
    Salva i risultati dell'analisi in formato JSON.

    :param results: Dizionario o lista dei risultati ottenuti dall'analisi.
    :param output_path: Percorso completo del file di output .json.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Errore durante il salvataggio dei risultati JSON: {e}")

def analyze_audio_in_segments(file_path, segment_duration=5.0):
    waveform, sr = librosa.load(file_path, sr=16000)
    total_duration = len(waveform) / sr
    segment_samples = int(segment_duration * sr)

    class_names = load_class_map()
    
    results = []
    for i in range(0, len(waveform), segment_samples):
        segment = waveform[i:i+segment_samples]
        if len(segment) < int(0.5 * segment_samples):  # meno di metà segmento
            continue
        segment_start = i / sr
        segment_end = min((i + segment_samples) / sr, total_duration)

        # Salva il segmento temporaneamente su disco
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, segment, sr)
            tmp_segment_path = tmpfile.name

        try:
            result = analyze_audio(tmp_segment_path, class_names)
            result["segment_start"] = segment_start
            result["segment_end"] = segment_end
            result["segment_index"] = i // segment_samples
            results.append(result)
        finally:
            os.remove(tmp_segment_path)

    return results