import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from .audio_utils import analyze_audio_in_segments, save_results_to_json

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verifica che ci sia un file
        if 'file' not in request.files:
            return render_template('index.html', error="Nessun file selezionato.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Nome file mancante.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Analisi audio in segmenti
            results = analyze_audio_in_segments(path, segment_duration=5.0)

            # Salvataggio risultati JSON
            output_path = os.path.splitext(path)[0] + "_results.json"
            save_results_to_json(results, output_path)

            return render_template('results.html', results=results, audio_file=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)