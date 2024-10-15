from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import model_predictor
import os
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Load both pre-trained models 
proposed_model_path = "saved_models/Baseline_model_freeze_unfreeze.keras"
baseline_model_path = "saved_models/Baseline_model_freeze_unfreeze.keras"
proposed_model = tf.keras.models.load_model(proposed_model_path)
baseline_model = tf.keras.models.load_model(baseline_model_path)

# Preprocess the audio
def preprocess_audio(audio_path, target_sr=16000, duration=3.0, n_fft=2048, hop_length=160):
    audio, sr = librosa.load(audio_path, sr=target_sr)

    if len(audio) > int(duration * target_sr):
        audio = audio[:int(duration * target_sr)]
    else:
        audio = np.pad(audio, (0, max(0, int(duration * target_sr) - len(audio))), 'constant')

    log_mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=128,
        fmax=8000,
        n_fft=n_fft,
        hop_length=hop_length
    )
    log_mel_spectrogram = librosa.power_to_db(log_mel_spectrogram, ref=np.max)

    if log_mel_spectrogram.shape[1] != 300:
        log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=300, axis=1)

    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)

    return log_mel_spectrogram

# Classify the stutter type using the proposed model
def classify_stutter_proposed(filename):
    return model_predictor.predict_stutter_type(filename, model_type='proposed')

# Classify the stutter type using the baseline model
def classify_stutter_baseline(filename):
    return model_predictor.predict_stutter_type(filename, model_type='baseline')

# Folder to store uploaded audio files
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio file types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)

        file = request.files['audio_file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('display_results', filename=filename))
    
    return render_template('upload.html')

@app.route('/results', methods=['GET'])
def display_results():
    filename = request.args.get('filename')

    # Check if the filename is valid
    if not filename or not allowed_file(filename):
        return redirect(url_for('upload_file'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Ensure the file exists
    if not os.path.exists(file_path):
        return "File not found", 404

    # Classify the stutter type using both models
    classification_result_1 = classify_stutter_proposed(filename)  # Changed to filename
    classification_result_2 = classify_stutter_baseline(filename)  # Changed to filename

    return render_template(
        'results.html',
        filename=filename, 
        classification_result_1=classification_result_1, 
        classification_result_2=classification_result_2
    )

@app.route('/upload_again', methods=['POST'])
def upload_again():
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
