from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import librosa
import torch
from notebooks.proposed_model.config import get_config
from saved_models.proposed_model.stutclass_model import preprocess_concat_audio, get_model, predict
from notebooks.baseline_model.config import get_baseline_config
from saved_models.baseline_model.baseline_model import baseline_preprocess_concat_audio, get_transformer_model, baseline_predict

app = Flask(__name__)

proposed_model_path = "saved_models/proposed_model/Proposed_model.pt"
baseline_model_path = "saved_models/baseline_model/Baseline_model.pt"

# Label map
label_map = {
    0: 'Block',
    1: 'Interjection',
    2: 'No Stutter',
    3: 'Prolongation',
    4: 'Sound Repetition',
    5: 'Word Repetition'
}

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
def classify_stutter_proposed(audio_path):
    config = get_config()

    input = preprocess_concat_audio(audio_path)

    input = torch.from_numpy(input).float()
    input = input.unsqueeze(0)

    input_shape = input.shape[-1]
    timesteps = input.shape[1]

    model = get_model(config, input_shape, timesteps)

    saved_model = proposed_model_path

    print(f'Preloading model {saved_model}')
    print(f"Audio File: {audio_path}")
    state = torch.load(saved_model)
    model.load_state_dict(state['model_state_dict'])

    predicted_class, probabilities = predict(model, input)

    print(f"Predicted Class: {label_map.get(predicted_class)}")
    print(f"Probability Distribution: {probabilities}")

    return label_map.get(predicted_class)

# Classify the stutter type using the baseline model
def classify_stutter_baseline(audio_path):
    config = get_baseline_config()

    input = baseline_preprocess_concat_audio(audio_path)

    input = torch.from_numpy(input).float()
    input = input.unsqueeze(0)

    input_shape = input.shape[-1]
    timesteps = input.shape[1]

    model = get_transformer_model(config, input_shape, timesteps)

    baseline_model = baseline_model_path

    print(f'Preloading model {baseline_model}')
    print(f"Audio File: {audio_path}")
    state = torch.load(baseline_model)
    model.load_state_dict(state['model_state_dict'])

    predicted_class, probabilities = baseline_predict(model, input)

    print(f"Predicted Class: {label_map.get(predicted_class)}")
    print(f"Probability Distribution: {probabilities}")

    return label_map.get(predicted_class)

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
    classification_result_1 = classify_stutter_proposed(file_path)
    classification_result_2 = classify_stutter_baseline(file_path)

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
