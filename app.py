from flask import Flask, render_template, request, redirect, url_for
# Insert Proposed model here !!
# Insert Baseline model here !!
from werkzeug.utils import secure_filename
# Insert Librosa for Feature or audio processing !!
import os

app = Flask(__name__)

# pmodel = load_pmodel('pmodel path') # proposed model
# bmodel = load+bmodel('bmodel path') # baseline model

# Folder to store uploaded audio files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio file types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the audio file part 
        if 'audio_file' not in request.files:
            return redirect(request.url)

        file = request.files['audio_file']
        label = request.form['label']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('display_results', filename=filename, label=label))
    
    return render_template('upload.html')



@app.route('/results', methods=['GET'])
def display_results():
    filename = request.args.get('filename')
    label = request.args.get('label')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # DITO YUNG CODE NG MODEL BOSSING !!!
    
    # Process Audio FIles (Feature Extraction ng model natin !)
    # audio_data, sample_rate = librosa.load(file_path, sr=None) #example
    
    # Prepare the audio for the model (Feature Fusion?)
    #features = extract_features_from_audio(audio_data) #example
    
    # Get Predictions
    
    classification_result_1 = "blockings"
    classification_result_2 = "No stutter"

    return render_template('results.html', filename=filename, label=label, 
                           classification_result_1=classification_result_1, 
                           classification_result_2=classification_result_2)

@app.route('/upload_again', methods=['POST'])
def upload_again():
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
