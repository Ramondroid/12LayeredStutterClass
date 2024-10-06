from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Folder to store uploaded audio files | ang folder saan mapupunta ang uploaded file ni user!!
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio file types | 3 only allowed files, para hindi ma upload ang images chunenes!
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the audio file part | para macheck natin kung may uploadedf na ba
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

    # DITO YUNG CODE NG MODEL BOSSING !!!
    # mag eme eme muna tayo ng results
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
