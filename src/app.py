from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import json
import zipfile

app = Flask(__name__)
DEFAULT_MODEL_PATH = "/app/model/phishsense-v2.pkl"

def start_languagetool():
    lt_dir = "/dependencies/LanguageTool-6.7-SNAPSHOT"
    jar_path = os.path.join(lt_dir, "languagetool-server.jar")
    props_path = os.path.join(lt_dir, "server.properties")

    subprocess.Popen([
        "java", "-cp", jar_path,
        "org.languagetool.server.HTTPServer",
        "--config", props_path,
        "--port", "8081",
        "--allow-origin"
    ])

start_languagetool() # Start LanguageTool server

@app.route('/predict', methods=['POST'])
def predict():
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file not found at: {model_path}'}), 500

    results = []

    # Case 1: ZIP file upload
    if 'email_zip' in request.files:
        uploaded_zip = request.files['email_zip']
        with tempfile.TemporaryDirectory() as extract_dir:
            zip_path = os.path.join(extract_dir, "emails.zip")
            uploaded_zip.save(zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            for fname in os.listdir(extract_dir):
                if fname.endswith(".zip"):
                    continue  # skip the zip itself
                fpath = os.path.join(extract_dir, fname)
                if not fname.lower().endswith(".eml"):
                    continue

                results.append(run_prediction(fpath, fname, model_path))

    # Case 2: One or more individual email files
    elif 'email_file' in request.files:
        files = request.files.getlist("email_file")
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as temp_file:
                f.save(temp_file.name)
                temp_path = temp_file.name
            results.append(run_prediction(temp_path, f.filename, model_path))
            os.remove(temp_path)

    else:
        return jsonify({'error': 'No valid input found. Please provide email_file(s) or email_zip.'}), 400

    return jsonify(results)

def run_prediction(email_path, filename, model_path):
    result = subprocess.run(
        [
            "python", "src/predict_email.py",
            "--email_path", email_path,
            "--model_path", model_path,
            "--show", "y"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        return {
            'filename': filename,
            'error': result.stderr.strip()
        }

    try:
        prediction = json.loads(result.stdout.strip())
        return {
            'filename': filename,
            'prediction': prediction
        }
    except json.JSONDecodeError:
        return {
            'filename': filename,
            'error': 'Invalid JSON output',
            'raw_output': result.stdout.strip()
        }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
