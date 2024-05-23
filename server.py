from flask import Flask, request, jsonify
from flask_cors import CORS
import model_forest  # Імпорт вашої моделі

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Тут ви обробляєте файл за допомогою вашої моделі
    number = model_forest.process_audio(file)

    return jsonify({'number': number})

if __name__ == '__main__':
    app.run(debug=True)
