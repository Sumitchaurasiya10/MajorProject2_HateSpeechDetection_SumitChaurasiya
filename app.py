from flask import Flask, render_template, request, jsonify
import os
import base64
from utils.predictor import HateSpeechPredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max (for video)

predictor = HateSpeechPredictor()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-analysis')
def text_analysis():
    return render_template('text_analysis.html')

@app.route('/media-analysis')
def media_analysis():
    return render_template('media_analysis.html')

@app.route('/api/health')
def health():
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    classifier = predictor.get_classifier_type()
    # Priority: bilstm > claude_ai > adam > keyword
    if classifier == 'keyword' and api_key:
        classifier = 'claude_ai'
    adam_ready = predictor.adam_clf is not None
    return jsonify({
        'status': 'ok',
        'classifier': classifier,
        'adam_ready': adam_ready,
        'ocr_ready': True,
        'claude_ai': bool(api_key)
    })

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 chars)'}), 413

    result = predictor.predict(text)
    return jsonify(result)

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400
    
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts must be a non-empty list'}), 400
    if len(texts) > 50:
        return jsonify({'error': 'Max 50 texts per batch'}), 413

    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            result = predictor.predict(text.strip())
            results.append({'text': text, **result})

    summary = {
        'total': len(results),
        'hate_speech': sum(1 for r in results if r.get('prediction') == 'Hate Speech'),
        'offensive': sum(1 for r in results if r.get('prediction') == 'Offensive'),
        'clean': sum(1 for r in results if r.get('prediction') == 'Normal')
    }
    return jsonify({'results': results, 'summary': summary})

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type'}), 400

    img_bytes = file.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    result = predictor.predict_image(img_b64, ext)
    return jsonify(result)

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type. Use MP4, AVI, MOV, MKV, WEBM'}), 400

    # Save temp file
    temp_path = os.path.join(UPLOAD_FOLDER, 'temp_video.' + ext)
    file.save(temp_path)

    try:
        result = predictor.predict_video(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 200MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
