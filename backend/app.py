import os
import io
import time
import json
from flask import Flask, request, jsonify, send_from_directory, send_file

try:
    from .audio_utils import load_audio_file
    from .model_detector import run_deepfake_analysis
except ImportError:
    from audio_utils import load_audio_file
    from model_detector import run_deepfake_analysis

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB Max Payload Limit

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/")
def index():
    root_html = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
    if os.path.exists(root_html):
        return send_file(root_html)
    frontend_html = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))
    if os.path.exists(frontend_html):
        return send_file(frontend_html)
    return "VocalTrace API Server Running", 200

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "VocalTrace Deepfake Detection Engine",
        "timestamp": time.time()
    })

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_audio_api():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "audio" not in request.files and not request.data:
        return jsonify({"error": "No audio file provided"}), 400
    
    filename = "uploaded_audio.wav"
    if "audio" in request.files:
        file = request.files["audio"]
        filename = file.filename or filename
        audio_bytes = file.read()
    else:
        audio_bytes = request.data

    try:
        start_t = time.time()
        audio_data, sr = load_audio_file(io.BytesIO(audio_bytes))
        results = run_deepfake_analysis(audio_data, sr, filename=filename)
        results["analysis_time_sec"] = round(time.time() - start_t, 3)
        return jsonify({
            "success": True,
            "filename": filename,
            "sample_rate": sr,
            "duration_sec": round(len(audio_data) / float(sr), 2),
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/samples/<path:filename>")
def serve_samples(filename):
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    return send_from_directory(samples_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting VocalTrace Backend Server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
