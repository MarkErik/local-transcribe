"""
Web-based transcript comparison tool.

Provides:
- Side-by-side diff visualization
- Audio playback with speed control
- Statistics and analysis dashboard
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from .extractor import extract_from_file, extract_from_text, ExtractedTranscript
from .diff_engine import compute_diff, generate_aligned_html, DiffResult, DiffType


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "uploads"

# Ensure upload folder exists
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# Global state for current comparison
current_state = {
    "transcript_a": None,
    "transcript_b": None,
    "diff_result": None,
    "audio_file": None,
}


@app.route("/")
def index():
    """Main comparison page."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Handle transcript and audio file uploads."""
    response = {"success": False, "errors": []}
    
    try:
        # Handle transcript A
        if "transcript_a" in request.files:
            file_a = request.files["transcript_a"]
            if file_a.filename:
                filepath_a = app.config["UPLOAD_FOLDER"] / secure_filename(file_a.filename)
                file_a.save(filepath_a)
                current_state["transcript_a"] = extract_from_file(filepath_a)
                response["transcript_a"] = {
                    "filename": file_a.filename,
                    "word_count": current_state["transcript_a"].word_count,
                    "format": current_state["transcript_a"].format_type,
                }
        
        # Handle transcript B
        if "transcript_b" in request.files:
            file_b = request.files["transcript_b"]
            if file_b.filename:
                filepath_b = app.config["UPLOAD_FOLDER"] / secure_filename(file_b.filename)
                file_b.save(filepath_b)
                current_state["transcript_b"] = extract_from_file(filepath_b)
                response["transcript_b"] = {
                    "filename": file_b.filename,
                    "word_count": current_state["transcript_b"].word_count,
                    "format": current_state["transcript_b"].format_type,
                }
        
        # Handle audio file
        if "audio_file" in request.files:
            audio = request.files["audio_file"]
            if audio.filename:
                audio_path = app.config["UPLOAD_FOLDER"] / secure_filename(audio.filename)
                audio.save(audio_path)
                current_state["audio_file"] = str(audio_path)
                response["audio_file"] = audio.filename
        
        response["success"] = True
        
    except Exception as e:
        response["errors"].append(str(e))
    
    return jsonify(response)


@app.route("/api/load-local", methods=["POST"])
def load_local_files():
    """Load files from local filesystem paths."""
    data = request.json
    response = {"success": False, "errors": []}
    
    try:
        if data.get("transcript_a_path"):
            path_a = Path(data["transcript_a_path"])
            current_state["transcript_a"] = extract_from_file(path_a)
            response["transcript_a"] = {
                "filename": path_a.name,
                "word_count": current_state["transcript_a"].word_count,
                "format": current_state["transcript_a"].format_type,
            }
        
        if data.get("transcript_b_path"):
            path_b = Path(data["transcript_b_path"])
            current_state["transcript_b"] = extract_from_file(path_b)
            response["transcript_b"] = {
                "filename": path_b.name,
                "word_count": current_state["transcript_b"].word_count,
                "format": current_state["transcript_b"].format_type,
            }
        
        if data.get("audio_path"):
            audio_path = Path(data["audio_path"])
            if audio_path.exists():
                current_state["audio_file"] = str(audio_path)
                response["audio_file"] = audio_path.name
        
        response["success"] = True
        
    except Exception as e:
        response["errors"].append(str(e))
    
    return jsonify(response)


@app.route("/api/compare", methods=["POST"])
def compare_transcripts():
    """Run comparison on loaded transcripts."""
    if not current_state["transcript_a"] or not current_state["transcript_b"]:
        return jsonify({
            "success": False,
            "error": "Please load both transcripts first"
        })
    
    try:
        # Compute diff
        diff_result = compute_diff(
            current_state["transcript_a"].words,
            current_state["transcript_b"].words
        )
        current_state["diff_result"] = diff_result
        
        # Generate aligned HTML
        html_a, html_b = generate_aligned_html(diff_result)
        
        # Build response
        response = {
            "success": True,
            "html_a": html_a,
            "html_b": html_b,
            "statistics": {
                "total_words_a": diff_result.total_words_a,
                "total_words_b": diff_result.total_words_b,
                "matching_words": diff_result.matching_words,
                "inserted_words": diff_result.inserted_words,
                "deleted_words": diff_result.deleted_words,
                "replaced_words_a": diff_result.replaced_words_a,
                "replaced_words_b": diff_result.replaced_words_b,
                "similarity_ratio": round(diff_result.similarity_ratio * 100, 2),
                "word_error_rate": round(diff_result.word_error_rate * 100, 2),
            },
            "common_substitutions": [
                {"word_a": s[0], "word_b": s[1], "count": s[2]}
                for s in diff_result.common_substitutions[:15]
            ],
            "unique_to_a": [
                {"word": w[0], "count": w[1]}
                for w in diff_result.unique_to_a[:15]
            ],
            "unique_to_b": [
                {"word": w[0], "count": w[1]}
                for w in diff_result.unique_to_b[:15]
            ],
            "diff_segments": _serialize_diff_segments(diff_result),
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def _serialize_diff_segments(result: DiffResult) -> list[dict]:
    """Serialize diff segments for JSON response."""
    segments = []
    for i, seg in enumerate(result.segments):
        if seg.diff_type != DiffType.EQUAL:
            segments.append({
                "index": i,
                "type": seg.diff_type.value,
                "words_a": seg.words_a,
                "words_b": seg.words_b,
                "position_a": seg.position_a,
                "position_b": seg.position_b,
            })
    return segments


@app.route("/api/audio/<path:filename>")
def serve_audio(filename):
    """Serve uploaded audio file."""
    if current_state["audio_file"]:
        return send_file(current_state["audio_file"])
    return jsonify({"error": "No audio file loaded"}), 404


@app.route("/api/get-audio-path")
def get_audio_path():
    """Get the current audio file path."""
    if current_state["audio_file"]:
        return jsonify({
            "success": True,
            "path": current_state["audio_file"],
            "filename": Path(current_state["audio_file"]).name
        })
    return jsonify({"success": False, "error": "No audio loaded"})


def run_server(host: str = "127.0.0.1", port: int = 5050, debug: bool = True):
    """Run the Flask development server."""
    print(f"\n🎯 Transcript Comparison Tool")
    print(f"   Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
