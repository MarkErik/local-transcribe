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

from .extractor import (
    extract_from_file, extract_from_text, ExtractedTranscript,
    extract_script_from_file, ExtractedScript, ScriptTurn, is_script_format
)
from .diff_engine import (
    compute_diff, generate_aligned_html, DiffResult, DiffType,
    analyze_transcript, compare_analyses, serialize_analysis,
    find_similar_word_pairs
)


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
    # Script mode state
    "script_raw": None,
    "script_cleaned": None,
    "comparison_mode": "word",  # "word" or "script"
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
        
        # Perform detailed analysis on each transcript
        analysis_a = analyze_transcript(current_state["transcript_a"].words)
        analysis_b = analyze_transcript(current_state["transcript_b"].words)
        
        # Compare the analyses
        analysis_comparison = compare_analyses(analysis_a, analysis_b)
        
        # Find similar word pairs (potential transcription errors)
        similar_pairs = find_similar_word_pairs(diff_result)
        
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
            # New detailed analysis
            "analysis_a": serialize_analysis(analysis_a),
            "analysis_b": serialize_analysis(analysis_b),
            "analysis_comparison": analysis_comparison,
            "similar_word_pairs": similar_pairs,
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


# ============================================================================
# Script Comparison Mode (Raw vs LLM-Cleaned)
# ============================================================================

@app.route("/api/upload-script", methods=["POST"])
def upload_script_files():
    """Handle script transcript file uploads for raw vs cleaned comparison."""
    response = {"success": False, "errors": []}
    
    try:
        # Handle raw script
        if "script_raw" in request.files:
            file_raw = request.files["script_raw"]
            if file_raw.filename:
                filepath_raw = app.config["UPLOAD_FOLDER"] / secure_filename(file_raw.filename)
                file_raw.save(filepath_raw)
                current_state["script_raw"] = extract_script_from_file(filepath_raw)
                response["script_raw"] = {
                    "filename": file_raw.filename,
                    "total_turns": current_state["script_raw"].total_turns,
                    "total_words": current_state["script_raw"].total_words,
                    "speakers": current_state["script_raw"].speakers,
                }
        
        # Handle cleaned script
        if "script_cleaned" in request.files:
            file_cleaned = request.files["script_cleaned"]
            if file_cleaned.filename:
                filepath_cleaned = app.config["UPLOAD_FOLDER"] / secure_filename(file_cleaned.filename)
                file_cleaned.save(filepath_cleaned)
                current_state["script_cleaned"] = extract_script_from_file(filepath_cleaned)
                response["script_cleaned"] = {
                    "filename": file_cleaned.filename,
                    "total_turns": current_state["script_cleaned"].total_turns,
                    "total_words": current_state["script_cleaned"].total_words,
                    "speakers": current_state["script_cleaned"].speakers,
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


@app.route("/api/load-script-local", methods=["POST"])
def load_script_local_files():
    """Load script files from local filesystem paths."""
    data = request.json
    response = {"success": False, "errors": []}
    
    try:
        if data.get("script_raw_path"):
            path_raw = Path(data["script_raw_path"])
            current_state["script_raw"] = extract_script_from_file(path_raw)
            response["script_raw"] = {
                "filename": path_raw.name,
                "total_turns": current_state["script_raw"].total_turns,
                "total_words": current_state["script_raw"].total_words,
                "speakers": current_state["script_raw"].speakers,
            }
        
        if data.get("script_cleaned_path"):
            path_cleaned = Path(data["script_cleaned_path"])
            current_state["script_cleaned"] = extract_script_from_file(path_cleaned)
            response["script_cleaned"] = {
                "filename": path_cleaned.name,
                "total_turns": current_state["script_cleaned"].total_turns,
                "total_words": current_state["script_cleaned"].total_words,
                "speakers": current_state["script_cleaned"].speakers,
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


@app.route("/api/compare-scripts", methods=["POST"])
def compare_scripts():
    """Run comparison on loaded script transcripts (raw vs cleaned)."""
    if not current_state["script_raw"] or not current_state["script_cleaned"]:
        return jsonify({
            "success": False,
            "error": "Please load both raw and cleaned script transcripts first"
        })
    
    try:
        raw_script: ExtractedScript = current_state["script_raw"]
        cleaned_script: ExtractedScript = current_state["script_cleaned"]
        
        # Perform word-level diff on the full transcripts
        diff_result = compute_diff(raw_script.all_words, cleaned_script.all_words)
        
        # Generate aligned HTML
        html_raw, html_cleaned = generate_aligned_html(diff_result)
        
        # Perform detailed analysis
        analysis_raw = analyze_transcript(raw_script.all_words)
        analysis_cleaned = analyze_transcript(cleaned_script.all_words)
        analysis_comparison = compare_analyses(analysis_raw, analysis_cleaned)
        similar_pairs = find_similar_word_pairs(diff_result)
        
        # Build turn-by-turn comparison
        turn_comparisons = _compare_script_turns(raw_script, cleaned_script)
        
        # Calculate LLM improvement metrics
        llm_metrics = _calculate_llm_metrics(raw_script, cleaned_script, diff_result, analysis_raw, analysis_cleaned)
        
        # Build timeline data for audio synchronization
        # Each entry has start time and optional end time (end of turn = start of next turn)
        turn_timeline = []
        for i, turn in enumerate(raw_script.turns):
            next_timestamp = raw_script.turns[i + 1].timestamp if i + 1 < len(raw_script.turns) else None
            turn_timeline.append({
                "index": i,
                "start": turn.timestamp,
                "end": next_timestamp,  # Will be None for last turn
                "speaker": turn.speaker,
            })
        
        response = {
            "success": True,
            "html_raw": html_raw,
            "html_cleaned": html_cleaned,
            "statistics": {
                "total_words_raw": diff_result.total_words_a,
                "total_words_cleaned": diff_result.total_words_b,
                "total_turns_raw": raw_script.total_turns,
                "total_turns_cleaned": cleaned_script.total_turns,
                "matching_words": diff_result.matching_words,
                "inserted_words": diff_result.inserted_words,
                "deleted_words": diff_result.deleted_words,
                "replaced_words_raw": diff_result.replaced_words_a,
                "replaced_words_cleaned": diff_result.replaced_words_b,
                "similarity_ratio": round(diff_result.similarity_ratio * 100, 2),
                "word_error_rate": round(diff_result.word_error_rate * 100, 2),
                "speakers_raw": raw_script.speakers,
                "speakers_cleaned": cleaned_script.speakers,
            },
            "turn_comparisons": turn_comparisons,
            "turn_timeline": turn_timeline,  # For audio sync
            "llm_metrics": llm_metrics,
            "common_substitutions": [
                {"word_a": s[0], "word_b": s[1], "count": s[2]}
                for s in diff_result.common_substitutions[:15]
            ],
            "unique_to_raw": [
                {"word": w[0], "count": w[1]}
                for w in diff_result.unique_to_a[:15]
            ],
            "unique_to_cleaned": [
                {"word": w[0], "count": w[1]}
                for w in diff_result.unique_to_b[:15]
            ],
            "analysis_raw": serialize_analysis(analysis_raw),
            "analysis_cleaned": serialize_analysis(analysis_cleaned),
            "analysis_comparison": analysis_comparison,
            "similar_word_pairs": similar_pairs,
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


def _compare_script_turns(raw: ExtractedScript, cleaned: ExtractedScript) -> list[dict]:
    """Compare turns between raw and cleaned scripts by matching timestamps."""
    comparisons = []
    
    # Build a map of cleaned turns by timestamp for quick lookup
    cleaned_by_time = {}
    for turn in cleaned.turns:
        key = (turn.speaker, round(turn.timestamp, 2))
        cleaned_by_time[key] = turn
    
    for raw_turn in raw.turns:
        key = (raw_turn.speaker, round(raw_turn.timestamp, 2))
        cleaned_turn = cleaned_by_time.get(key)
        
        comparison = {
            "speaker": raw_turn.speaker,
            "timestamp": raw_turn.timestamp,
            "raw_text": raw_turn.text,
            "cleaned_text": cleaned_turn.text if cleaned_turn else None,
            "has_match": cleaned_turn is not None,
        }
        
        # Calculate turn-level diff if we have a match
        if cleaned_turn:
            turn_diff = compute_diff(raw_turn.words, cleaned_turn.words)
            comparison["turn_similarity"] = round(turn_diff.similarity_ratio * 100, 2)
            comparison["words_changed"] = (
                turn_diff.inserted_words + 
                turn_diff.deleted_words + 
                turn_diff.replaced_words_a
            )
            # Generate HTML diff for this turn
            html_raw, html_cleaned = generate_aligned_html(turn_diff)
            comparison["html_raw"] = html_raw
            comparison["html_cleaned"] = html_cleaned
        
        comparisons.append(comparison)
    
    return comparisons


def _calculate_llm_metrics(
    raw: ExtractedScript, 
    cleaned: ExtractedScript, 
    diff: DiffResult,
    analysis_raw,
    analysis_cleaned
) -> dict:
    """Calculate metrics showing how the LLM improved the transcript."""
    
    # Calculate filler word reduction
    filler_raw = analysis_raw.total_filler_count
    filler_cleaned = analysis_cleaned.total_filler_count
    filler_reduction = filler_raw - filler_cleaned if filler_raw > 0 else 0
    filler_reduction_pct = round((filler_reduction / filler_raw * 100) if filler_raw > 0 else 0, 1)
    
    # Calculate repetition reduction (stutters)
    stutters_raw = len(analysis_raw.consecutive_repetitions)
    stutters_cleaned = len(analysis_cleaned.consecutive_repetitions)
    stutter_reduction = stutters_raw - stutters_cleaned
    
    # Word consolidation (how many raw turns were merged)
    turn_consolidation = raw.total_turns - cleaned.total_turns
    
    # Calculate readability improvement (rough metric based on changes)
    words_modified = diff.inserted_words + diff.deleted_words + diff.replaced_words_a
    modification_rate = round((words_modified / diff.total_words_a * 100) if diff.total_words_a > 0 else 0, 1)
    
    # Grammar/capitalization improvements (words that differ only in case)
    case_changes = 0
    for seg in diff.segments:
        if seg.diff_type == DiffType.REPLACE:
            for wa, wb in zip(seg.words_a, seg.words_b):
                if wa.lower() == wb.lower() and wa != wb:
                    case_changes += 1
    
    return {
        "filler_reduction": {
            "raw_count": filler_raw,
            "cleaned_count": filler_cleaned,
            "reduction": filler_reduction,
            "reduction_percentage": filler_reduction_pct,
        },
        "stutter_reduction": {
            "raw_count": stutters_raw,
            "cleaned_count": stutters_cleaned,
            "reduction": stutter_reduction,
        },
        "turn_consolidation": {
            "raw_turns": raw.total_turns,
            "cleaned_turns": cleaned.total_turns,
            "turns_merged": turn_consolidation,
        },
        "modification_summary": {
            "words_added": diff.inserted_words,
            "words_removed": diff.deleted_words,
            "words_replaced": diff.replaced_words_a,
            "total_modifications": words_modified,
            "modification_rate": modification_rate,
        },
        "formatting_improvements": {
            "case_corrections": case_changes,
        },
        "overall_improvement_score": _calculate_improvement_score(
            filler_reduction_pct, stutter_reduction, modification_rate, diff.similarity_ratio
        ),
    }


def _calculate_improvement_score(filler_pct: float, stutter_reduction: int, mod_rate: float, similarity: float) -> dict:
    """Calculate an overall improvement score."""
    # Higher is better for: filler reduction, stutter reduction
    # Want moderate modification rate (too low = no changes, too high = too many changes)
    # Want high similarity (changes should be targeted, not wholesale rewrites)
    
    # Score components (0-100 each)
    filler_score = min(filler_pct, 100)  # Cap at 100%
    stutter_score = min(stutter_reduction * 10, 50)  # Up to 50 points
    
    # Modification rate: ideal is 5-20%, penalize too low or too high
    if mod_rate < 5:
        mod_score = mod_rate * 10  # 0-50
    elif mod_rate <= 20:
        mod_score = 50  # Ideal range
    else:
        mod_score = max(0, 50 - (mod_rate - 20))  # Penalize high modification
    
    # Similarity should be high (70-95% is good)
    sim_pct = similarity * 100
    if sim_pct >= 70:
        sim_score = min((sim_pct - 70) * 2, 50)
    else:
        sim_score = 0
    
    total = filler_score * 0.25 + stutter_score * 0.25 + mod_score * 0.25 + sim_score * 0.25
    
    # Determine quality label
    if total >= 70:
        label = "Excellent"
        description = "LLM made targeted, meaningful improvements"
    elif total >= 50:
        label = "Good"
        description = "LLM made reasonable improvements"
    elif total >= 30:
        label = "Moderate"
        description = "Some improvements made, review recommended"
    else:
        label = "Needs Review"
        description = "Changes may need manual verification"
    
    return {
        "score": round(total, 1),
        "label": label,
        "description": description,
    }


def run_server(host: str = "127.0.0.1", port: int = 5050, debug: bool = True):
    """Run the Flask development server."""
    print(f"\n🎯 Transcript Comparison Tool")
    print(f"   Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
