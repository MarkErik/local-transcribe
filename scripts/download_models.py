# scripts/download_models.py
"""
Bootstrap downloader for local-transcribe.

- Forces ONLINE just for this script (ignores HF_HUB_OFFLINE=1 in .env).
- Caches everything inside ./models/ so runtime can be fully offline later.
- Always downloads:
    * openai/whisper-medium.en
    * openai/whisper-large-v3-turbo
    * WhisperX English aligner
    * pyannote/speaker-diarization-3.1 (pipeline + deps)

Re-run safely any time; it will reuse the cache.
"""

from __future__ import annotations
import os
import sys
import json
import pathlib
from typing import Optional

# --- Force ONLINE for bootstrap only (runtime will be offline) ---------------
os.environ["HF_HUB_OFFLINE"] = "0"
# -----------------------------------------------------------------------------

# Minimal .env loader (no extra deps required)
def load_dotenv_file(path: str | os.PathLike = ".env") -> None:
    path = str(path)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

def ensure_dir(p: str | os.PathLike) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    print(msg, flush=True)

def fail(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)

def set_project_caches(models_root: str) -> None:
    """
    Force all caches into the repo so later runs can be fully offline.
    We use HF_HOME as the primary cache root.
    """
    os.environ.setdefault("HF_HOME", models_root)
    # Avoid TRANSFORMERS_CACHE deprecation warning by not setting it separately.
    os.environ.setdefault("PYANNOTE_CACHE", str(pathlib.Path(models_root, "diarization")))
    os.environ.setdefault("XDG_CACHE_HOME", str(pathlib.Path(models_root, ".xdg")))

def snapshot(repo_id: str, token: str, local_dir_hint: Optional[str] = None) -> str:
    """
    Download a HF repo snapshot into the local cache under ./models/.
    Returns the local cache path.
    """
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "token": token or None,     # allow empty if not needed
        "force_download": False,    # idempotent; resumes if partial
        "local_files_only": False,  # online for bootstrap
        "resume_download": True,
    }
    if local_dir_hint:
        kwargs["cache_dir"] = local_dir_hint

    local_path = snapshot_download(**kwargs)
    return local_path

def download_asr_models(models_root: str, token: str) -> dict:
    """
    Always fetch both ASR models:
      - openai/whisper-medium.en
      - openai/whisper-large-v3-turbo
    """
    asr_root = str(pathlib.Path(models_root, "asr"))
    ensure_dir(asr_root)

    results = {}
    targets = [
        "openai/whisper-medium.en",
        "openai/whisper-large-v3-turbo",
    ]
    for rid in targets:
        log(f"⬇️  Downloading ASR model: {rid}")
        local = snapshot(rid, token, local_dir_hint=asr_root)
        results[rid] = local
        log(f"   ✅ Cached at: {local}")
    return results

def download_ct2_faster_whisper(models_root: str, token: str) -> dict:
    """
    Download CTranslate2 (faster-whisper) models for offline use.
    We try a list of known repos per model and take the first that exists.
    """
    from huggingface_hub.utils import RepositoryNotFoundError
    ct2_root = str(pathlib.Path(models_root, "asr", "ct2"))
    ensure_dir(ct2_root)

    choices = {
        "medium.en": [
            "Systran/faster-whisper-medium.en",
            "guillaumekln/faster-whisper-medium.en",
        ],
        "large-v3-turbo": [
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "h2oai/faster-whisper-large-v3-turbo",
        ],
    }

    results = {}
    for key, repos in choices.items():
        last_err = None
        for rid in repos:
            log(f"⬇️  Downloading CT2 model candidate for {key}: {rid}")
            try:
                local = snapshot(rid, token, local_dir_hint=ct2_root)
                results[key] = {"repo": rid, "path": local}
                log(f"   ✅ Using {rid} at: {local}")
                break
            except RepositoryNotFoundError as e:
                last_err = e
                log(f"   ↪︎ Not found: {rid} (trying next)")
        else:
            raise last_err or RuntimeError(f"Could not fetch any CT2 repo for {key}")
    return results

def download_align_en(models_root: str, token: str) -> str:
    """
    Fetch the WhisperX English alignment model by asking WhisperX to load it.
    WhisperX does NOT accept an hf_token param here; it will use the env/caches.
    """
    align_root = str(pathlib.Path(models_root, "align"))
    ensure_dir(align_root)

    log("⬇️  Downloading WhisperX English alignment model(s)…")
    import torch
    import whisperx

    device = "cpu"  # just to materialize files; no heavy inference here
    # NOTE: do not pass hf_token (API no longer supports it)
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device=device,
        model_dir=align_root,
    )

    # Free memory; files remain cached
    try:
        del align_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    log("   ✅ WhisperX English aligner is cached.")
    return align_root

def download_diarization(models_root: str, token: str) -> dict:
    """
    Download pyannote diarization pipeline (and its dependencies):
      - pyannote/speaker-diarization-3.1

    Newer pyannote versions do not accept `use_auth_token=`; they read the token
    from environment (HUGGINGFACE_HUB_TOKEN/HF_TOKEN) or your HF login cache.
    """
    diar_root = str(pathlib.Path(models_root, "diarization"))
    ensure_dir(diar_root)

    # Mirror our .env token into the env var pyannote/huggingface_hub actually reads.
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)  # some libs still check this

    results = {}
    log("⬇️  Downloading pyannote diarization pipeline (speaker-diarization-3.1)…")
    from pyannote.audio import Pipeline

    # No use_auth_token kwarg; rely on env + HF cache dirs.
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=diar_root,
    )
    # No inference here; we just want to ensure the artifacts are downloaded.
    try:
        _ = pipe
    finally:
        del pipe

    log("   ✅ Pyannote diarization pipeline cached.")
    results["pyannote/speaker-diarization-3.1"] = diar_root
    return results

def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    models_root = str(repo_root / "models")

    load_dotenv_file(repo_root / ".env")

    token = os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        fail(
            "HUGGINGFACE_TOKEN is not set. Edit .env (copied from .env.example) and re-run:\n"
            "  uv run python scripts/download_models.py"
        )

    # Force all caches into ./models
    ensure_dir(models_root)
    set_project_caches(models_root)

    log("📦 Using project-local cache directories:")
    log(f"   HF_HOME={os.environ.get('HF_HOME')}")
    log(f"   PYANNOTE_CACHE={os.environ.get('PYANNOTE_CACHE')}")
    log(f"   XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')}")

    # Download everything (idempotent)
    summary = {
        "asr": download_asr_models(models_root, token),
        "ct2": download_ct2_faster_whisper(models_root, token),
        "align_en": download_align_en(models_root, token),
        "diarization": download_diarization(models_root, token),
    }

    log("\n=== ✅ Model download complete ===")
    log("All models and caches are now inside ./models/.")
    log("You can now run fully offline (HF_HUB_OFFLINE=1).")

    # Write a small manifest for traceability
    manifest_path = pathlib.Path(models_root, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user.", code=130)
