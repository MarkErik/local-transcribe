// API communication functions

// Import global state variables from app.js
import {
    filesLoaded,
    scriptFilesLoaded,
    turnComparisons,
    diffSegments,
    currentDiffIndex,
    turnTimeline,
    currentPlayingTurnIndex
} from './app.js';

export async function handleFileSelect(which, file, filesLoadedParam, updateCompareButtonParam) {
    if (!file) return;
    
    const formData = new FormData();
    formData.append(`transcript_${which}`, file);
    const infoEl = document.getElementById(`info-${which}`);
    infoEl.textContent = 'Loading...';
    infoEl.style.color = 'var(--text-secondary)';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success && data[`transcript_${which}`]) {
            const info = data[`transcript_${which}`];
            infoEl.textContent = `✓ ${info.word_count} words (${info.format})`;
            infoEl.style.color = 'var(--success)';
            filesLoadedParam[which] = true;
            updateCompareButtonParam(filesLoaded);
        } else {
            const errorMsg = data.errors?.length ? data.errors.join(', ') : 'Unknown format or failed to parse';
            infoEl.textContent = `✗ ${errorMsg}`;
            infoEl.style.color = 'var(--danger)';
            filesLoadedParam[which] = false;
            updateCompareButtonParam(filesLoaded);
        }
    } catch (error) {
        console.error('Upload error:', error);
        infoEl.textContent = `✗ Upload failed: ${error.message}`;
        infoEl.style.color = 'var(--danger)';
        filesLoadedParam[which] = false;
        updateCompareButtonParam(filesLoaded);
    }
}

export async function handlePathInput(which, path, filesLoadedParam, updateCompareButtonParam) {
    if (!path.trim()) return;
    
    const infoEl = document.getElementById(`info-${which}`);
    infoEl.textContent = 'Loading...';
    infoEl.style.color = 'var(--text-secondary)';
    
    try {
        const response = await fetch('/api/load-local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [`transcript_${which}_path`]: path })
        });
        const data = await response.json();
        
        if (data.success && data[`transcript_${which}`]) {
            const info = data[`transcript_${which}`];
            infoEl.textContent = `✓ ${info.word_count} words (${info.format})`;
            infoEl.style.color = 'var(--success)';
            filesLoadedParam[which] = true;
            updateCompareButtonParam(filesLoaded);
        } else {
            const errorMsg = data.errors?.length ? data.errors.join(', ') : 'File not found or unknown format';
            infoEl.textContent = `✗ ${errorMsg}`;
            infoEl.style.color = 'var(--danger)';
            filesLoadedParam[which] = false;
            updateCompareButtonParam(filesLoaded);
        }
    } catch (error) {
        console.error('Load error:', error);
        infoEl.textContent = `✗ Load failed: ${error.message}`;
        infoEl.style.color = 'var(--danger)';
        filesLoadedParam[which] = false;
        updateCompareButtonParam(filesLoaded);
    }
}

export async function handleAudioSelect(file) {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('audio_file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success && data.audio_file) {
            document.getElementById('info-audio').textContent = `✓ ${data.audio_file}`;
            const audioPlayer = document.getElementById('audio-player');
            const audioSection = document.getElementById('audio-section');
            if (audioPlayer) audioPlayer.src = `/api/audio/${data.audio_file}`;
            if (audioSection) audioSection.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Audio upload error:', error);
    }
}

export async function handleAudioPath(path) {
    if (!path.trim()) return;
    
    try {
        const response = await fetch('/api/load-local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path: path })
        });
        const data = await response.json();
        
        if (data.success && data.audio_file) {
            document.getElementById('info-audio').textContent = `✓ ${data.audio_file}`;
            const audioPlayer = document.getElementById('audio-player');
            const audioSection = document.getElementById('audio-section');
            if (audioPlayer) audioPlayer.src = `/api/audio/${data.audio_file}`;
            if (audioSection) audioSection.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Audio load error:', error);
    }
}

export async function handleScriptFileSelect(which, file, scriptFilesLoadedParam, updateScriptCompareButtonParam) {
    if (!file) return;
    
    const formData = new FormData();
    formData.append(`script_${which}`, file);
    const infoEl = document.getElementById(`info-script-${which}`);
    infoEl.textContent = 'Loading...';
    infoEl.style.color = 'var(--text-secondary)';
    
    try {
        const response = await fetch('/api/upload-script', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success && data[`script_${which}`]) {
            const info = data[`script_${which}`];
            infoEl.textContent = `✓ ${info.total_turns} turns, ${info.total_words} words`;
            infoEl.style.color = 'var(--success)';
            scriptFilesLoadedParam[which] = true;
            updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
        } else {
            const errorMsg = data.errors?.length ? data.errors.join(', ') : 'Unknown format or failed to parse';
            infoEl.textContent = `✗ ${errorMsg}`;
            infoEl.style.color = 'var(--danger)';
            scriptFilesLoadedParam[which] = false;
            updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
        }
    } catch (error) {
        console.error('Upload error:', error);
        infoEl.textContent = `✗ Upload failed: ${error.message}`;
        infoEl.style.color = 'var(--danger)';
        scriptFilesLoadedParam[which] = false;
        updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
    }
}

export async function handleScriptPathInput(which, path, scriptFilesLoadedParam, updateScriptCompareButtonParam) {
    if (!path.trim()) return;
    
    const infoEl = document.getElementById(`info-script-${which}`);
    infoEl.textContent = 'Loading...';
    infoEl.style.color = 'var(--text-secondary)';
    
    try {
        const response = await fetch('/api/load-script-local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [`script_${which}_path`]: path })
        });
        const data = await response.json();
        
        if (data.success && data[`script_${which}`]) {
            const info = data[`script_${which}`];
            infoEl.textContent = `✓ ${info.total_turns} turns, ${info.total_words} words`;
            infoEl.style.color = 'var(--success)';
            scriptFilesLoadedParam[which] = true;
            updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
        } else {
            const errorMsg = data.errors?.length ? data.errors.join(', ') : 'File not found or unknown format';
            infoEl.textContent = `✗ ${errorMsg}`;
            infoEl.style.color = 'var(--danger)';
            scriptFilesLoadedParam[which] = false;
            updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
        }
    } catch (error) {
        console.error('Load error:', error);
        infoEl.textContent = `✗ Load failed: ${error.message}`;
        infoEl.style.color = 'var(--danger)';
        scriptFilesLoadedParam[which] = false;
        updateScriptCompareButtonParam(scriptFilesLoaded, document.getElementById('btn-compare-scripts'));
    }
}

export async function handleScriptAudioSelect(file) {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('audio_file', file);
    
    try {
        const response = await fetch('/api/upload-script', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success && data.audio_file) {
            document.getElementById('info-script-audio').textContent = `✓ ${data.audio_file}`;
            const scriptAudioPlayer = document.getElementById('script-audio-player');
            const scriptAudioSection = document.getElementById('script-audio-section');
            if (scriptAudioPlayer) scriptAudioPlayer.src = `/api/audio/${data.audio_file}`;
            if (scriptAudioSection) scriptAudioSection.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Audio upload error:', error);
    }
}

export async function handleScriptAudioPath(path) {
    if (!path.trim()) return;
    
    try {
        const response = await fetch('/api/load-script-local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path: path })
        });
        const data = await response.json();
        
        if (data.success && data.audio_file) {
            document.getElementById('info-script-audio').textContent = `✓ ${data.audio_file}`;
            const scriptAudioPlayer = document.getElementById('script-audio-player');
            const scriptAudioSection = document.getElementById('script-audio-section');
            if (scriptAudioPlayer) scriptAudioPlayer.src = `/api/audio/${data.audio_file}`;
            if (scriptAudioSection) scriptAudioSection.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Audio load error:', error);
    }
}

export async function runComparison(diffSegmentsParam, currentDiffIndexParam, displayResultsParam) {
    const btnCompare = document.getElementById('btn-compare');
    if (btnCompare) {
        btnCompare.textContent = 'Comparing...';
        btnCompare.disabled = true;
    }
    
    try {
        const response = await fetch('/api/compare', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            displayResultsParam(data, diffSegments, currentDiffIndex);
        } else {
            alert(data.error || 'Comparison failed');
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('Comparison failed: ' + error.message);
    } finally {
        if (btnCompare) {
            btnCompare.textContent = 'Compare Transcripts';
            btnCompare.disabled = false;
        }
    }
}

export async function runScriptComparison(turnTimelineParam, currentPlayingTurnIndexParam, turnComparisonsParam, displayScriptResultsParam) {
    const btnCompareScripts = document.getElementById('btn-compare-scripts');
    if (btnCompareScripts) {
        btnCompareScripts.textContent = 'Comparing...';
        btnCompareScripts.disabled = true;
    }
    
    try {
        const response = await fetch('/api/compare-scripts', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            displayScriptResultsParam(data, turnTimeline, { value: currentPlayingTurnIndex }, turnComparisons);
        } else {
            alert(data.error || 'Comparison failed');
            console.error(data.traceback);
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('Comparison failed: ' + error.message);
    } finally {
        if (btnCompareScripts) {
            btnCompareScripts.textContent = 'Compare Scripts';
            btnCompareScripts.disabled = false;
        }
    }
}