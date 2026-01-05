// File handling

// Import global state variables from app.js
import {
    filesLoaded,
    scriptFilesLoaded,
    diffSegments,
    currentDiffIndex,
    turnTimeline,
    currentPlayingTurnIndex,
    turnComparisons
} from './app.js';

// Initialize event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const fileA = document.getElementById('file-a');
    const fileB = document.getElementById('file-b');
    const fileAudio = document.getElementById('file-audio');
    const pathA = document.getElementById('path-a');
    const pathB = document.getElementById('path-b');
    const pathAudio = document.getElementById('path-audio');
    const btnCompare = document.getElementById('btn-compare');
    
    // Script Mode Elements
    const fileScriptRaw = document.getElementById('file-script-raw');
    const fileScriptCleaned = document.getElementById('file-script-cleaned');
    const fileScriptAudio = document.getElementById('file-script-audio');
    const pathScriptRaw = document.getElementById('path-script-raw');
    const pathScriptCleaned = document.getElementById('path-script-cleaned');
    const pathScriptAudio = document.getElementById('path-script-audio');

    // Global state variables are imported from app.js

    // Event Listeners
    if (fileA) fileA.addEventListener('change', () => handleFileSelect('a', fileA.files[0], filesLoaded, updateCompareButton));
    if (fileB) fileB.addEventListener('change', () => handleFileSelect('b', fileB.files[0], filesLoaded, updateCompareButton));
    if (fileAudio) fileAudio.addEventListener('change', () => handleAudioSelect(fileAudio.files[0]));
    if (pathA) pathA.addEventListener('blur', () => handlePathInput('a', pathA.value, filesLoaded, updateCompareButton));
    if (pathB) pathB.addEventListener('blur', () => handlePathInput('b', pathB.value, filesLoaded, updateCompareButton));
    if (pathAudio) pathAudio.addEventListener('blur', () => handleAudioPath(pathAudio.value));

    if (btnCompare) btnCompare.addEventListener('click', () => runComparison(diffSegments, currentDiffIndex, displayResults));

    // Script Mode Event Listeners
    if (fileScriptRaw) fileScriptRaw.addEventListener('change', () => handleScriptFileSelect('raw', fileScriptRaw.files[0], scriptFilesLoaded, updateScriptCompareButton));
    if (fileScriptCleaned) fileScriptCleaned.addEventListener('change', () => handleScriptFileSelect('cleaned', fileScriptCleaned.files[0], scriptFilesLoaded, updateScriptCompareButton));
    if (fileScriptAudio) fileScriptAudio.addEventListener('change', () => handleScriptAudioSelect(fileScriptAudio.files[0]));

    if (pathScriptRaw) pathScriptRaw.addEventListener('blur', () => handleScriptPathInput('raw', pathScriptRaw.value, scriptFilesLoaded, updateScriptCompareButton));
    if (pathScriptCleaned) pathScriptCleaned.addEventListener('blur', () => handleScriptPathInput('cleaned', pathScriptCleaned.value, scriptFilesLoaded, updateScriptCompareButton));
    if (pathScriptAudio) pathScriptAudio.addEventListener('blur', () => handleScriptAudioPath(pathScriptAudio.value));
    // Additional event listeners that need to be set up after DOM is loaded
    const btnCompareScripts = document.getElementById('btn-compare-scripts');
    if (btnCompareScripts) btnCompareScripts.addEventListener('click', () => runScriptComparison(turnTimeline, currentPlayingTurnIndex, turnComparisons, displayScriptResults));

    // Diff navigation
    const btnPrevDiff = document.getElementById('btn-prev-diff');
    const btnNextDiff = document.getElementById('btn-next-diff');
    if (btnPrevDiff) btnPrevDiff.addEventListener('click', () => navigateDiff(-1, diffSegments, currentDiffIndex));
    if (btnNextDiff) btnNextDiff.addEventListener('click', () => navigateDiff(1, diffSegments, currentDiffIndex));

    // Tab handling for detailed analysis
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active tab button
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show corresponding content
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });

    // Turn filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterTurns(btn.dataset.filter);
        });
    });

    // Mode switching
    let currentMode = 'word';
    const resultsSection = document.getElementById('results-section');
    const scriptResultsSection = document.getElementById('script-results-section');

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('disabled')) return;
            
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            currentMode = btn.dataset.mode;
            
            // Show/hide appropriate sections
            const wordModeSection = document.getElementById('word-mode-section');
            const scriptModeSection = document.getElementById('script-mode-section');
            if (wordModeSection) wordModeSection.classList.toggle('hidden', currentMode !== 'word');
            if (scriptModeSection) scriptModeSection.classList.toggle('hidden', currentMode !== 'script');
            if (resultsSection) resultsSection.classList.add('hidden');
            if (scriptResultsSection) scriptResultsSection.classList.add('hidden');
        });
    });
});