// File handling

// Event Listeners
fileA.addEventListener('change', () => handleFileSelect('a', fileA.files[0]));
fileB.addEventListener('change', () => handleFileSelect('b', fileB.files[0]));
fileAudio.addEventListener('change', () => handleAudioSelect(fileAudio.files[0]));
pathA.addEventListener('blur', () => handlePathInput('a', pathA.value));
pathB.addEventListener('blur', () => handlePathInput('b', pathB.value));
pathAudio.addEventListener('blur', () => handleAudioPath(pathAudio.value));

btnCompare.addEventListener('click', runComparison);

// Script Mode Event Listeners
fileScriptRaw.addEventListener('change', () => handleScriptFileSelect('raw', fileScriptRaw.files[0]));
fileScriptCleaned.addEventListener('change', () => handleScriptFileSelect('cleaned', fileScriptCleaned.files[0]));
fileScriptAudio.addEventListener('change', () => handleScriptAudioSelect(fileScriptAudio.files[0]));

pathScriptRaw.addEventListener('blur', () => handleScriptPathInput('raw', pathScriptRaw.value));
pathScriptCleaned.addEventListener('blur', () => handleScriptPathInput('cleaned', pathScriptCleaned.value));
pathScriptAudio.addEventListener('blur', () => handleScriptAudioPath(pathScriptAudio.value));

btnCompareScripts.addEventListener('click', runScriptComparison);

// Diff navigation
document.getElementById('btn-prev-diff').addEventListener('click', () => navigateDiff(-1));
document.getElementById('btn-next-diff').addEventListener('click', () => navigateDiff(1));

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
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.classList.contains('disabled')) return;
        
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        currentMode = btn.dataset.mode;
        
        // Show/hide appropriate sections
        document.getElementById('word-mode-section').classList.toggle('hidden', currentMode !== 'word');
        document.getElementById('script-mode-section').classList.toggle('hidden', currentMode !== 'script');
        resultsSection.classList.add('hidden');
        scriptResultsSection.classList.add('hidden');
    });
});