// Main application initialization

// State
let diffSegments = [];
let currentDiffIndex = -1;
let filesLoaded = { a: false, b: false };
let scriptFilesLoaded = { raw: false, cleaned: false };
let currentMode = 'word';
let turnComparisons = [];
let turnTimeline = [];  // For audio sync
let currentPlayingTurnIndex = -1;  // Track currently highlighted turn

// DOM Elements - Word Mode
const fileA = document.getElementById('file-a');
const fileB = document.getElementById('file-b');
const fileAudio = document.getElementById('file-audio');
const pathA = document.getElementById('path-a');
const pathB = document.getElementById('path-b');
const pathAudio = document.getElementById('path-audio');
const btnCompare = document.getElementById('btn-compare');
const resultsSection = document.getElementById('results-section');
const audioPlayer = document.getElementById('audio-player');
const audioSection = document.getElementById('audio-section');

// DOM Elements - Script Mode
const fileScriptRaw = document.getElementById('file-script-raw');
const fileScriptCleaned = document.getElementById('file-script-cleaned');
const fileScriptAudio = document.getElementById('file-script-audio');
const pathScriptRaw = document.getElementById('path-script-raw');
const pathScriptCleaned = document.getElementById('path-script-cleaned');
const pathScriptAudio = document.getElementById('path-script-audio');
const btnCompareScripts = document.getElementById('btn-compare-scripts');
const scriptResultsSection = document.getElementById('script-results-section');
const scriptAudioPlayer = document.getElementById('script-audio-player');
const scriptAudioSection = document.getElementById('script-audio-section');

// Initialize the application
function init() {
    // Hide audio sections initially
    audioSection.classList.add('hidden');
    scriptAudioSection.classList.add('hidden');
    
    // Initialize audio sync when DOM is ready
    initializeAudioSync();
}

// Run initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', init);