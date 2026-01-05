import { escapeHtml } from './utils.js';
import {
  handleFileSelect,
  handlePathInput,
  handleAudioSelect,
  handleAudioPath,
  handleScriptFileSelect,
  handleScriptPathInput,
  handleScriptAudioSelect,
  handleScriptAudioPath,
  runComparison,
  runScriptComparison
} from './api.js';
import {
  updateCompareButton,
  updateScriptCompareButton,
  displayResults,
  displayScriptResults,
  displaySubstitutions,
  displaySimilarPairs,
  displayUniqueWords,
  displayScriptSubstitutions,
  displayScriptSimilarPairs,
  displayScriptUniqueWords,
  displayTurnComparisons,
  filterTurns,
  displayDetailedAnalysis,
  displayRepetitions,
  displayFillers,
  displayFrequency,
  displayPhrases,
  displaySummaryInsights
} from './ui.js';
import {
  findTurnIndexAtTime,
  highlightTurnAtTime,
  seekAudioToTurn,
  initializeAudioSync
} from './audio.js';
import {
  navigateDiff,
  highlightCurrentDiff
} from './diff.js';

// Main application initialization

// State
export const diffSegments = [];
export const currentDiffIndex = { value: -1 };
export const filesLoaded = { a: false, b: false };
export const scriptFilesLoaded = { raw: false, cleaned: false };
let currentMode = 'word';
export const turnComparisons = [];
export const turnTimeline = [];  // For audio sync
export const currentPlayingTurnIndex = { value: -1 };  // Track currently highlighted turn

// DOM Elements - Word Mode
let fileA, fileB, fileAudio, pathA, pathB, pathAudio, btnCompare, resultsSection, audioPlayer, audioSection;

// DOM Elements - Script Mode
let fileScriptRaw, fileScriptCleaned, fileScriptAudio, pathScriptRaw, pathScriptCleaned, pathScriptAudio,
    btnCompareScripts, scriptResultsSection, scriptAudioPlayer, scriptAudioSection;

// Initialize the application
function init() {
    // Get DOM elements
    fileA = document.getElementById('file-a');
    fileB = document.getElementById('file-b');
    fileAudio = document.getElementById('file-audio');
    pathA = document.getElementById('path-a');
    pathB = document.getElementById('path-b');
    pathAudio = document.getElementById('path-audio');
    btnCompare = document.getElementById('btn-compare');
    resultsSection = document.getElementById('results-section');
    audioPlayer = document.getElementById('audio-player');
    audioSection = document.getElementById('audio-section');
    
    // Script Mode Elements
    fileScriptRaw = document.getElementById('file-script-raw');
    fileScriptCleaned = document.getElementById('file-script-cleaned');
    fileScriptAudio = document.getElementById('file-script-audio');
    pathScriptRaw = document.getElementById('path-script-raw');
    pathScriptCleaned = document.getElementById('path-script-cleaned');
    pathScriptAudio = document.getElementById('path-script-audio');
    btnCompareScripts = document.getElementById('btn-compare-scripts');
    scriptResultsSection = document.getElementById('script-results-section');
    scriptAudioPlayer = document.getElementById('script-audio-player');
    scriptAudioSection = document.getElementById('script-audio-section');
    
    // Hide audio sections initially
    if (audioSection) audioSection.classList.add('hidden');
    if (scriptAudioSection) scriptAudioSection.classList.add('hidden');
    
    // Initialize audio sync when DOM is ready
    initializeAudioSync();
}

// Explicit initialization function that initializes all modules
function initializeApp() {
    // Get DOM elements
    fileA = document.getElementById('file-a');
    fileB = document.getElementById('file-b');
    fileAudio = document.getElementById('file-audio');
    pathA = document.getElementById('path-a');
    pathB = document.getElementById('path-b');
    pathAudio = document.getElementById('path-audio');
    btnCompare = document.getElementById('btn-compare');
    resultsSection = document.getElementById('results-section');
    audioPlayer = document.getElementById('audio-player');
    audioSection = document.getElementById('audio-section');
    
    // Script Mode Elements
    fileScriptRaw = document.getElementById('file-script-raw');
    fileScriptCleaned = document.getElementById('file-script-cleaned');
    fileScriptAudio = document.getElementById('file-script-audio');
    pathScriptRaw = document.getElementById('path-script-raw');
    pathScriptCleaned = document.getElementById('path-script-cleaned');
    pathScriptAudio = document.getElementById('path-script-audio');
    btnCompareScripts = document.getElementById('btn-compare-scripts');
    scriptResultsSection = document.getElementById('script-results-section');
    scriptAudioPlayer = document.getElementById('script-audio-player');
    scriptAudioSection = document.getElementById('script-audio-section');
    
    // Hide audio sections initially
    if (audioSection) audioSection.classList.add('hidden');
    if (scriptAudioSection) scriptAudioSection.classList.add('hidden');
    
    // Initialize audio module
    initializeAudioSync(turnTimeline, { value: currentPlayingTurnIndex });
    
    // Additional module initializations can be added here as needed
}

// Run initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);