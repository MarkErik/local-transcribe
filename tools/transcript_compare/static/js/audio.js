// Audio player functionality

// ============================================================================
// Audio Synchronization Functions
// ============================================================================

// Import global state variables from app.js
import {
    turnTimeline,
    currentPlayingTurnIndex
} from './app.js';

/**
 * Find the turn index that corresponds to a given time in seconds
 */
export function findTurnIndexAtTime(timeSeconds, turnTimeline) {
    if (!turnTimeline || turnTimeline.length === 0) return -1;
    
    for (let i = turnTimeline.length - 1; i >= 0; i--) {
        const turn = turnTimeline[i];
        if (timeSeconds >= turn.start) {
            // Check if we're still within this turn's time range
            if (turn.end === null || timeSeconds < turn.end) {
                return turn.index;
            }
        }
    }
    return -1;
}

/**
 * Highlight the turn that matches the current audio playback time
 */
export function highlightTurnAtTime(timeSeconds, turnTimeline, currentPlayingTurnIndex) {
    const turnIndex = findTurnIndexAtTime(timeSeconds, turnTimeline);
    
    if (turnIndex === currentPlayingTurnIndex.value) return; // No change needed
    
    // Remove highlight from previous turn
    document.querySelectorAll('.turn-card.audio-playing').forEach(card => {
        card.classList.remove('audio-playing', 'pulse');
    });
    
    currentPlayingTurnIndex.value = turnIndex;
    
    if (turnIndex >= 0) {
        const turnCard = document.querySelector(`.turn-card[data-turn-index="${turnIndex}"]`);
        if (turnCard) {
            turnCard.classList.add('audio-playing');
            
            // Add pulse animation when audio is actually playing
            const scriptAudioPlayer = document.getElementById('script-audio-player');
            if (scriptAudioPlayer && !scriptAudioPlayer.paused) {
                turnCard.classList.add('pulse');
            }
            
            // Scroll the turn into view if it's not visible
            const turnList = document.getElementById('turn-list');
            if (turnList) {
                const turnRect = turnCard.getBoundingClientRect();
                const listRect = turnList.getBoundingClientRect();
                
                if (turnRect.top < listRect.top || turnRect.bottom > listRect.bottom) {
                    turnCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }
    }
}

/**
 * Seek audio to a specific timestamp (called when clicking a turn)
 */
export function seekAudioToTurn(timestampSeconds, turnTimeline, currentPlayingTurnIndex) {
    const scriptAudioPlayer = document.getElementById('script-audio-player');
    
    if (!scriptAudioPlayer || !scriptAudioPlayer.src || scriptAudioPlayer.src === window.location.href) {
        // No audio loaded
        return;
    }
    
    // Seek to the timestamp
    scriptAudioPlayer.currentTime = timestampSeconds;
    
    // Highlight the turn immediately
    highlightTurnAtTime(timestampSeconds, turnTimeline, currentPlayingTurnIndex);
    
    // Start playing if not already
    if (scriptAudioPlayer.paused) {
        scriptAudioPlayer.play().catch(err => {
            console.log('Auto-play prevented:', err);
        });
    }
}

/**
 * Initialize audio sync event listeners
 */
export function initializeAudioSync(turnTimeline, currentPlayingTurnIndex) {
    // Get DOM elements
    const scriptAudioPlayer = document.getElementById('script-audio-player');
    
    if (scriptAudioPlayer) {
        // Update highlight during playback
        scriptAudioPlayer.addEventListener('timeupdate', () => {
            highlightTurnAtTime(scriptAudioPlayer.currentTime, turnTimeline, currentPlayingTurnIndex);
        });
        
        // Handle seeking (user scrubbing the timeline)
        scriptAudioPlayer.addEventListener('seeked', () => {
            highlightTurnAtTime(scriptAudioPlayer.currentTime, turnTimeline, currentPlayingTurnIndex);
        });
        
        // Add/remove pulse animation based on play/pause
        scriptAudioPlayer.addEventListener('play', () => {
            const currentCard = document.querySelector('.turn-card.audio-playing');
            if (currentCard) {
                currentCard.classList.add('pulse');
            }
        });
        
        scriptAudioPlayer.addEventListener('pause', () => {
            document.querySelectorAll('.turn-card.pulse').forEach(card => {
                card.classList.remove('pulse');
            });
        });
        
        // Reset highlight when audio ends
        scriptAudioPlayer.addEventListener('ended', () => {
            document.querySelectorAll('.turn-card.audio-playing').forEach(card => {
                card.classList.remove('audio-playing', 'pulse');
            });
            currentPlayingTurnIndex.value = -1;
        });
    }
}

// Initialize speed button event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Speed buttons for audio
    const audioPlayer = document.getElementById('audio-player');
    const speedButtons = document.querySelectorAll('.speed-btn');
    
    speedButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            speedButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (audioPlayer) {
                audioPlayer.playbackRate = parseFloat(btn.dataset.speed);
            }
        });
    });

    // Speed buttons for script audio
    const scriptAudioPlayer = document.getElementById('script-audio-player');
    const scriptSpeedButtons = document.querySelectorAll('.speed-btn-script');
    
    scriptSpeedButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            scriptSpeedButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (scriptAudioPlayer) {
                scriptAudioPlayer.playbackRate = parseFloat(btn.dataset.speed);
            }
        });
    });
});