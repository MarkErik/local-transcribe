// Audio player functionality

// ============================================================================
// Audio Synchronization Functions
// ============================================================================

/**
 * Find the turn index that corresponds to a given time in seconds
 */
function findTurnIndexAtTime(timeSeconds) {
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
function highlightTurnAtTime(timeSeconds) {
    const turnIndex = findTurnIndexAtTime(timeSeconds);
    
    if (turnIndex === currentPlayingTurnIndex) return; // No change needed
    
    // Remove highlight from previous turn
    document.querySelectorAll('.turn-card.audio-playing').forEach(card => {
        card.classList.remove('audio-playing', 'pulse');
    });
    
    currentPlayingTurnIndex = turnIndex;
    
    if (turnIndex >= 0) {
        const turnCard = document.querySelector(`.turn-card[data-turn-index="${turnIndex}"]`);
        if (turnCard) {
            turnCard.classList.add('audio-playing');
            
            // Add pulse animation when audio is actually playing
            if (!scriptAudioPlayer.paused) {
                turnCard.classList.add('pulse');
            }
            
            // Scroll the turn into view if it's not visible
            const turnList = document.getElementById('turn-list');
            const turnRect = turnCard.getBoundingClientRect();
            const listRect = turnList.getBoundingClientRect();
            
            if (turnRect.top < listRect.top || turnRect.bottom > listRect.bottom) {
                turnCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }
}

/**
 * Seek audio to a specific timestamp (called when clicking a turn)
 */
function seekAudioToTurn(timestampSeconds) {
    if (!scriptAudioPlayer.src || scriptAudioPlayer.src === window.location.href) {
        // No audio loaded
        return;
    }
    
    // Seek to the timestamp
    scriptAudioPlayer.currentTime = timestampSeconds;
    
    // Highlight the turn immediately
    highlightTurnAtTime(timestampSeconds);
    
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
function initializeAudioSync() {
    // Update highlight during playback
    scriptAudioPlayer.addEventListener('timeupdate', () => {
        highlightTurnAtTime(scriptAudioPlayer.currentTime);
    });
    
    // Handle seeking (user scrubbing the timeline)
    scriptAudioPlayer.addEventListener('seeked', () => {
        highlightTurnAtTime(scriptAudioPlayer.currentTime);
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
        currentPlayingTurnIndex = -1;
    });
}

// Speed buttons for audio
document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        audioPlayer.playbackRate = parseFloat(btn.dataset.speed);
    });
});

// Speed buttons for script audio
document.querySelectorAll('.speed-btn-script').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn-script').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        scriptAudioPlayer.playbackRate = parseFloat(btn.dataset.speed);
    });
});