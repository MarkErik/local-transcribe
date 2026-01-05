// Diff visualization logic

// Import global state variables from app.js
import {
    diffSegments,
    currentDiffIndex
} from './app.js';

export function navigateDiff(direction, diffSegments, currentDiffIndex) {
    if (diffSegments.length === 0) return;
    
    currentDiffIndex.value += direction;
    if (currentDiffIndex.value < 0) currentDiffIndex.value = diffSegments.length - 1;
    if (currentDiffIndex.value >= diffSegments.length) currentDiffIndex.value = 0;
    
    document.getElementById('current-diff').textContent = currentDiffIndex.value + 1;
    
    // Highlight and scroll to the diff
    highlightCurrentDiff(diffSegments, currentDiffIndex);
}

export function highlightCurrentDiff(diffSegments, currentDiffIndex) {
    // Remove previous highlights
    document.querySelectorAll('.current-highlight').forEach(el => {
        el.classList.remove('current-highlight');
    });
    
    const segment = diffSegments[currentDiffIndex.value];
    if (!segment) return;
    
    // Find and highlight the corresponding spans
    const contentA = document.getElementById('content-a');
    const contentB = document.getElementById('content-b');
    
    // Get all non-equal spans
    const spansA = contentA.querySelectorAll('.replace, .delete');
    const spansB = contentB.querySelectorAll('.replace, .insert');
    
    // Simple approach: highlight by index
    if (spansA[currentDiffIndex.value]) {
        spansA[currentDiffIndex.value].classList.add('current-highlight');
        spansA[currentDiffIndex.value].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (spansB[currentDiffIndex.value]) {
        spansB[currentDiffIndex.value].classList.add('current-highlight');
    }
}