// Diff visualization logic

function navigateDiff(direction) {
    if (diffSegments.length === 0) return;
    
    currentDiffIndex += direction;
    if (currentDiffIndex < 0) currentDiffIndex = diffSegments.length - 1;
    if (currentDiffIndex >= diffSegments.length) currentDiffIndex = 0;
    
    document.getElementById('current-diff').textContent = currentDiffIndex + 1;
    
    // Highlight and scroll to the diff
    highlightCurrentDiff();
}

function highlightCurrentDiff() {
    // Remove previous highlights
    document.querySelectorAll('.current-highlight').forEach(el => {
        el.classList.remove('current-highlight');
    });
    
    const segment = diffSegments[currentDiffIndex];
    if (!segment) return;
    
    // Find and highlight the corresponding spans
    const contentA = document.getElementById('content-a');
    const contentB = document.getElementById('content-b');
    
    // Get all non-equal spans
    const spansA = contentA.querySelectorAll('.replace, .delete');
    const spansB = contentB.querySelectorAll('.replace, .insert');
    
    // Simple approach: highlight by index
    if (spansA[currentDiffIndex]) {
        spansA[currentDiffIndex].classList.add('current-highlight');
        spansA[currentDiffIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (spansB[currentDiffIndex]) {
        spansB[currentDiffIndex].classList.add('current-highlight');
    }
}