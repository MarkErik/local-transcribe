// Utility functions

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add highlight style
const style = document.createElement('style');
style.textContent = `
    .current-highlight {
        outline: 3px solid var(--accent) !important;
        outline-offset: 2px;
    }
`;
document.head.appendChild(style);