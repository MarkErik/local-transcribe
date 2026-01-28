/**
 * Keyboard Shortcuts component
 * 
 * Provides keyboard shortcut handling and a help modal showing available shortcuts.
 */

import { useEffect } from 'react';

export interface KeyboardShortcutsProps {
  /** Whether shortcuts are enabled */
  enabled?: boolean;
  /** Shortcuts configuration */
  shortcuts?: ShortcutConfig[];
  /** Show the help modal */
  showHelp?: boolean;
  /** Called to toggle help modal */
  onToggleHelp?: () => void;
}

export interface ShortcutConfig {
  /** Unique key for the shortcut */
  key: string;
  /** Human-readable label */
  label: string;
  /** Category for grouping */
  category: 'playback' | 'editing' | 'navigation' | 'general';
  /** Keyboard key combination */
  keys: string[];
  /** Whether Ctrl/Cmd is required */
  ctrl?: boolean;
  /** Whether Shift is required */
  shift?: boolean;
  /** Whether Alt is required */
  alt?: boolean;
  /** Handler function */
  handler: () => void;
}

// Default shortcuts
export const DEFAULT_SHORTCUTS: Omit<ShortcutConfig, 'handler'>[] = [
  // Playback
  { key: 'play-pause', label: 'Play / Pause', category: 'playback', keys: ['Space'] },
  { key: 'seek-back', label: 'Seek back 5s', category: 'playback', keys: ['ArrowLeft'] },
  { key: 'seek-forward', label: 'Seek forward 5s', category: 'playback', keys: ['ArrowRight'] },
  { key: 'seek-back-long', label: 'Seek back 30s', category: 'playback', keys: ['ArrowLeft'], shift: true },
  { key: 'seek-forward-long', label: 'Seek forward 30s', category: 'playback', keys: ['ArrowRight'], shift: true },
  { key: 'speed-down', label: 'Slow down', category: 'playback', keys: ['['] },
  { key: 'speed-up', label: 'Speed up', category: 'playback', keys: [']'] },
  
  // Editing
  { key: 'undo', label: 'Undo', category: 'editing', keys: ['z'], ctrl: true },
  { key: 'redo', label: 'Redo', category: 'editing', keys: ['z'], ctrl: true, shift: true },
  { key: 'redo-alt', label: 'Redo', category: 'editing', keys: ['y'], ctrl: true },
  { key: 'find-replace', label: 'Find & Replace', category: 'editing', keys: ['f'], ctrl: true },
  { key: 'save', label: 'Save', category: 'editing', keys: ['s'], ctrl: true },
  { key: 'delete-word', label: 'Delete word', category: 'editing', keys: ['Delete'] },
  { key: 'edit-word', label: 'Edit selected word', category: 'editing', keys: ['Enter'] },
  
  // Navigation
  { key: 'prev-turn', label: 'Previous turn', category: 'navigation', keys: ['ArrowUp'] },
  { key: 'next-turn', label: 'Next turn', category: 'navigation', keys: ['ArrowDown'] },
  { key: 'prev-word', label: 'Previous word', category: 'navigation', keys: ['ArrowLeft'], alt: true },
  { key: 'next-word', label: 'Next word', category: 'navigation', keys: ['ArrowRight'], alt: true },
  { key: 'jump-start', label: 'Jump to start', category: 'navigation', keys: ['Home'], ctrl: true },
  { key: 'jump-end', label: 'Jump to end', category: 'navigation', keys: ['End'], ctrl: true },
  
  // General
  { key: 'help', label: 'Show shortcuts', category: 'general', keys: ['?'] },
  { key: 'escape', label: 'Cancel / Close', category: 'general', keys: ['Escape'] },
  { key: 'history', label: 'Edit history', category: 'general', keys: ['h'], ctrl: true },
];

function formatShortcut(shortcut: Omit<ShortcutConfig, 'handler'>): string {
  const parts: string[] = [];
  
  if (shortcut.ctrl) {
    parts.push(navigator.platform.includes('Mac') ? '⌘' : 'Ctrl');
  }
  if (shortcut.shift) {
    parts.push('⇧');
  }
  if (shortcut.alt) {
    parts.push(navigator.platform.includes('Mac') ? '⌥' : 'Alt');
  }
  
  // Format key names
  const keyNames: Record<string, string> = {
    'Space': '␣',
    'ArrowLeft': '←',
    'ArrowRight': '→',
    'ArrowUp': '↑',
    'ArrowDown': '↓',
    'Enter': '↵',
    'Escape': 'Esc',
    'Delete': 'Del',
    'Backspace': '⌫',
    'Home': 'Home',
    'End': 'End',
  };
  
  parts.push(...shortcut.keys.map(k => keyNames[k] || k.toUpperCase()));
  
  return parts.join(' + ');
}

const CATEGORY_LABELS: Record<string, string> = {
  playback: '🎵 Playback',
  editing: '✏️ Editing',
  navigation: '🧭 Navigation',
  general: '⚙️ General',
};

interface ShortcutsHelpModalProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts: Omit<ShortcutConfig, 'handler'>[];
}

function ShortcutsHelpModal({ isOpen, onClose, shortcuts }: ShortcutsHelpModalProps) {
  // Group by category
  const grouped = shortcuts.reduce((acc, s) => {
    if (!acc[s.category]) acc[s.category] = [];
    acc[s.category].push(s);
    return acc;
  }, {} as Record<string, typeof shortcuts>);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          <div className="grid grid-cols-2 gap-6">
            {(['playback', 'editing', 'navigation', 'general'] as const).map((category) => (
              <div key={category}>
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                  {CATEGORY_LABELS[category]}
                </h3>
                <div className="space-y-1">
                  {grouped[category]?.map((shortcut) => (
                    <div
                      key={shortcut.key}
                      className="flex items-center justify-between py-1"
                    >
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {shortcut.label}
                      </span>
                      <kbd className="px-2 py-0.5 text-xs font-mono bg-gray-100 dark:bg-gray-700 
                                     rounded border border-gray-300 dark:border-gray-600">
                        {formatShortcut(shortcut)}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Press <kbd className="px-1 py-0.5 text-xs bg-gray-200 dark:bg-gray-700 rounded">?</kbd> at any time to show this help
          </p>
        </div>
      </div>
    </div>
  );
}

export function useKeyboardShortcuts(
  shortcuts: ShortcutConfig[],
  enabled: boolean = true,
) {
  useEffect(() => {
    if (!enabled) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        // Allow escape in inputs
        if (e.key !== 'Escape') return;
      }
      
      for (const shortcut of shortcuts) {
        const keyMatches = shortcut.keys.some(k => 
          k.toLowerCase() === e.key.toLowerCase() || k === e.code
        );
        
        const ctrlMatches = shortcut.ctrl 
          ? (e.ctrlKey || e.metaKey)
          : !(e.ctrlKey || e.metaKey);
        
        const shiftMatches = shortcut.shift ? e.shiftKey : !e.shiftKey;
        const altMatches = shortcut.alt ? e.altKey : !e.altKey;
        
        if (keyMatches && ctrlMatches && shiftMatches && altMatches) {
          e.preventDefault();
          shortcut.handler();
          return;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);
}

export function KeyboardShortcuts({
  enabled = true,
  shortcuts = [],
  showHelp = false,
  onToggleHelp,
}: KeyboardShortcutsProps) {
  // Use the hook to register shortcuts
  useKeyboardShortcuts(shortcuts, enabled);
  
  return (
    <ShortcutsHelpModal
      isOpen={showHelp}
      onClose={() => onToggleHelp?.()}
      shortcuts={shortcuts.map(({ handler, ...rest }) => rest)}
    />
  );
}

// Re-export the help modal for standalone use
export { ShortcutsHelpModal };
export default KeyboardShortcuts;
