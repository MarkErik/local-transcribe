/**
 * Annotation Menu component
 * 
 * Provides a dropdown menu for inserting common annotations like
 * [laughter], [pause], [inaudible], etc.
 */

import { useState, useRef, useEffect, useCallback } from 'react';

export interface AnnotationMenuProps {
  /** Position to show menu (absolute pixel coordinates) */
  position?: { x: number; y: number };
  /** Whether the menu is visible */
  isOpen: boolean;
  /** Called when an annotation is selected */
  onSelect: (annotationType: string) => void;
  /** Called when menu should close */
  onClose: () => void;
}

const COMMON_ANNOTATIONS = [
  { type: 'laughter', label: '[laughter]', description: 'Speaker laughing' },
  { type: 'pause', label: '[pause]', description: 'Brief pause in speech' },
  { type: 'long_pause', label: '[long pause]', description: 'Extended pause' },
  { type: 'inaudible', label: '[inaudible]', description: 'Cannot understand' },
  { type: 'crosstalk', label: '[crosstalk]', description: 'Multiple speakers' },
  { type: 'sigh', label: '[sigh]', description: 'Audible sigh' },
  { type: 'cough', label: '[cough]', description: 'Speaker coughing' },
  { type: 'throat_clear', label: '[clears throat]', description: 'Throat clearing' },
  { type: 'background_noise', label: '[background noise]', description: 'Environmental noise' },
  { type: 'phone_rings', label: '[phone rings]', description: 'Phone ringing' },
];

export function AnnotationMenu({
  position,
  isOpen,
  onSelect,
  onClose,
}: AnnotationMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [customValue, setCustomValue] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  
  // Close on escape or click outside
  useEffect(() => {
    if (!isOpen) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);
  
  const handleSelect = useCallback((annotationType: string) => {
    onSelect(annotationType);
    onClose();
  }, [onSelect, onClose]);
  
  const handleCustomSubmit = useCallback(() => {
    if (customValue.trim()) {
      onSelect(customValue.trim());
      setCustomValue('');
      setShowCustomInput(false);
      onClose();
    }
  }, [customValue, onSelect, onClose]);
  
  if (!isOpen) return null;
  
  const style = position
    ? { left: position.x, top: position.y }
    : {};
  
  return (
    <div
      ref={menuRef}
      className="absolute z-50 bg-white dark:bg-gray-800 rounded-lg shadow-lg border 
                 border-gray-200 dark:border-gray-700 py-2 min-w-[200px]"
      style={style}
    >
      <div className="px-3 py-1 text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
        Insert Annotation
      </div>
      
      <div className="max-h-64 overflow-y-auto">
        {COMMON_ANNOTATIONS.map((anno) => (
          <button
            key={anno.type}
            className="w-full px-3 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700
                       flex items-center justify-between group"
            onClick={() => handleSelect(anno.type)}
          >
            <span className="font-mono text-sm text-blue-600 dark:text-blue-400">
              {anno.label}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500 
                           opacity-0 group-hover:opacity-100 transition-opacity">
              {anno.description}
            </span>
          </button>
        ))}
      </div>
      
      <div className="border-t border-gray-200 dark:border-gray-700 mt-1 pt-1">
        {showCustomInput ? (
          <div className="px-3 py-2">
            <div className="flex gap-2">
              <input
                type="text"
                value={customValue}
                onChange={(e) => setCustomValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCustomSubmit()}
                placeholder="Custom annotation..."
                className="flex-1 px-2 py-1 text-sm border rounded 
                          dark:bg-gray-700 dark:border-gray-600"
                autoFocus
              />
              <button
                onClick={handleCustomSubmit}
                className="px-2 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Add
              </button>
            </div>
          </div>
        ) : (
          <button
            className="w-full px-3 py-2 text-left text-sm text-gray-600 dark:text-gray-400
                       hover:bg-gray-100 dark:hover:bg-gray-700"
            onClick={() => setShowCustomInput(true)}
          >
            + Custom annotation...
          </button>
        )}
      </div>
    </div>
  );
}

export default AnnotationMenu;
