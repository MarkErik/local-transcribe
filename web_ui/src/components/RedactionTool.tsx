/**
 * Redaction Tool component.
 * 
 * Provides tools for manually redacting text as PII and restoring
 * incorrectly redacted text.
 */

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createPIIRedaction, restorePII, PIIReplacement } from '../api/client';
import { useDeIdentificationStore } from '../store';

export interface RedactionToolProps {
  jobId: string;
  selectedTurnId?: number;
  selectedWordIndex?: number;
  selectedWordText?: string;
  selectedSpeaker?: string;
  onRedactionComplete?: () => void;
  className?: string;
}

export const REDACTION_OPTIONS = [
  { value: '[NAME]', label: 'Name' },
  { value: '[LOCATION]', label: 'Location' },
  { value: '[ORGANIZATION]', label: 'Organization' },
  { value: '[DATE]', label: 'Date' },
  { value: '[PHONE]', label: 'Phone' },
  { value: '[EMAIL]', label: 'Email' },
  { value: '[ADDRESS]', label: 'Address' },
  { value: '[OTHER]', label: 'Other PII' },
];

export function RedactionTool({
  jobId,
  selectedTurnId,
  selectedWordIndex,
  selectedWordText,
  selectedSpeaker,
  onRedactionComplete,
  className = '',
}: RedactionToolProps) {
  const queryClient = useQueryClient();
  const { addPIIReplacement } = useDeIdentificationStore();
  
  const [replacementType, setReplacementType] = useState('[NAME]');
  const [customReplacement, setCustomReplacement] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  
  const redactMutation = useMutation({
    mutationFn: () => {
      if (selectedTurnId === undefined || selectedWordIndex === undefined || !selectedWordText) {
        throw new Error('No word selected');
      }
      
      const replacement = showCustomInput && customReplacement 
        ? `[${customReplacement.toUpperCase()}]`
        : replacementType;
      
      return createPIIRedaction(
        jobId,
        selectedTurnId,
        selectedWordIndex,
        selectedWordText,
        replacement,
        undefined,
        selectedSpeaker
      );
    },
    onSuccess: (replacement) => {
      addPIIReplacement(replacement);
      queryClient.invalidateQueries({ queryKey: ['piiReplacements', jobId] });
      queryClient.invalidateQueries({ queryKey: ['transcript', jobId] });
      onRedactionComplete?.();
    },
  });
  
  const handleRedact = () => {
    redactMutation.mutate();
  };
  
  const isDisabled = selectedTurnId === undefined || selectedWordIndex === undefined;
  
  return (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm ${className}`}>
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-900">Redact as PII</h3>
      </div>
      
      <div className="px-4 py-3 space-y-3">
        {/* Selected word display */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Selected Text</label>
          <div className={`px-3 py-2 bg-gray-50 rounded border ${isDisabled ? 'border-gray-200 text-gray-400' : 'border-gray-300 text-gray-900'}`}>
            {selectedWordText || 'Select a word to redact'}
          </div>
        </div>
        
        {/* Replacement type selector */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Replacement Type</label>
          <div className="grid grid-cols-2 gap-2">
            {REDACTION_OPTIONS.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  setReplacementType(option.value);
                  setShowCustomInput(false);
                }}
                className={`
                  px-3 py-1.5 text-sm rounded border
                  ${replacementType === option.value && !showCustomInput
                    ? 'bg-red-50 border-red-300 text-red-700'
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                  }
                `}
              >
                {option.label}
              </button>
            ))}
            <button
              onClick={() => setShowCustomInput(true)}
              className={`
                px-3 py-1.5 text-sm rounded border
                ${showCustomInput
                  ? 'bg-red-50 border-red-300 text-red-700'
                  : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                }
              `}
            >
              Custom...
            </button>
          </div>
        </div>
        
        {/* Custom replacement input */}
        {showCustomInput && (
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Custom Type</label>
            <input
              type="text"
              value={customReplacement}
              onChange={(e) => setCustomReplacement(e.target.value)}
              placeholder="e.g., MEDICAL"
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-red-500 focus:border-red-500"
            />
          </div>
        )}
        
        {/* Preview */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Preview</label>
          <div className="px-3 py-2 bg-red-50 rounded border border-red-200 text-red-700 font-mono text-sm">
            {showCustomInput && customReplacement 
              ? `[${customReplacement.toUpperCase()}]`
              : replacementType
            }
          </div>
        </div>
        
        {/* Redact button */}
        <button
          onClick={handleRedact}
          disabled={isDisabled || redactMutation.isPending}
          className={`
            w-full px-4 py-2 rounded-md text-sm font-medium
            ${isDisabled
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-red-600 text-white hover:bg-red-700'
            }
            flex items-center justify-center
          `}
        >
          {redactMutation.isPending ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Redacting...
            </>
          ) : (
            <>
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
              </svg>
              Redact Selected Text
            </>
          )}
        </button>
        
        {/* Error display */}
        {redactMutation.isError && (
          <p className="text-sm text-red-600">
            {redactMutation.error?.message || 'Failed to redact'}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * Restore PII component - allows restoring incorrectly redacted text.
 */
interface RestorePIIButtonProps {
  jobId: string;
  replacement: PIIReplacement;
  onRestoreComplete?: () => void;
  className?: string;
}

export function RestorePIIButton({
  jobId,
  replacement,
  onRestoreComplete,
  className = '',
}: RestorePIIButtonProps) {
  const queryClient = useQueryClient();
  const { addPIIReplacement } = useDeIdentificationStore();
  const [showConfirm, setShowConfirm] = useState(false);
  
  const restoreMutation = useMutation({
    mutationFn: () => restorePII(jobId, replacement.id),
    onSuccess: (override) => {
      addPIIReplacement(override);
      queryClient.invalidateQueries({ queryKey: ['piiReplacements', jobId] });
      queryClient.invalidateQueries({ queryKey: ['transcript', jobId] });
      setShowConfirm(false);
      onRestoreComplete?.();
    },
  });
  
  if (showConfirm) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <span className="text-xs text-amber-600">Restore "{replacement.original_text}"?</span>
        <button
          onClick={() => restoreMutation.mutate()}
          disabled={restoreMutation.isPending}
          className="px-2 py-1 bg-amber-500 text-white text-xs rounded hover:bg-amber-600"
        >
          Yes
        </button>
        <button
          onClick={() => setShowConfirm(false)}
          className="px-2 py-1 bg-gray-300 text-gray-700 text-xs rounded hover:bg-gray-400"
        >
          No
        </button>
      </div>
    );
  }
  
  return (
    <button
      onClick={() => setShowConfirm(true)}
      className={`text-amber-600 hover:text-amber-700 text-xs underline ${className}`}
      title="Restore original text (undo redaction)"
    >
      Restore
    </button>
  );
}

export default RedactionTool;
