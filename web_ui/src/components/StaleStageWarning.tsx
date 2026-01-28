/**
 * Stale Stage Warning component
 * 
 * Shows a banner when edits have been made and downstream stages need re-running.
 */

import { useCallback } from 'react';
import { rerunJob } from '../api/client';

interface StaleStageWarningProps {
  jobId: string;
  currentStage: string;
  hasEdits: boolean;
  onRerunRequested?: (newJobId: string) => void;
  onDismiss?: () => void;
}

// Stage order for determining which stages are downstream
const STAGE_ORDER = [
  'vad_transcription',
  'de_identification',
  'speaker_naming',
  'transcript_cleanup',
];

export function StaleStageWarning({
  jobId,
  currentStage,
  hasEdits,
  onRerunRequested,
  onDismiss,
}: StaleStageWarningProps) {
  const handleRerun = useCallback(async () => {
    try {
      // Determine the next stage to start from
      const currentIndex = STAGE_ORDER.indexOf(currentStage);
      const nextStage = currentIndex >= 0 && currentIndex < STAGE_ORDER.length - 1
        ? STAGE_ORDER[currentIndex + 1]
        : undefined;
      
      const result = await rerunJob(jobId, nextStage);
      onRerunRequested?.(result.job_id);
    } catch (err) {
      alert(`Failed to rerun: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, [jobId, currentStage, onRerunRequested]);
  
  if (!hasEdits) {
    return null;
  }
  
  // Determine downstream stages
  const currentIndex = STAGE_ORDER.indexOf(currentStage);
  const downstreamStages = currentIndex >= 0
    ? STAGE_ORDER.slice(currentIndex + 1)
    : [];
  
  const formatStageName = (stage: string) => {
    return stage
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  };
  
  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-3">
      <div className="flex items-start gap-3">
        {/* Warning icon */}
        <div className="flex-shrink-0 mt-0.5">
          <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        
        {/* Message */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-amber-800">
            Edits detected - downstream stages need to be re-run
          </p>
          {downstreamStages.length > 0 && (
            <p className="mt-1 text-sm text-amber-700">
              The following stages will be affected:{' '}
              <span className="font-medium">
                {downstreamStages.map(formatStageName).join(', ')}
              </span>
            </p>
          )}
        </div>
        
        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={handleRerun}
            className="px-3 py-1.5 bg-amber-500 text-white text-sm font-medium rounded 
                       hover:bg-amber-600 transition-colors"
          >
            Re-run Now
          </button>
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="p-1.5 text-amber-600 hover:bg-amber-100 rounded"
              title="Dismiss"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default StaleStageWarning;
