/**
 * Stage selector dropdown for switching between pipeline output stages.
 */

import { useMemo } from 'react';

export interface Stage {
  stage: string;
  file?: string;
  display_name?: string;
  has_edits: boolean;
  source?: string;
}

export interface StageSelectorProps {
  /** Available stages */
  stages: Stage[];
  /** Currently selected stage */
  selectedStage?: string;
  /** Called when stage selection changes */
  onStageChange: (stage: string) => void;
  /** Whether to show loading state */
  isLoading?: boolean;
}

// Display names for stages (supports both old and new stage names)
const STAGE_DISPLAY_NAMES: Record<string, string> = {
  // New stage names
  base: 'Raw Transcription',
  de_identified: 'De-identified',
  cleaned: 'Cleaned',
  // Legacy stage names (for backward compatibility)
  vad_transcription: 'Raw Transcription',
  de_identification: 'De-identified',
  speaker_naming: 'Speaker Named',
  transcript_cleanup: 'Cleaned',
};

// Stage order for sorting (supports both old and new names)
const STAGE_ORDER: Record<string, number> = {
  // New stage names
  base: 1,
  de_identified: 2,
  cleaned: 3,
  // Legacy stage names
  vad_transcription: 1,
  de_identification: 2,
  speaker_naming: 3,
  transcript_cleanup: 4,
};

function getStageDisplayName(stage: Stage): string {
  // Prefer display_name from backend if provided
  if (stage.display_name) {
    return stage.display_name;
  }
  return STAGE_DISPLAY_NAMES[stage.stage] || stage.stage;
}

export function StageSelector({
  stages,
  selectedStage,
  onStageChange,
  isLoading = false,
}: StageSelectorProps) {
  // Sort stages by order
  const sortedStages = useMemo(() => {
    return [...stages].sort((a, b) => {
      const orderA = STAGE_ORDER[a.stage] ?? 99;
      const orderB = STAGE_ORDER[b.stage] ?? 99;
      return orderA - orderB;
    });
  }, [stages]);

  if (stages.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400">
        No stages available
      </div>
    );
  }

  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Stage:
      </label>
      <div className="relative">
        <select
          value={selectedStage || stages[0]?.stage}
          onChange={(e) => onStageChange(e.target.value)}
          disabled={isLoading}
          className="
            appearance-none
            bg-white dark:bg-gray-800
            border border-gray-300 dark:border-gray-600
            text-gray-700 dark:text-gray-300
            rounded-lg
            px-4 py-2 pr-10
            text-sm
            focus:outline-none focus:ring-2 focus:ring-indigo-500
            disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          {sortedStages.map((stage) => (
            <option key={stage.stage} value={stage.stage}>
              {getStageDisplayName(stage)}
              {stage.has_edits ? ' *' : ''}
            </option>
          ))}
        </select>
        
        {/* Dropdown arrow */}
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="animate-spin rounded-full h-4 w-4 border-2 border-indigo-600 border-t-transparent" />
      )}
    </div>
  );
}

/**
 * Stage badges for displaying available stages as pills
 */
export function StageBadges({
  stages,
  selectedStage,
  onStageChange,
}: StageSelectorProps) {
  // Sort stages by order
  const sortedStages = useMemo(() => {
    return [...stages].sort((a, b) => {
      const orderA = STAGE_ORDER[a.stage] ?? 99;
      const orderB = STAGE_ORDER[b.stage] ?? 99;
      return orderA - orderB;
    });
  }, [stages]);

  return (
    <div className="flex flex-wrap gap-2">
      {sortedStages.map((stage) => {
        const isSelected = stage.stage === selectedStage;
        return (
          <button
            key={stage.stage}
            onClick={() => onStageChange(stage.stage)}
            className={`
              px-3 py-1.5 rounded-full text-sm font-medium transition-colors
              ${isSelected
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }
            `}
          >
            {getStageDisplayName(stage)}
            {stage.has_edits && (
              <span className={`ml-1 ${isSelected ? 'text-indigo-200' : 'text-yellow-500'}`}>
                ●
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default StageSelector;
