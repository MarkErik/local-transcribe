/**
 * Comparison View Component
 * 
 * Side-by-side comparison of transcripts from different pipeline stages.
 * Uses diff_engine to highlight differences between versions.
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  getComparisonData, 
  getAvailableStages, 
  ComparisonData,
  DiffSegment,
} from '../api';

interface ComparisonViewProps {
  jobId: string;
  initialStageA?: string;
  initialStageB?: string;
  onClose: () => void;
}

export function ComparisonView({ 
  jobId, 
  initialStageA, 
  initialStageB,
  onClose 
}: ComparisonViewProps) {
  const [stageA, setStageA] = useState<string>(initialStageA || '');
  const [stageB, setStageB] = useState<string>(initialStageB || '');
  const [viewMode, setViewMode] = useState<'side-by-side' | 'unified'>('side-by-side');

  // Fetch available stages
  const { data: stages = [] } = useQuery({
    queryKey: ['stages', jobId],
    queryFn: () => getAvailableStages(jobId),
  });

  // Initialize stages when loaded
  useMemo(() => {
    if (stages.length >= 2 && !stageA && !stageB) {
      // Default: compare first and last stages
      const stageOrder = ['vad_transcription', 'de_identification', 'speaker_naming', 'transcript_cleanup'];
      const availableStages = stages.map(s => s.stage);
      const orderedAvailable = stageOrder.filter(s => availableStages.includes(s));
      
      if (orderedAvailable.length >= 2) {
        setStageA(orderedAvailable[0]);
        setStageB(orderedAvailable[orderedAvailable.length - 1]);
      }
    }
  }, [stages, stageA, stageB]);

  // Fetch comparison data
  const { 
    data: comparison, 
    isLoading, 
    error 
  } = useQuery({
    queryKey: ['comparison', jobId, stageA, stageB],
    queryFn: () => getComparisonData(jobId, stageA, stageB),
    enabled: !!stageA && !!stageB && stageA !== stageB,
  });

  const formatStageName = (stage: string): string => {
    const names: Record<string, string> = {
      'vad_transcription': 'Raw Transcription',
      'de_identification': 'De-identified',
      'speaker_naming': 'Named Speakers',
      'transcript_cleanup': 'LLM Cleaned',
    };
    return names[stage] || stage;
  };

  return (
    <div className="fixed inset-0 bg-white z-50 flex flex-col">
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">Compare Versions</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Stage selectors */}
        <div className="mt-4 flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">From:</label>
            <select
              value={stageA}
              onChange={(e) => setStageA(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select stage...</option>
              {stages.map(s => (
                <option key={s.stage} value={s.stage} disabled={s.stage === stageB}>
                  {formatStageName(s.stage)}
                </option>
              ))}
            </select>
          </div>

          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">To:</label>
            <select
              value={stageB}
              onChange={(e) => setStageB(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select stage...</option>
              {stages.map(s => (
                <option key={s.stage} value={s.stage} disabled={s.stage === stageA}>
                  {formatStageName(s.stage)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1" />

          {/* View mode toggle */}
          <div className="flex items-center border border-gray-300 rounded-md overflow-hidden">
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`px-3 py-1.5 text-sm ${
                viewMode === 'side-by-side'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Side by Side
            </button>
            <button
              onClick={() => setViewMode('unified')}
              className={`px-3 py-1.5 text-sm border-l border-gray-300 ${
                viewMode === 'unified'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Unified
            </button>
          </div>
        </div>

        {/* Stats */}
        {comparison && !comparison.diff.error && (
          <div className="mt-4 flex items-center gap-6 text-sm">
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-green-200" />
              <span className="text-gray-600">
                {comparison.diff.matching_words} matching
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-red-200" />
              <span className="text-gray-600">
                {comparison.diff.deleted_words} deleted
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-blue-200" />
              <span className="text-gray-600">
                {comparison.diff.inserted_words} inserted
              </span>
            </div>
            <div className="text-gray-500">
              Similarity: {(comparison.diff.similarity_ratio * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {!stageA || !stageB ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            Select two stages to compare
          </div>
        ) : stageA === stageB ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            Please select two different stages to compare
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center gap-2 text-gray-500">
              <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Loading comparison...</span>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-red-500">
              Error loading comparison: {(error as Error).message}
            </div>
          </div>
        ) : comparison ? (
          viewMode === 'side-by-side' ? (
            <SideBySideView comparison={comparison} />
          ) : (
            <UnifiedView comparison={comparison} />
          )
        ) : null}
      </div>
    </div>
  );
}

interface ViewProps {
  comparison: ComparisonData;
}

function SideBySideView({ comparison }: ViewProps) {
  return (
    <div className="h-full flex">
      {/* Left panel (Stage A) */}
      <div className="flex-1 overflow-auto border-r border-gray-200">
        <div className="sticky top-0 bg-gray-50 px-4 py-2 border-b border-gray-200 font-medium text-sm text-gray-600">
          {formatStageName(comparison.stage_a)} ({comparison.diff.total_words_a} words)
        </div>
        <div className="p-4 space-y-4">
          {comparison.transcript_a.turns.map((turn, idx) => (
            <TurnDisplay 
              key={turn.turn_id || idx} 
              turn={turn}
            />
          ))}
        </div>
      </div>

      {/* Right panel (Stage B) */}
      <div className="flex-1 overflow-auto">
        <div className="sticky top-0 bg-gray-50 px-4 py-2 border-b border-gray-200 font-medium text-sm text-gray-600">
          {formatStageName(comparison.stage_b)} ({comparison.diff.total_words_b} words)
        </div>
        <div className="p-4 space-y-4">
          {comparison.transcript_b.turns.map((turn, idx) => (
            <TurnDisplay 
              key={turn.turn_id || idx} 
              turn={turn}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function UnifiedView({ comparison }: ViewProps) {
  // Render unified diff with inline changes
  const segments = comparison.diff.segments || [];
  
  return (
    <div className="h-full overflow-auto p-4">
      <div className="max-w-4xl mx-auto space-y-1">
        {segments.map((segment, idx) => (
          <DiffSegmentDisplay key={idx} segment={segment} />
        ))}
      </div>
    </div>
  );
}

interface DiffSegmentDisplayProps {
  segment: DiffSegment;
}

function DiffSegmentDisplay({ segment }: DiffSegmentDisplayProps) {
  const getClassName = () => {
    switch (segment.type) {
      case 'equal':
        return 'text-gray-800';
      case 'delete':
        return 'bg-red-100 text-red-800 line-through';
      case 'insert':
        return 'bg-green-100 text-green-800';
      case 'replace':
        return '';
      default:
        return 'text-gray-800';
    }
  };

  if (segment.type === 'replace') {
    return (
      <span>
        <span className="bg-red-100 text-red-800 line-through">
          {segment.words_a.join(' ')}
        </span>
        {' '}
        <span className="bg-green-100 text-green-800">
          {segment.words_b.join(' ')}
        </span>
        {' '}
      </span>
    );
  }

  const words = segment.type === 'delete' ? segment.words_a : segment.words_b;
  
  return (
    <span className={getClassName()}>
      {words.join(' ')}{' '}
    </span>
  );
}

interface TurnDisplayProps {
  turn: {
    turn_id?: number;
    primary_speaker: string;
    text: string;
    start_time: number;
    end_time: number;
  };
}

function TurnDisplay({ turn }: TurnDisplayProps) {
  const speakerColors: Record<string, string> = {
    'Interviewer': 'text-blue-600',
    'Participant': 'text-green-600',
    'Unknown': 'text-gray-600',
  };

  const speakerClass = speakerColors[turn.primary_speaker] || speakerColors['Unknown'];

  return (
    <div className="pb-3 border-b border-gray-100 last:border-0">
      <div className={`font-medium ${speakerClass}`}>
        {turn.primary_speaker}
      </div>
      <p className="text-gray-800 leading-relaxed">
        {turn.text}
      </p>
    </div>
  );
}

function formatStageName(stage: string): string {
  const names: Record<string, string> = {
    'vad_transcription': 'Raw Transcription',
    'de_identification': 'De-identified',
    'speaker_naming': 'Named Speakers',
    'transcript_cleanup': 'LLM Cleaned',
  };
  return names[stage] || stage;
}

export default ComparisonView;
