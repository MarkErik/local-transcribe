/**
 * Transcript Editor Page
 * 
 * Full transcript viewing and editing interface with:
 * - Dual-track audio player (interviewer + participant)
 * - Transcript view with block-level highlighting
 * - Stage selector for viewing different pipeline outputs
 */

import { useState, useCallback, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { 
  getJob, 
  getTranscript, 
  getAvailableStages, 
  getAudioUrl,
} from '../api';
import { DualTrackPlayer, TranscriptView, StageBadges } from '../components';

export function TranscriptEditor() {
  const { jobId } = useParams<{ jobId: string }>();
  
  // Current playback time for synchronization
  const [currentTime, setCurrentTime] = useState(0);
  
  // Selected stage
  const [selectedStage, setSelectedStage] = useState<string | undefined>();
  
  // Selected turn for editing
  const [selectedTurnId, setSelectedTurnId] = useState<number | null>(null);
  
  // Auto-scroll toggle
  const [autoScroll, setAutoScroll] = useState(true);

  // Fetch job details
  const { data: job, isLoading: isLoadingJob, error: jobError } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
  });

  // Fetch available stages
  const { data: stages = [], isLoading: isLoadingStages } = useQuery({
    queryKey: ['stages', jobId],
    queryFn: () => getAvailableStages(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  });

  // Set initial stage when stages are loaded
  useMemo(() => {
    if (stages.length > 0 && !selectedStage) {
      // Default to the most processed stage
      const stageOrder = ['transcript_cleanup', 'speaker_naming', 'de_identification', 'vad_transcription'];
      for (const stageName of stageOrder) {
        const found = stages.find(s => s.stage === stageName);
        if (found) {
          setSelectedStage(found.stage);
          break;
        }
      }
    }
  }, [stages, selectedStage]);

  // Fetch transcript for selected stage
  const { 
    data: transcript, 
    isLoading: isLoadingTranscript,
    error: transcriptError,
  } = useQuery({
    queryKey: ['transcript', jobId, selectedStage],
    queryFn: () => getTranscript(jobId!, selectedStage),
    enabled: !!jobId && !!selectedStage && job?.status === 'completed',
  });

  // Audio URLs
  const audioUrls = useMemo(() => {
    if (!job) return null;
    return {
      interviewer: job.interviewer_file_id ? getAudioUrl(job.interviewer_file_id) : null,
      participant: job.participant_file_id ? getAudioUrl(job.participant_file_id) : null,
    };
  }, [job]);

  // Handle time updates from audio player
  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  // Handle seek from transcript view
  const handleSeek = useCallback((time: number) => {
    // This would be handled by DualTrackPlayer via ref, but for now
    // we rely on the player's internal state
    setCurrentTime(time);
  }, []);

  // Handle stage change
  const handleStageChange = useCallback((stage: string) => {
    setSelectedStage(stage);
  }, []);

  // Handle turn selection
  const handleTurnSelect = useCallback((turnId: number) => {
    setSelectedTurnId(turnId === selectedTurnId ? null : turnId);
  }, [selectedTurnId]);

  // Loading state
  if (isLoadingJob) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading job...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (jobError) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-4">
            Failed to load job: {jobError instanceof Error ? jobError.message : 'Unknown error'}
          </p>
          <Link to="/" className="text-indigo-600 hover:underline">
            ← Back to Jobs
          </Link>
        </div>
      </div>
    );
  }

  // Job not completed
  if (job && job.status !== 'completed') {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Job is not yet complete (status: {job.status})
          </p>
          <Link to={`/jobs/${jobId}`} className="text-indigo-600 hover:underline">
            ← View Job Progress
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </Link>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
              Transcript Editor
            </h1>
            <span className="text-sm text-gray-500 dark:text-gray-400 font-mono">
              {jobId?.slice(0, 8)}...
            </span>
          </div>
          
          {/* Stage selector */}
          <div className="flex items-center space-x-4">
            {stages.length > 0 && (
              <StageBadges
                stages={stages}
                selectedStage={selectedStage}
                onStageChange={handleStageChange}
                isLoading={isLoadingTranscript}
              />
            )}
            
            {/* Auto-scroll toggle */}
            <label className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-gray-600 dark:text-gray-400">Auto-scroll</span>
            </label>
          </div>
        </div>
      </header>

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Audio player */}
        <div className="w-1/3 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto p-4">
          {audioUrls?.interviewer && audioUrls?.participant ? (
            <DualTrackPlayer
              interviewerUrl={audioUrls.interviewer}
              participantUrl={audioUrls.participant}
              onTimeUpdate={handleTimeUpdate}
              onSeek={handleSeek}
            />
          ) : (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              Audio files not available
            </div>
          )}
          
          {/* Selected turn info */}
          {selectedTurnId !== null && transcript?.turns && (
            <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-2">
                Selected Turn #{selectedTurnId}
              </h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                Click on words to edit, or use the editing controls below.
              </p>
              <button
                onClick={() => setSelectedTurnId(null)}
                className="mt-2 text-xs text-yellow-600 dark:text-yellow-400 hover:underline"
              >
                Clear selection
              </button>
            </div>
          )}
        </div>

        {/* Right panel: Transcript view */}
        <div className="flex-1 overflow-hidden">
          {isLoadingTranscript ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto" />
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  Loading transcript...
                </p>
              </div>
            </div>
          ) : transcriptError ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-red-600 dark:text-red-400">
                Failed to load transcript: {transcriptError instanceof Error ? transcriptError.message : 'Unknown error'}
              </p>
            </div>
          ) : transcript?.turns ? (
            <TranscriptView
              turns={transcript.turns}
              currentTime={currentTime}
              onSeek={handleSeek}
              autoScroll={autoScroll}
              selectedTurnId={selectedTurnId}
              onTurnSelect={handleTurnSelect}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500 dark:text-gray-400">
                No transcript available
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer status bar */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-2">
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div>
            {transcript?.turns && (
              <span>{transcript.turns.length} turns</span>
            )}
          </div>
          <div className="flex items-center space-x-4">
            <span>Stage: {selectedStage || 'None'}</span>
            {isLoadingStages && <span>Loading stages...</span>}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default TranscriptEditor;
