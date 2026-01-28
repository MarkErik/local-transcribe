/**
 * Transcript Editor Page
 * 
 * Full transcript viewing and editing interface with:
 * - Dual-track audio player (interviewer + participant)
 * - Transcript view with block-level highlighting
 * - Stage selector for viewing different pipeline outputs
 * - Word-level editing with undo/redo
 * - Auto-save functionality
 * - PII highlighting and audit trail
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  getJob, 
  getTranscript, 
  getAvailableStages, 
  getAudioUrl,
  listEdits,
  getPIIReplacements,
} from '../api';
import { 
  DualTrackPlayer, 
  TranscriptView, 
  StageBadges,
  EditToolbar,
  StaleStageWarning,
  PIIHighlightMode,
  PIIAuditTrail,
  RedactionTool,
  ExportDialog,
  PrintView,
  ComparisonView,
  VirtualizedTranscriptView,
  ErrorBoundary,
} from '../components';
import { useEditStore, useDeIdentificationStore } from '../store';

// Threshold for using virtualized list
const VIRTUALIZATION_THRESHOLD = 200;

export function TranscriptEditor() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  // Current playback time for synchronization
  const [currentTime, setCurrentTime] = useState(0);
  
  // Selected stage
  const [selectedStage, setSelectedStage] = useState<string | undefined>();
  
  // Selected turn for editing
  const [selectedTurnId, setSelectedTurnId] = useState<number | null>(null);
  
  // Selected word for PII redaction
  const [selectedWordIndex, setSelectedWordIndex] = useState<number | null>(null);
  const [selectedWordText, setSelectedWordText] = useState<string | null>(null);
  
  // Auto-scroll toggle
  const [autoScroll, setAutoScroll] = useState(true);
  
  // Dismiss stale warning
  const [staleWarningDismissed, setStaleWarningDismissed] = useState(false);
  
  // Panel visibility
  const [showAuditTrail, setShowAuditTrail] = useState(false);
  const [showRedactionTool, setShowRedactionTool] = useState(false);
  
  // Phase 6: Export, Print, Compare dialogs
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [showPrintView, setShowPrintView] = useState(false);
  const [showComparisonView, setShowComparisonView] = useState(false);
  
  // De-identification store
  const { setPIIReplacements, piiHighlightEnabled } = useDeIdentificationStore();
  // Edit store
  const { 
    savedEdits, 
    pendingEdits, 
    setCurrentJob,
    setSavedEdits,
    reset: resetEditStore,
  } = useEditStore();

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
  
  // Fetch PII replacements for the job
  const { data: piiData } = useQuery({
    queryKey: ['piiReplacements', jobId],
    queryFn: () => getPIIReplacements(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  });
  
  // Update store when PII data changes
  useEffect(() => {
    if (piiData) {
      setPIIReplacements(piiData.replacements);
    }
  }, [piiData, setPIIReplacements]);

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
  
  // Initialize edit store when job/stage changes
  useEffect(() => {
    if (jobId && selectedStage) {
      setCurrentJob(jobId, selectedStage);
      // Load existing edits
      listEdits(jobId, selectedStage)
        .then(edits => setSavedEdits(edits))
        .catch(err => console.error('Failed to load edits:', err));
    }
    
    return () => {
      // Clean up on unmount
      resetEditStore();
    };
  }, [jobId, selectedStage, setCurrentJob, setSavedEdits, resetEditStore]);

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
    setStaleWarningDismissed(false);
  }, []);
  
  // Note: Word-level editing handlers will be added when TranscriptView is enhanced
  // to support inline editing. For now, the edit store and backend are ready.
  // Use: const { addPendingEdit } = useEditStore();
  // Then: addPendingEdit({ edit_type, turn_id, ... })
  
  // Handle rerun request
  const handleRerunRequested = useCallback((newJobId: string) => {
    // Invalidate caches and navigate to the new job
    queryClient.invalidateQueries({ queryKey: ['jobs'] });
    navigate(`/jobs/${newJobId}`);
  }, [queryClient, navigate]);
  
  // Check if there are edits for stale warning
  const hasEdits = savedEdits.length > 0 || pendingEdits.length > 0;

  // Handle turn selection
  const handleTurnSelect = useCallback((turnId: number) => {
    setSelectedTurnId(turnId === selectedTurnId ? null : turnId);
  }, [selectedTurnId]);

  // Handle word selection for PII redaction
  const handleWordSelect = useCallback((turnId: number, wordIndex: number, word: string) => {
    setSelectedTurnId(turnId);
    setSelectedWordIndex(wordIndex);
    setSelectedWordText(word);
    setShowRedactionTool(true);
  }, []);

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
            
            {/* PII highlight mode toggle */}
            <PIIHighlightMode />
            
            {/* Audit trail toggle */}
            <button
              onClick={() => setShowAuditTrail(!showAuditTrail)}
              className={`flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md transition-colors ${
                showAuditTrail 
                  ? 'bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300' 
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>Audit Trail</span>
            </button>
            
            {/* Divider */}
            <div className="h-6 w-px bg-gray-300 dark:bg-gray-600" />
            
            {/* Compare button */}
            <button
              onClick={() => setShowComparisonView(true)}
              disabled={stages.length < 2}
              className="flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Compare pipeline stages"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              <span>Compare</span>
            </button>
            
            {/* Print button */}
            <button
              onClick={() => setShowPrintView(true)}
              disabled={!transcript}
              className="flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Print transcript"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
              </svg>
              <span>Print</span>
            </button>
            
            {/* Export button */}
            <button
              onClick={() => setShowExportDialog(true)}
              disabled={!transcript}
              className="flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export transcript"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>Export</span>
            </button>
          </div>
        </div>
      </header>

      {/* Edit toolbar */}
      {jobId && selectedStage && (
        <EditToolbar 
          jobId={jobId} 
          stageName={selectedStage}
          onRerunRequested={handleRerunRequested}
        />
      )}
      
      {/* Stale stage warning */}
      {!staleWarningDismissed && jobId && selectedStage && (
        <StaleStageWarning
          jobId={jobId}
          currentStage={selectedStage}
          hasEdits={hasEdits}
          onRerunRequested={handleRerunRequested}
          onDismiss={() => setStaleWarningDismissed(true)}
        />
      )}

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Audio player */}
        <div className="w-1/3 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto p-4">
          <ErrorBoundary
            fallback={
              <div className="text-center py-12 text-red-500">
                Audio player failed to load. Please refresh.
              </div>
            }
          >
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
          </ErrorBoundary>
          
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
          
          {/* PII Audit Trail Panel */}
          {showAuditTrail && piiHighlightEnabled && jobId && (
            <div className="mt-4">
              <PIIAuditTrail jobId={jobId} />
            </div>
          )}
          
          {/* Redaction Tool Panel */}
          {showRedactionTool && jobId && (
            <div className="mt-4">
              <RedactionTool 
                jobId={jobId}
                selectedTurnId={selectedTurnId ?? undefined}
                selectedWordIndex={selectedWordIndex ?? undefined}
                selectedWordText={selectedWordText ?? undefined}
                onRedactionComplete={() => {
                  setSelectedWordIndex(null);
                  setSelectedWordText(null);
                }}
              />
            </div>
          )}
        </div>

        {/* Right panel: Transcript view */}
        <div className="flex-1 overflow-hidden">
          <ErrorBoundary
            fallback={
              <div className="flex items-center justify-center h-full text-red-500">
                Transcript view failed to load. Please refresh.
              </div>
            }
          >
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
              // Use virtualized view for large transcripts
              transcript.turns.length > VIRTUALIZATION_THRESHOLD ? (
                <VirtualizedTranscriptView
                  turns={transcript.turns}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                  autoScroll={autoScroll}
                  selectedTurnId={selectedTurnId}
                  onTurnSelect={handleTurnSelect}
                  onWordSelect={handleWordSelect}
                />
              ) : (
                <TranscriptView
                  turns={transcript.turns}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                  autoScroll={autoScroll}
                  selectedTurnId={selectedTurnId}
                  onTurnSelect={handleTurnSelect}
                  onWordSelect={handleWordSelect}
                  piiReplacements={piiData?.replacements}
                />
              )
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-500 dark:text-gray-400">
                  No transcript available
                </p>
              </div>
            )}
          </ErrorBoundary>
        </div>
      </div>

      {/* Footer status bar */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-2">
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div>
            {transcript?.turns && (
              <span>
                {transcript.turns.length} turns
                {transcript.turns.length > VIRTUALIZATION_THRESHOLD && (
                  <span className="ml-2 text-green-600">(virtualized)</span>
                )}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-4">
            <span>Stage: {selectedStage || 'None'}</span>
            {isLoadingStages && <span>Loading stages...</span>}
          </div>
        </div>
      </footer>
      
      {/* Export Dialog */}
      {showExportDialog && jobId && (
        <ExportDialog
          jobId={jobId}
          stages={stages}
          currentStage={selectedStage}
          onClose={() => setShowExportDialog(false)}
        />
      )}
      
      {/* Print View */}
      {showPrintView && transcript && jobId && selectedStage && (
        <PrintView
          transcript={transcript}
          jobId={jobId}
          stage={selectedStage}
          onClose={() => setShowPrintView(false)}
        />
      )}
      
      {/* Comparison View */}
      {showComparisonView && jobId && (
        <ComparisonView
          jobId={jobId}
          onClose={() => setShowComparisonView(false)}
        />
      )}
    </div>
  );
}

export default TranscriptEditor;
