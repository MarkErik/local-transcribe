import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getJob, getTranscript, subscribeToJobProgress, getDeIdentificationStatus, Transcript } from '../api';
import { useJobProgressStore } from '../store';
import { NameListReview } from '../components';

function ProgressBar({ current, total, label }: { current: number; total: number; label: string }) {
  const percent = total > 0 ? (current / total) * 100 : 0;
  
  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
        <span>{label}</span>
        <span>{current} / {total}</span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function JobProgress({ jobId }: { jobId: string }) {
  const { jobs, setJobProgress } = useJobProgressStore();
  const progress = jobs[jobId];
  
  useEffect(() => {
    const unsubscribe = subscribeToJobProgress(
      jobId,
      (event) => {
        switch (event.type) {
          case 'stage_start':
            setJobProgress(jobId, {
              currentStage: event.data.stage as string,
              status: 'running',
            });
            break;
          case 'block_progress':
            setJobProgress(jobId, {
              blockProgress: {
                current: event.data.current as number,
                total: event.data.total as number,
                speaker: event.data.speaker as string,
              },
            });
            break;
          case 'stage_complete':
            setJobProgress(jobId, {
              completedStages: [
                ...(progress?.completedStages || []),
                event.data.stage as string,
              ],
              blockProgress: undefined,
            });
            break;
          case 'job_complete':
            setJobProgress(jobId, {
              status: 'completed',
              currentStage: undefined,
              blockProgress: undefined,
            });
            break;
          case 'job_error':
            setJobProgress(jobId, {
              status: 'failed',
              error: event.data.error as string,
            });
            break;
        }
      },
      (error) => {
        console.error('SSE error:', error);
      }
    );
    
    return unsubscribe;
  }, [jobId, setJobProgress, progress?.completedStages]);
  
  if (!progress) {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
      </div>
    );
  }
  
  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
      <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Progress
      </h2>
      
      {progress.currentStage && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Current stage: <span className="font-medium">{progress.currentStage}</span>
        </p>
      )}
      
      {progress.blockProgress && (
        <ProgressBar
          current={progress.blockProgress.current}
          total={progress.blockProgress.total}
          label={`Transcribing (${progress.blockProgress.speaker})`}
        />
      )}
      
      {progress.completedStages && progress.completedStages.length > 0 && (
        <div className="mt-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Completed stages:</p>
          <div className="flex flex-wrap gap-2">
            {progress.completedStages.map((stage) => (
              <span
                key={stage}
                className="px-2 py-1 text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded"
              >
                ✓ {stage}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {progress.error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded text-red-700 dark:text-red-300 text-sm">
          {progress.error}
        </div>
      )}
    </div>
  );
}

function TranscriptView({ transcript }: { transcript: Transcript }) {
  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6 mt-6">
      <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Transcript
      </h2>
      
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {transcript.turns.map((turn) => (
          <div
            key={turn.turn_id}
            className={`p-4 rounded-lg ${
              turn.primary_speaker === 'Interviewer'
                ? 'bg-sky-50 dark:bg-sky-900/20 border-l-4 border-sky-600'
                : 'bg-pink-50 dark:bg-pink-900/20 border-l-4 border-pink-600'
            }`}
          >
            <div className="flex justify-between items-start mb-2">
              <span className={`text-sm font-medium ${
                turn.primary_speaker === 'Interviewer'
                  ? 'text-sky-900 dark:text-sky-200'
                  : 'text-pink-900 dark:text-pink-200'
              }`}>
                {turn.primary_speaker}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {turn.start.toFixed(1)}s - {turn.end.toFixed(1)}s
              </span>
            </div>
            <p className="text-gray-800 dark:text-gray-200">
              {turn.text}
            </p>
            
            {turn.interjections && turn.interjections.length > 0 && (
              <div className="mt-2 pl-4 border-l-2 border-gray-300 dark:border-gray-600">
                {turn.interjections.map((interjection, idx) => (
                  <p key={idx} className="text-sm text-gray-600 dark:text-gray-400 italic">
                    [{interjection.speaker}]: {interjection.text}
                  </p>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const [showNameReview, setShowNameReview] = useState(false);
  
  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'pending' ? 2000 : false;
    },
  });
  
  // Check de-identification status for completed jobs
  const { data: deIdStatus } = useQuery({
    queryKey: ['deIdStatus', jobId],
    queryFn: () => getDeIdentificationStatus(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  });
  
  const { data: transcript, isLoading: transcriptLoading } = useQuery({
    queryKey: ['transcript', jobId],
    queryFn: () => getTranscript(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  });
  
  // Check if we need name review (first pass done, second pass not)
  const needsNameReview = deIdStatus?.first_pass_complete && !deIdStatus?.second_pass_complete;
  
  if (!jobId) {
    return <div>Job ID is required</div>;
  }
  
  if (jobLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  
  if (!job) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 dark:text-gray-400">Job not found</p>
        <Link to="/" className="mt-4 text-blue-600 dark:text-blue-400 hover:underline">
          Back to jobs
        </Link>
      </div>
    );
  }
  
  // Name review modal
  if (showNameReview && jobId) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
        <NameListReview
          jobId={jobId}
          onComplete={() => setShowNameReview(false)}
          onCancel={() => setShowNameReview(false)}
        />
      </div>
    );
  }
  
  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
            ← Back to jobs
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
            Job {job.id.slice(0, 8)}...
          </h1>
        </div>
        <span className={`px-3 py-1 text-sm font-medium rounded-full ${
          job.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
          job.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
          job.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
          'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
        }`}>
          {job.status}
        </span>
      </div>
      
      {/* Job metadata */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Details
        </h2>
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-gray-500 dark:text-gray-400">Mode</dt>
            <dd className="text-gray-900 dark:text-white">{job.mode}</dd>
          </div>
          <div>
            <dt className="text-gray-500 dark:text-gray-400">Created</dt>
            <dd className="text-gray-900 dark:text-white">
              {job.created_at ? new Date(job.created_at).toLocaleString() : '-'}
            </dd>
          </div>
          {job.duration_seconds && (
            <div>
              <dt className="text-gray-500 dark:text-gray-400">Processing Time</dt>
              <dd className="text-gray-900 dark:text-white">
                {Math.round(job.duration_seconds)}s
              </dd>
            </div>
          )}
        </dl>
        
        {job.error_message && (
          <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded text-red-700 dark:text-red-300 text-sm">
            <strong>Error:</strong> {job.error_message}
          </div>
        )}
      </div>
      
      {/* Progress for running jobs */}
      {(job.status === 'running' || job.status === 'pending') && (
        <JobProgress jobId={jobId} />
      )}
      
      {/* Actions for completed jobs */}
      {job.status === 'completed' && (
        <div className="mb-6 flex space-x-4">
          <Link
            to={`/jobs/${jobId}/edit`}
            className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            Edit Transcript
          </Link>
        </div>
      )}
      
      {/* De-identification review banner */}
      {job.status === 'completed' && needsNameReview && (
        <div className="mb-6 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-purple-600 dark:text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-purple-800 dark:text-purple-200">
                De-identification Review Required
              </h3>
              <p className="mt-1 text-sm text-purple-700 dark:text-purple-300">
                {deIdStatus?.discovered_names_count || 0} names were discovered during de-identification.
                Review and approve the name list to complete the process.
              </p>
              <div className="mt-3">
                <button
                  onClick={() => setShowNameReview(true)}
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Review Names
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* De-identification complete indicator */}
      {job.status === 'completed' && deIdStatus?.second_pass_complete && (
        <div className="mb-6 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
          <div className="flex items-center">
            <svg className="h-5 w-5 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="ml-2 text-sm text-green-800 dark:text-green-200">
              De-identification complete: {deIdStatus?.total_pii_replacements || 0} names redacted
              {deIdStatus?.manual_redactions ? ` (${deIdStatus.manual_redactions} manual)` : ''}
            </span>
          </div>
        </div>
      )}
      
      {/* Transcript preview for completed jobs */}
      {job.status === 'completed' && transcript && (
        <TranscriptView transcript={transcript} />
      )}
      
      {job.status === 'completed' && transcriptLoading && (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
        </div>
      )}
    </div>
  );
}
