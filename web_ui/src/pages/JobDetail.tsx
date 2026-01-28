import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getJob, getTranscript, subscribeToJobProgress, Transcript } from '../api';
import { useJobProgressStore } from '../store';

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
                {turn.start_time.toFixed(1)}s - {turn.end_time.toFixed(1)}s
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
  
  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'pending' ? 2000 : false;
    },
  });
  
  const { data: transcript, isLoading: transcriptLoading } = useQuery({
    queryKey: ['transcript', jobId],
    queryFn: () => getTranscript(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  });
  
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
              <dt className="text-gray-500 dark:text-gray-400">Duration</dt>
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
