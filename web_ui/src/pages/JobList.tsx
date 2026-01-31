import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { listJobs, deleteJob, Job } from '../api';

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    completed: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
    cancelled: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  };

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status] || styles.pending}`}>
      {status}
    </span>
  );
}

function formatDuration(seconds?: number): string {
  if (!seconds) return '-';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${minutes}m ${secs}s`;
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '-';
  const date = new Date(dateStr);
  return date.toLocaleString();
}

function JobCard({ job, onDelete }: { job: Job; onDelete: (id: string, force?: boolean) => void }) {
  const isFinished = ['completed', 'failed', 'cancelled'].includes(job.status);
  const isStale = ['pending', 'running'].includes(job.status);
  
  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    if (isStale) {
      // For stale running/pending jobs, offer force delete
      if (window.confirm(
        `Job ${job.id.slice(0, 8)}... appears to be stuck in "${job.status}" state.\n\n` +
        `This can happen if the server was restarted while the job was running.\n\n` +
        `Do you want to force delete this job? This cannot be undone.`
      )) {
        onDelete(job.id, true);
      }
    } else {
      if (window.confirm(`Are you sure you want to delete job ${job.id.slice(0, 8)}...? This cannot be undone.`)) {
        onDelete(job.id, false);
      }
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div>
          <Link
            to={`/jobs/${job.id}`}
            className="text-lg font-medium text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400"
          >
            Job {job.id.slice(0, 8)}...
          </Link>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Mode: {job.mode}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.status} />
          {(isFinished || isStale) && (
            <button
              onClick={handleDelete}
              className={`p-1 transition-colors ${isStale 
                ? 'text-orange-400 hover:text-orange-600 dark:hover:text-orange-400' 
                : 'text-gray-400 hover:text-red-600 dark:hover:text-red-400'}`}
              title={isStale ? "Force delete stale job" : "Delete job"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          )}
        </div>
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500 dark:text-gray-400">Created:</span>
          <p className="text-gray-900 dark:text-white">{formatDate(job.created_at)}</p>
        </div>
        <div>
          <span className="text-gray-500 dark:text-gray-400">Processing Time:</span>
          <p className="text-gray-900 dark:text-white">{formatDuration(job.duration_seconds)}</p>
        </div>
      </div>
      
      {job.error_message && (
        <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded text-sm text-red-700 dark:text-red-300">
          {job.error_message}
        </div>
      )}
      
      {job.status === 'completed' && (
        <div className="mt-4">
          <Link
            to={`/jobs/${job.id}/edit`}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            View Transcript →
          </Link>
        </div>
      )}
    </div>
  );
}

export function JobList() {
  const queryClient = useQueryClient();
  
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => listJobs(),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
  });
  
  const deleteMutation = useMutation({
    mutationFn: ({ jobId, force }: { jobId: string; force?: boolean }) => deleteJob(jobId, force),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
  
  const handleDelete = (jobId: string, force?: boolean) => {
    deleteMutation.mutate({ jobId, force });
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <h3 className="text-red-800 dark:text-red-200 font-medium">Error loading jobs</h3>
        <p className="text-red-600 dark:text-red-300 text-sm mt-1">
          {error instanceof Error ? error.message : 'Unknown error'}
        </p>
        <button
          onClick={() => refetch()}
          className="mt-3 text-sm text-red-700 dark:text-red-300 hover:underline"
        >
          Try again
        </button>
      </div>
    );
  }

  const jobs = data?.jobs || [];

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Jobs</h1>
        <Link
          to="/new"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          New Job
        </Link>
      </div>
      
      {deleteMutation.isError && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-700 dark:text-red-300">
          Failed to delete job: {deleteMutation.error instanceof Error ? deleteMutation.error.message : 'Unknown error'}
        </div>
      )}

      {jobs.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">No jobs yet.</p>
          <Link
            to="/new"
            className="mt-4 inline-block text-blue-600 dark:text-blue-400 hover:underline"
          >
            Create your first job
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
