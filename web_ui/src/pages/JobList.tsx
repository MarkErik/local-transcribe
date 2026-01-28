import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { listJobs, Job } from '../api';

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

function JobCard({ job }: { job: Job }) {
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
        <StatusBadge status={job.status} />
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500 dark:text-gray-400">Created:</span>
          <p className="text-gray-900 dark:text-white">{formatDate(job.created_at)}</p>
        </div>
        <div>
          <span className="text-gray-500 dark:text-gray-400">Duration:</span>
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
            to={`/jobs/${job.id}`}
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
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => listJobs(),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
  });

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
            <JobCard key={job.id} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}
