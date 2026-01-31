import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { uploadFile, createJob, UploadCompleteResponse } from '../api';
import { useUploadStore } from '../store';

interface FileUploadProps {
  label: string;
  id: string;
  onComplete: (result: UploadCompleteResponse) => void;
}

function FileUpload({ label, id, onComplete }: FileUploadProps) {
  const { uploads, setUpload, clearUpload } = useUploadStore();
  const upload = uploads[id];
  
  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setUpload(id, {
      filename: file.name,
      progress: 0,
      status: 'uploading',
    });
    
    try {
      const result = await uploadFile(file, (progress) => {
        setUpload(id, { progress });
      });
      
      setUpload(id, {
        status: 'complete',
        progress: 100,
        fileId: result.file_id,
      });
      
      onComplete(result);
    } catch (error) {
      setUpload(id, {
        status: 'error',
        error: error instanceof Error ? error.message : 'Upload failed',
      });
    }
  }, [id, setUpload, onComplete]);
  
  const handleClear = () => {
    clearUpload(id);
  };
  
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        {label}
      </label>
      
      {!upload || upload.status === 'error' ? (
        <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            className="hidden"
            id={`file-${id}`}
          />
          <label
            htmlFor={`file-${id}`}
            className="cursor-pointer text-blue-600 dark:text-blue-400 hover:underline"
          >
            Select audio file
          </label>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            Supports M4A, MP3, WAV, and other audio formats
          </p>
          {upload?.error && (
            <p className="text-sm text-red-600 dark:text-red-400 mt-2">
              Error: {upload.error}
            </p>
          )}
        </div>
      ) : (
        <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
              {upload.filename}
            </span>
            {upload.status === 'complete' && (
              <button
                onClick={handleClear}
                className="text-sm text-red-600 dark:text-red-400 hover:underline"
              >
                Remove
              </button>
            )}
          </div>
          
          {upload.status === 'uploading' && (
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${upload.progress}%` }}
              />
            </div>
          )}
          
          {upload.status === 'complete' && (
            <p className="text-sm text-green-600 dark:text-green-400">
              ✓ Upload complete
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export function NewJob() {
  const navigate = useNavigate();
  const [interviewerFileId, setInterviewerFileId] = useState<string | null>(null);
  const [participantFileId, setParticipantFileId] = useState<string | null>(null);
  const [options, setOptions] = useState({
    enableDeIdentification: true,
    enableCleanup: false,
  });
  
  const createJobMutation = useMutation({
    mutationFn: createJob,
    onSuccess: (data) => {
      navigate(`/jobs/${data.job_id}`);
    },
  });
  
  const canSubmit = interviewerFileId && participantFileId && !createJobMutation.isPending;
  
  const handleSubmit = () => {
    if (!interviewerFileId || !participantFileId) return;
    
    createJobMutation.mutate({
      interviewer_file_id: interviewerFileId,
      participant_file_id: participantFileId,
      mode: 'vad_split_audio',
      options: {
        enable_de_identification: options.enableDeIdentification,
        enable_cleanup: options.enableCleanup,
        output_formats: ['turns-json', 'timestamped-txt'],
      },
    });
  };
  
  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        New Transcription Job
      </h1>
      
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Audio Files
        </h2>
        
        <FileUpload
          label="Interviewer Audio"
          id="interviewer"
          onComplete={(result) => setInterviewerFileId(result.file_id)}
        />
        
        <FileUpload
          label="Participant Audio"
          id="participant"
          onComplete={(result) => setParticipantFileId(result.file_id)}
        />
        
        <hr className="my-6 border-gray-200 dark:border-gray-700" />
        
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Options
        </h2>
        
        <div className="space-y-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.enableDeIdentification}
              onChange={(e) => setOptions({ ...options, enableDeIdentification: e.target.checked })}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Enable LLM-based de-identification
            </span>
          </label>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.enableCleanup}
              onChange={(e) => setOptions({ ...options, enableCleanup: e.target.checked })}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Enable LLM-based transcript cleanup
            </span>
          </label>
        </div>
        
        <hr className="my-6 border-gray-200 dark:border-gray-700" />
        
        {createJobMutation.error && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded text-red-700 dark:text-red-300 text-sm">
            {createJobMutation.error instanceof Error
              ? createJobMutation.error.message
              : 'Failed to create job'}
          </div>
        )}
        
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
            canSubmit
              ? 'bg-blue-600 hover:bg-blue-700'
              : 'bg-gray-400 cursor-not-allowed'
          }`}
        >
          {createJobMutation.isPending ? 'Creating...' : 'Start Transcription'}
        </button>
      </div>
    </div>
  );
}
