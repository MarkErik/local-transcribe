/**
 * Export Dialog Component
 * 
 * Modal dialog for exporting transcripts in various formats.
 * Supports format selection, stage selection, and export options.
 */

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  getExportFormats, 
  exportTranscript, 
  getExportUrl,
  ExportFormat,
  ExportOptions,
  TranscriptStage,
} from '../api';

interface ExportDialogProps {
  jobId: string;
  stages: TranscriptStage[];
  currentStage?: string;
  onClose: () => void;
}

export function ExportDialog({ jobId, stages, currentStage, onClose }: ExportDialogProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('timestamped-txt');
  const [selectedStage, setSelectedStage] = useState<string>(currentStage || stages[0]?.stage || '');
  const [options, setOptions] = useState<ExportOptions>({
    include_timestamps: true,
    include_speaker_labels: true,
    include_interjections: true,
    include_metadata: true,
  });

  // Fetch available formats
  const { data: formats = [], isLoading: formatsLoading } = useQuery({
    queryKey: ['exportFormats', jobId],
    queryFn: () => getExportFormats(jobId),
  });

  // Export mutation
  const exportMutation = useMutation({
    mutationFn: async () => {
      const blob = await exportTranscript(jobId, {
        format: selectedFormat,
        stage: selectedStage,
        options,
      });
      
      // Trigger download
      const format = formats.find(f => f.id === selectedFormat);
      const filename = `transcript-${jobId.slice(0, 8)}-${selectedStage}${format?.extension || '.txt'}`;
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      return blob;
    },
    onSuccess: () => {
      onClose();
    },
  });

  const handleDirectDownload = () => {
    // For simple cases, use direct download URL
    const url = getExportUrl(jobId, selectedFormat, selectedStage);
    window.open(url, '_blank');
    onClose();
  };

  const formatGroups = {
    text: formats.filter(f => ['timestamped-txt', 'plain-txt', 'dialogue-script'].includes(f.id)),
    structured: formats.filter(f => ['turns-json', 'markdown'].includes(f.id)),
    media: formats.filter(f => ['srt'].includes(f.id)),
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto py-4"
      onClick={(e) => {
        // Close when clicking the backdrop (outside the dialog)
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg mx-4 my-auto max-h-[calc(100vh-2rem)] overflow-y-auto">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 sticky top-0 bg-white z-10">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-gray-900">Export Transcript</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-6">
          {/* Stage Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Export Stage
            </label>
            <select
              value={selectedStage}
              onChange={(e) => setSelectedStage(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              {stages.map(stage => (
                <option key={stage.stage} value={stage.stage}>
                  {formatStageName(stage.stage)}
                  {stage.has_edits && ' (edited)'}
                </option>
              ))}
            </select>
          </div>

          {/* Format Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Export Format
            </label>
            {formatsLoading ? (
              <div className="text-sm text-gray-500">Loading formats...</div>
            ) : (
              <div className="space-y-4">
                {/* Text Formats */}
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                    Text Formats
                  </div>
                  <div className="space-y-2">
                    {formatGroups.text.map(format => (
                      <FormatOption
                        key={format.id}
                        format={format}
                        selected={selectedFormat === format.id}
                        onSelect={() => setSelectedFormat(format.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* Structured Formats */}
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                    Structured Formats
                  </div>
                  <div className="space-y-2">
                    {formatGroups.structured.map(format => (
                      <FormatOption
                        key={format.id}
                        format={format}
                        selected={selectedFormat === format.id}
                        onSelect={() => setSelectedFormat(format.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* Media Formats */}
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                    Subtitle Formats
                  </div>
                  <div className="space-y-2">
                    {formatGroups.media.map(format => (
                      <FormatOption
                        key={format.id}
                        format={format}
                        selected={selectedFormat === format.id}
                        onSelect={() => setSelectedFormat(format.id)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Export Options (advanced, collapsible) */}
          <details className="text-sm">
            <summary className="cursor-pointer text-gray-600 hover:text-gray-900">
              Advanced Options
            </summary>
            <div className="mt-3 space-y-2 pl-4">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={options.include_timestamps}
                  onChange={(e) => setOptions({ ...options, include_timestamps: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Include timestamps</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={options.include_speaker_labels}
                  onChange={(e) => setOptions({ ...options, include_speaker_labels: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Include speaker labels</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={options.include_interjections}
                  onChange={(e) => setOptions({ ...options, include_interjections: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Include interjections</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={options.include_metadata}
                  onChange={(e) => setOptions({ ...options, include_metadata: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Include summary/metadata</span>
              </label>
            </div>
          </details>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3 sticky bottom-0 bg-white">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 hover:text-gray-900"
          >
            Cancel
          </button>
          <button
            onClick={handleDirectDownload}
            disabled={!selectedFormat || exportMutation.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {exportMutation.isPending ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Exporting...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <span>Download</span>
              </>
            )}
          </button>
        </div>

        {/* Error display */}
        {exportMutation.isError && (
          <div className="px-6 pb-4">
            <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
              {(exportMutation.error as Error).message}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface FormatOptionProps {
  format: ExportFormat;
  selected: boolean;
  onSelect: () => void;
}

function FormatOption({ format, selected, onSelect }: FormatOptionProps) {
  return (
    <label
      className={`flex items-start p-3 border rounded-md cursor-pointer transition-colors ${
        selected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <input
        type="radio"
        checked={selected}
        onChange={onSelect}
        className="mt-0.5 text-blue-600 focus:ring-blue-500"
      />
      <div className="ml-3">
        <div className="font-medium text-gray-900">{format.name}</div>
        <div className="text-sm text-gray-500">{format.description}</div>
        <div className="text-xs text-gray-400 mt-1">{format.extension}</div>
      </div>
    </label>
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

export default ExportDialog;
