/**
 * Print View Component
 * 
 * Print-friendly transcript view with proper styling for printing.
 * Opens in a new window/tab with print-optimized CSS.
 */

import { useEffect, useRef } from 'react';
import { Transcript, TranscriptTurn } from '../api';

interface PrintViewProps {
  transcript: Transcript;
  jobId: string;
  stage: string;
  title?: string;
  onClose: () => void;
}

export function PrintView({ transcript, jobId, stage, title, onClose }: PrintViewProps) {
  const printRef = useRef<HTMLDivElement>(null);

  // Auto-open print dialog
  useEffect(() => {
    const timer = setTimeout(() => {
      window.print();
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const formatTimestamp = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const formatStageName = (stage: string): string => {
    const names: Record<string, string> = {
      'vad_transcription': 'Raw Transcription',
      'de_identification': 'De-identified',
      'speaker_naming': 'Named Speakers',
      'transcript_cleanup': 'LLM Cleaned',
    };
    return names[stage] || stage;
  };

  const formatDate = (dateStr?: string): string => {
    if (!dateStr) return new Date().toLocaleDateString();
    return new Date(dateStr).toLocaleDateString();
  };

  // Calculate statistics
  const stats = {
    totalTurns: transcript.turns.length,
    totalWords: transcript.turns.reduce((acc, turn) => {
      return acc + (turn.text?.split(/\s+/).length || 0);
    }, 0),
    totalInterjections: transcript.turns.reduce((acc, turn) => {
      return acc + (turn.interjections?.length || 0);
    }, 0),
    duration: transcript.turns.length > 0
      ? Math.max(...transcript.turns.map(t => t.end_time || 0))
      : 0,
    speakers: [...new Set(transcript.turns.map(t => t.primary_speaker))],
  };

  return (
    <div className="fixed inset-0 bg-white z-50 overflow-auto print:relative print:inset-auto">
      {/* Close button (hidden in print) */}
      <div className="fixed top-4 right-4 print:hidden">
        <button
          onClick={onClose}
          className="bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-700 flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          <span>Close</span>
        </button>
      </div>

      {/* Print action button (hidden in print) */}
      <div className="fixed top-4 left-4 print:hidden">
        <button
          onClick={() => window.print()}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
          </svg>
          <span>Print</span>
        </button>
      </div>

      {/* Print content */}
      <div ref={printRef} className="max-w-4xl mx-auto p-8 print:p-0 print:max-w-none">
        {/* Header */}
        <header className="mb-8 pb-4 border-b-2 border-gray-300">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            {title || 'Conversation Transcript'}
          </h1>
          <div className="text-sm text-gray-600 space-y-1">
            <p><span className="font-medium">Job ID:</span> {jobId}</p>
            <p><span className="font-medium">Stage:</span> {formatStageName(stage)}</p>
            <p><span className="font-medium">Exported:</span> {formatDate()}</p>
          </div>
        </header>

        {/* Summary */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-800 mb-3">Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-gray-50 p-3 rounded print:bg-transparent print:border print:border-gray-300">
              <div className="text-gray-500">Duration</div>
              <div className="font-semibold">{formatTimestamp(stats.duration)}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded print:bg-transparent print:border print:border-gray-300">
              <div className="text-gray-500">Turns</div>
              <div className="font-semibold">{stats.totalTurns}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded print:bg-transparent print:border print:border-gray-300">
              <div className="text-gray-500">Words</div>
              <div className="font-semibold">{stats.totalWords.toLocaleString()}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded print:bg-transparent print:border print:border-gray-300">
              <div className="text-gray-500">Speakers</div>
              <div className="font-semibold">{stats.speakers.join(', ')}</div>
            </div>
          </div>
        </section>

        {/* Transcript */}
        <section>
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Transcript</h2>
          <div className="space-y-4">
            {transcript.turns.map((turn, index) => (
              <PrintTurn key={turn.turn_id || index} turn={turn} formatTimestamp={formatTimestamp} />
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-12 pt-4 border-t border-gray-200 text-xs text-gray-500 print:mt-8">
          <p>Generated by Local Transcribe • {formatDate()}</p>
        </footer>
      </div>

      {/* Print-specific styles */}
      <style>{`
        @media print {
          @page {
            margin: 1in;
            size: letter;
          }
          
          body {
            print-color-adjust: exact;
            -webkit-print-color-adjust: exact;
          }
          
          .print\\:hidden {
            display: none !important;
          }
          
          .print\\:relative {
            position: relative !important;
          }
          
          .print\\:inset-auto {
            inset: auto !important;
          }
          
          .print\\:p-0 {
            padding: 0 !important;
          }
          
          .print\\:max-w-none {
            max-width: none !important;
          }
          
          .print\\:bg-transparent {
            background-color: transparent !important;
          }
          
          .print\\:border {
            border-width: 1px !important;
          }
          
          .print\\:border-gray-300 {
            border-color: #d1d5db !important;
          }
          
          .print\\:mt-8 {
            margin-top: 2rem !important;
          }
        }
      `}</style>
    </div>
  );
}

interface PrintTurnProps {
  turn: TranscriptTurn;
  formatTimestamp: (seconds: number) => string;
}

function PrintTurn({ turn, formatTimestamp }: PrintTurnProps) {
  const speakerColors: Record<string, string> = {
    'Interviewer': '#2563eb', // blue
    'Participant': '#16a34a', // green
    'Unknown': '#6b7280', // gray
  };

  const speakerColor = speakerColors[turn.primary_speaker] || speakerColors['Unknown'];

  return (
    <div className="break-inside-avoid">
      {/* Turn header */}
      <div className="flex items-baseline gap-2 mb-1">
        <span 
          className="font-semibold"
          style={{ color: speakerColor }}
        >
          {turn.primary_speaker}
        </span>
        <span className="text-xs text-gray-400">
          [{formatTimestamp(turn.start_time)} - {formatTimestamp(turn.end_time)}]
        </span>
      </div>
      
      {/* Turn text */}
      <p className="text-gray-800 leading-relaxed pl-4 border-l-2" style={{ borderColor: speakerColor }}>
        {turn.text}
      </p>
      
      {/* Interjections */}
      {turn.interjections && turn.interjections.length > 0 && (
        <div className="mt-2 pl-8 space-y-1">
          {turn.interjections.map((interjection, idx) => (
            <div key={idx} className="text-sm text-gray-600 italic">
              <span className="font-medium">[{interjection.speaker}]</span>{' '}
              {interjection.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default PrintView;
