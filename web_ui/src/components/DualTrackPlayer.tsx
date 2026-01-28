/**
 * Dual-track audio player for synchronized playback of two speaker tracks.
 * 
 * Used for VAD-split-audio mode where interviewer and participant have
 * separate audio files.
 */

import { useRef, useState, useCallback, useEffect } from 'react';
import { AudioPlayer, AudioPlayerRef } from './AudioPlayer';

export interface DualTrackPlayerProps {
  /** URL to interviewer audio file */
  interviewerUrl: string;
  /** URL to participant audio file */
  participantUrl: string;
  /** Called when playback time updates */
  onTimeUpdate?: (currentTime: number) => void;
  /** Called when seeking to a position */
  onSeek?: (time: number) => void;
  /** Called when audio is ready */
  onReady?: () => void;
}

export interface DualTrackPlayerRef {
  play: () => void;
  pause: () => void;
  seekTo: (time: number) => void;
  getCurrentTime: () => number;
  getDuration: () => number;
  isPlaying: () => boolean;
  setPlaybackRate: (rate: number) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function DualTrackPlayer({
  interviewerUrl,
  participantUrl,
  onTimeUpdate,
  onSeek,
  onReady,
}: DualTrackPlayerProps) {
  const interviewerRef = useRef<AudioPlayerRef>(null);
  const participantRef = useRef<AudioPlayerRef>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [interviewerReady, setInterviewerReady] = useState(false);
  const [participantReady, setParticipantReady] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [syncTime, setSyncTime] = useState<number | undefined>(undefined);

  const isReady = interviewerReady && participantReady;

  // Notify when both tracks are ready
  useEffect(() => {
    if (isReady && onReady) {
      onReady();
    }
  }, [isReady, onReady]);

  // Handle interviewer time updates (master track)
  const handleInterviewerTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
    setSyncTime(time);
    onTimeUpdate?.(time);
  }, [onTimeUpdate]);

  // Handle interviewer ready
  const handleInterviewerReady = useCallback((dur: number) => {
    setInterviewerReady(true);
    setDuration(dur);
  }, []);

  // Handle participant ready
  const handleParticipantReady = useCallback(() => {
    setParticipantReady(true);
  }, []);

  // Play both tracks
  const handlePlay = useCallback(() => {
    interviewerRef.current?.play();
    participantRef.current?.play();
    setIsPlaying(true);
  }, []);

  // Pause both tracks
  const handlePause = useCallback(() => {
    interviewerRef.current?.pause();
    participantRef.current?.pause();
    setIsPlaying(false);
  }, []);

  // Toggle play/pause
  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      handlePause();
    } else {
      handlePlay();
    }
  }, [isPlaying, handlePlay, handlePause]);

  // Seek both tracks
  const handleSeek = useCallback((time: number) => {
    interviewerRef.current?.seekTo(time);
    participantRef.current?.seekTo(time);
    setCurrentTime(time);
    setSyncTime(time);
    onSeek?.(time);
  }, [onSeek]);

  // Seek backward 5 seconds
  const handleSeekBackward = useCallback(() => {
    const newTime = Math.max(0, currentTime - 5);
    handleSeek(newTime);
  }, [currentTime, handleSeek]);

  // Seek forward 5 seconds
  const handleSeekForward = useCallback(() => {
    const newTime = Math.min(duration, currentTime + 5);
    handleSeek(newTime);
  }, [currentTime, duration, handleSeek]);

  // Change playback rate on both tracks
  const handleRateChange = useCallback((rate: number) => {
    interviewerRef.current?.setPlaybackRate(rate);
    participantRef.current?.setPlaybackRate(rate);
    setPlaybackRate(rate);
  }, []);

  // Handle click on progress bar
  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isReady || duration === 0) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newTime = percent * duration;
    handleSeek(newTime);
  }, [isReady, duration, handleSeek]);

  // Handle finish (when master track ends)
  const handleFinish = useCallback(() => {
    handlePause();
  }, [handlePause]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case ' ':
          e.preventDefault();
          handlePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handleSeekBackward();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleSeekForward();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePlayPause, handleSeekBackward, handleSeekForward]);

  return (
    <div className="space-y-4">
      {/* Unified progress bar */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Combined Audio
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
        
        {/* Progress bar */}
        <div
          className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-indigo-600 rounded-full transition-all duration-100"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center space-x-2">
            {/* Seek backward */}
            <button
              onClick={handleSeekBackward}
              disabled={!isReady}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
              title="Seek back 5s (←)"
            >
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.334 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
              </svg>
            </button>

            {/* Play/Pause */}
            <button
              onClick={handlePlayPause}
              disabled={!isReady}
              className="p-3 bg-indigo-600 hover:bg-indigo-700 rounded-full text-white transition-colors disabled:opacity-50"
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            >
              {isPlaying ? (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </button>

            {/* Seek forward */}
            <button
              onClick={handleSeekForward}
              disabled={!isReady}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
              title="Seek forward 5s (→)"
            >
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
              </svg>
            </button>
          </div>

          {/* Playback rate */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Speed:</span>
            <select
              value={playbackRate}
              onChange={(e) => handleRateChange(parseFloat(e.target.value))}
              disabled={!isReady}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50"
            >
              <option value={0.5}>0.5×</option>
              <option value={0.75}>0.75×</option>
              <option value={1}>1×</option>
              <option value={1.25}>1.25×</option>
              <option value={1.5}>1.5×</option>
              <option value={2}>2×</option>
            </select>
          </div>

          {/* Keyboard shortcuts hint */}
          <div className="hidden sm:block text-xs text-gray-400 dark:text-gray-500">
            Space: play/pause • ←/→: seek
          </div>
        </div>
      </div>

      {/* Stacked waveforms */}
      <div className="grid grid-cols-1 gap-2">
        {/* Interviewer track */}
        <AudioPlayer
          ref={interviewerRef}
          audioUrl={interviewerUrl}
          label="Interviewer"
          waveColor="#3b82f6"
          progressColor="#93c5fd"
          height={60}
          isMaster={true}
          onTimeUpdate={handleInterviewerTimeUpdate}
          onReady={handleInterviewerReady}
          onFinish={handleFinish}
        />

        {/* Participant track */}
        <AudioPlayer
          ref={participantRef}
          audioUrl={participantUrl}
          label="Participant"
          waveColor="#10b981"
          progressColor="#6ee7b7"
          height={60}
          isMaster={false}
          syncTime={syncTime}
          onReady={handleParticipantReady}
        />
      </div>
    </div>
  );
}

export default DualTrackPlayer;
