/**
 * Single-track audio player component using wavesurfer.js
 * 
 * Provides waveform visualization, playback controls, and time display.
 */

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import WaveSurfer from 'wavesurfer.js';

export interface AudioPlayerProps {
  /** URL to the audio file */
  audioUrl: string;
  /** Label for this track (e.g., "Interviewer", "Participant") */
  label?: string;
  /** Color for the waveform */
  waveColor?: string;
  /** Color for the progress portion of the waveform */
  progressColor?: string;
  /** Height of the waveform in pixels */
  height?: number;
  /** Called when playback time updates */
  onTimeUpdate?: (currentTime: number) => void;
  /** Called when audio finishes playing */
  onFinish?: () => void;
  /** Called when audio is ready */
  onReady?: (duration: number) => void;
  /** Whether this player is the master (controls playback) */
  isMaster?: boolean;
  /** External time to seek to (for slave sync) */
  syncTime?: number;
}

export interface AudioPlayerRef {
  play: () => void;
  pause: () => void;
  seekTo: (time: number) => void;
  getCurrentTime: () => number;
  getDuration: () => number;
  isPlaying: () => boolean;
  setPlaybackRate: (rate: number) => void;
  setMuted: (muted: boolean) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export const AudioPlayer = forwardRef<AudioPlayerRef, AudioPlayerProps>(function AudioPlayer(
  {
    audioUrl,
    label,
    waveColor = '#4f46e5',
    progressColor = '#818cf8',
    height = 80,
    onTimeUpdate,
    onFinish,
    onReady,
    isMaster = true,
    syncTime,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  // Initialize WaveSurfer
  useEffect(() => {
    if (!containerRef.current) return;

    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      height,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      cursorWidth: 1,
      cursorColor: '#1f2937',
      normalize: true,
      backend: 'WebAudio',
    });

    wavesurfer.load(audioUrl);

    wavesurfer.on('ready', () => {
      setIsReady(true);
      const dur = wavesurfer.getDuration();
      setDuration(dur);
      onReady?.(dur);
    });

    wavesurfer.on('audioprocess', (time: number) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });

    wavesurfer.on('seeking', (time: number) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });

    wavesurfer.on('play', () => setIsPlaying(true));
    wavesurfer.on('pause', () => setIsPlaying(false));
    wavesurfer.on('finish', () => {
      setIsPlaying(false);
      onFinish?.();
    });

    wavesurferRef.current = wavesurfer;

    return () => {
      wavesurfer.destroy();
    };
  }, [audioUrl, waveColor, progressColor, height, onTimeUpdate, onFinish, onReady]);

  // Sync to external time (for slave players)
  useEffect(() => {
    if (!isMaster && syncTime !== undefined && wavesurferRef.current && isReady) {
      const ws = wavesurferRef.current;
      const currentWsTime = ws.getCurrentTime();
      // Only sync if drift is more than 0.5 seconds
      if (Math.abs(currentWsTime - syncTime) > 0.5) {
        ws.seekTo(syncTime / ws.getDuration());
      }
    }
  }, [syncTime, isMaster, isReady]);

  // Expose player controls via ref
  useImperativeHandle(ref, () => ({
    play: () => wavesurferRef.current?.play(),
    pause: () => wavesurferRef.current?.pause(),
    seekTo: (time: number) => {
      if (wavesurferRef.current && duration > 0) {
        wavesurferRef.current.seekTo(time / duration);
      }
    },
    getCurrentTime: () => wavesurferRef.current?.getCurrentTime() ?? 0,
    getDuration: () => wavesurferRef.current?.getDuration() ?? 0,
    isPlaying: () => wavesurferRef.current?.isPlaying() ?? false,
    setPlaybackRate: (rate: number) => {
      wavesurferRef.current?.setPlaybackRate(rate);
      setPlaybackRate(rate);
    },
    setMuted: (muted: boolean) => {
      wavesurferRef.current?.setMuted(muted);
      setIsMuted(muted);
    },
  }));

  const handlePlayPause = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  const handleSeekBackward = useCallback(() => {
    if (wavesurferRef.current) {
      const ws = wavesurferRef.current;
      const newTime = Math.max(0, ws.getCurrentTime() - 5);
      ws.seekTo(newTime / ws.getDuration());
    }
  }, []);

  const handleSeekForward = useCallback(() => {
    if (wavesurferRef.current) {
      const ws = wavesurferRef.current;
      const newTime = Math.min(ws.getDuration(), ws.getCurrentTime() + 5);
      ws.seekTo(newTime / ws.getDuration());
    }
  }, []);

  const handleRateChange = useCallback((rate: number) => {
    wavesurferRef.current?.setPlaybackRate(rate);
    setPlaybackRate(rate);
  }, []);

  const handleMuteToggle = useCallback(() => {
    if (wavesurferRef.current) {
      const newMuted = !isMuted;
      wavesurferRef.current.setMuted(newMuted);
      setIsMuted(newMuted);
    }
  }, [isMuted]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      {/* Header with label and time */}
      <div className="flex justify-between items-center mb-2">
        {label && (
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </span>
        )}
        <span className="text-sm text-gray-500 dark:text-gray-400 font-mono">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      {/* Waveform container */}
      <div ref={containerRef} className="mb-3" />

      {/* Loading indicator */}
      {!isReady && (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600" />
          <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
            Loading audio...
          </span>
        </div>
      )}

      {/* Controls - only show for master player */}
      {isReady && isMaster && (
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {/* Seek backward */}
            <button
              onClick={handleSeekBackward}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Seek back 5s"
            >
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.334 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
              </svg>
            </button>

            {/* Play/Pause */}
            <button
              onClick={handlePlayPause}
              className="p-3 bg-indigo-600 hover:bg-indigo-700 rounded-full text-white transition-colors"
              title={isPlaying ? 'Pause' : 'Play'}
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
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Seek forward 5s"
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
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300"
            >
              <option value={0.5}>0.5×</option>
              <option value={0.75}>0.75×</option>
              <option value={1}>1×</option>
              <option value={1.25}>1.25×</option>
              <option value={1.5}>1.5×</option>
              <option value={2}>2×</option>
            </select>
          </div>

          {/* Mute toggle */}
          <button
            onClick={handleMuteToggle}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? (
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clipRule="evenodd" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
              </svg>
            )}
          </button>
        </div>
      )}

      {/* Slave player controls - just mute */}
      {isReady && !isMaster && (
        <div className="flex items-center justify-end">
          <button
            onClick={handleMuteToggle}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              isMuted 
                ? 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400' 
                : 'bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300'
            }`}
          >
            {isMuted ? 'Muted' : 'Playing'}
          </button>
        </div>
      )}
    </div>
  );
});

export default AudioPlayer;
