/**
 * Combined Audio Control Component
 * 
 * A unified audio control bar for synchronized dual-track playback.
 * Displays both interviewer and participant waveforms in a mirrored layout
 * with the interviewer above the timeline and participant below.
 */

import { useRef, useState, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react';
import WaveSurfer from 'wavesurfer.js';

export interface CombinedAudioControlProps {
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

export interface CombinedAudioControlRef {
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

export const CombinedAudioControl = forwardRef<CombinedAudioControlRef, CombinedAudioControlProps>(function CombinedAudioControl({
  interviewerUrl,
  participantUrl,
  onTimeUpdate,
  onSeek,
  onReady,
}, ref) {
  // Refs for WaveSurfer instances
  const interviewerContainerRef = useRef<HTMLDivElement>(null);
  const participantContainerRef = useRef<HTMLDivElement>(null);
  const interviewerWsRef = useRef<WaveSurfer | null>(null);
  const participantWsRef = useRef<WaveSurfer | null>(null);
  
  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [interviewerReady, setInterviewerReady] = useState(false);
  const [participantReady, setParticipantReady] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [loadError, setLoadError] = useState<string | null>(null);

  const isReady = interviewerReady && participantReady;

  // Initialize WaveSurfer instances
  useEffect(() => {
    if (!interviewerContainerRef.current || !participantContainerRef.current) return;

    setLoadError(null);
    setInterviewerReady(false);
    setParticipantReady(false);

    // Create interviewer WaveSurfer
    const interviewerWs = WaveSurfer.create({
      container: interviewerContainerRef.current,
      waveColor: '#3b82f6',
      progressColor: '#1d4ed8',
      height: 50,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      cursorWidth: 2,
      cursorColor: '#dc2626',
      normalize: true,
      interact: true,
      fetchParams: { mode: 'cors' },
    });

    // Create participant WaveSurfer
    const participantWs = WaveSurfer.create({
      container: participantContainerRef.current,
      waveColor: '#10b981',
      progressColor: '#047857',
      height: 50,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      cursorWidth: 2,
      cursorColor: '#dc2626',
      normalize: true,
      interact: true,
      fetchParams: { mode: 'cors' },
    });

    // Load audio files
    interviewerWs.load(interviewerUrl);
    participantWs.load(participantUrl);

    // Interviewer events (master track)
    interviewerWs.on('ready', () => {
      setInterviewerReady(true);
      const dur = interviewerWs.getDuration();
      setDuration(dur);
    });

    interviewerWs.on('error', (err: Error | string) => {
      const errorMessage = typeof err === 'string' ? err : err.message;
      console.error('Interviewer WaveSurfer error:', errorMessage);
      setLoadError(errorMessage || 'Failed to load interviewer audio');
    });

    interviewerWs.on('audioprocess', (time: number) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
      // Sync participant
      if (participantWsRef.current) {
        const participantTime = participantWsRef.current.getCurrentTime();
        if (Math.abs(participantTime - time) > 0.5) {
          participantWsRef.current.seekTo(time / participantWsRef.current.getDuration());
        }
      }
    });

    interviewerWs.on('seeking', (time: number) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
      // Sync participant on seek
      if (participantWsRef.current && participantWsRef.current.getDuration() > 0) {
        participantWsRef.current.seekTo(time / participantWsRef.current.getDuration());
      }
    });

    interviewerWs.on('play', () => setIsPlaying(true));
    interviewerWs.on('pause', () => setIsPlaying(false));
    interviewerWs.on('finish', () => {
      setIsPlaying(false);
      participantWsRef.current?.pause();
    });

    // Participant events
    participantWs.on('ready', () => {
      setParticipantReady(true);
    });

    participantWs.on('error', (err: Error | string) => {
      const errorMessage = typeof err === 'string' ? err : err.message;
      console.error('Participant WaveSurfer error:', errorMessage);
      setLoadError(errorMessage || 'Failed to load participant audio');
    });

    // Handle clicks on participant waveform to seek both
    participantWs.on('seeking', (time: number) => {
      if (interviewerWsRef.current && interviewerWsRef.current.getDuration() > 0) {
        interviewerWsRef.current.seekTo(time / interviewerWsRef.current.getDuration());
        setCurrentTime(time);
        onTimeUpdate?.(time);
      }
    });

    interviewerWsRef.current = interviewerWs;
    participantWsRef.current = participantWs;

    return () => {
      interviewerWs.destroy();
      participantWs.destroy();
    };
  }, [interviewerUrl, participantUrl, onTimeUpdate]);

  // Notify when both tracks are ready
  useEffect(() => {
    if (isReady && onReady) {
      onReady();
    }
  }, [isReady, onReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      interviewerWsRef.current?.pause();
      participantWsRef.current?.pause();
    };
  }, []);

  // Play both tracks
  const handlePlay = useCallback(() => {
    interviewerWsRef.current?.play();
    participantWsRef.current?.play();
    setIsPlaying(true);
  }, []);

  // Pause both tracks
  const handlePause = useCallback(() => {
    interviewerWsRef.current?.pause();
    participantWsRef.current?.pause();
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
    if (interviewerWsRef.current && interviewerWsRef.current.getDuration() > 0) {
      interviewerWsRef.current.seekTo(time / interviewerWsRef.current.getDuration());
    }
    if (participantWsRef.current && participantWsRef.current.getDuration() > 0) {
      participantWsRef.current.seekTo(time / participantWsRef.current.getDuration());
    }
    setCurrentTime(time);
    onSeek?.(time);
  }, [onSeek]);

  // Expose ref API
  useImperativeHandle(ref, () => ({
    play: () => {
      interviewerWsRef.current?.play();
      participantWsRef.current?.play();
      setIsPlaying(true);
    },
    pause: () => {
      interviewerWsRef.current?.pause();
      participantWsRef.current?.pause();
      setIsPlaying(false);
    },
    seekTo: handleSeek,
    getCurrentTime: () => currentTime,
    getDuration: () => duration,
    isPlaying: () => isPlaying,
    setPlaybackRate: (rate: number) => {
      interviewerWsRef.current?.setPlaybackRate(rate);
      participantWsRef.current?.setPlaybackRate(rate);
      setPlaybackRate(rate);
    },
  }), [handleSeek, currentTime, duration, isPlaying]);

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
    interviewerWsRef.current?.setPlaybackRate(rate);
    participantWsRef.current?.setPlaybackRate(rate);
    setPlaybackRate(rate);
  }, []);

  // Handle progress bar click
  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isReady || duration === 0) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newTime = percent * duration;
    handleSeek(newTime);
  }, [isReady, duration, handleSeek]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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

  // Expose seek method for external use (e.g., clicking on transcript)
  useEffect(() => {
    // This allows the parent component to call handleSeek through onSeek
  }, [handleSeek]);

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 shadow-lg">
      {/* Controls Row */}
      <div className="px-4 py-2 flex items-center justify-between border-b border-gray-100 dark:border-gray-700">
        {/* Left: Play controls */}
        <div className="flex items-center space-x-2">
          {/* Speed selector */}
          <div className="flex items-center space-x-1">
            <span className="text-xs text-gray-500 dark:text-gray-400">Speed</span>
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
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
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

        {/* Center: Progress bar (simplified, main interaction is on waveforms) */}
        <div className="flex-1 mx-4">
          <div
            className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer"
            onClick={handleProgressClick}
          >
            <div
              className="h-full bg-indigo-600 rounded-full transition-all duration-100"
              style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%' }}
            />
          </div>
        </div>

        {/* Right: Time display and hints */}
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-700 dark:text-gray-300 font-mono min-w-[100px] text-right">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
          <div className="hidden lg:block text-xs text-gray-400 dark:text-gray-500">
            Space: play/pause • ←/→: seek
          </div>
        </div>
      </div>

      {/* Combined Waveform Area */}
      <div className="px-4 py-2">
        {/* Loading indicator */}
        {!isReady && !loadError && (
          <div className="flex items-center justify-center py-6">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600" />
            <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
              Loading audio...
            </span>
          </div>
        )}

        {/* Error state */}
        {loadError && (
          <div className="flex items-center justify-center py-6 text-red-500 dark:text-red-400">
            <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm">Audio error: {loadError}</span>
          </div>
        )}

        {/* Waveforms container */}
        <div className={`relative ${!isReady && !loadError ? 'invisible h-0' : ''}`}>
          {/* Speaker labels */}
          <div className="absolute left-0 top-0 bottom-0 w-20 flex flex-col justify-between py-1 z-10 pointer-events-none">
            <span className="text-xs font-medium text-blue-600 dark:text-blue-400 bg-white/80 dark:bg-gray-800/80 px-1 rounded">
              INT
            </span>
            <span className="text-xs font-medium text-green-600 dark:text-green-400 bg-white/80 dark:bg-gray-800/80 px-1 rounded">
              PAR
            </span>
          </div>

          {/* Interviewer waveform (above timeline) */}
          <div className="ml-12">
            <div 
              ref={interviewerContainerRef} 
              className="w-full cursor-pointer"
            />
            
            {/* Center timeline */}
            <div className="h-px bg-gray-300 dark:bg-gray-600 w-full" />
            
            {/* Participant waveform (below timeline, mirrored) */}
            <div 
              ref={participantContainerRef} 
              className="w-full cursor-pointer"
              style={{ transform: 'scaleY(-1)' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
});

export default CombinedAudioControl;
