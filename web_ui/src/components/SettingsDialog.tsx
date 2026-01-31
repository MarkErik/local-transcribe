/**
 * Settings Dialog Component
 * 
 * Modal dialog for configuring application settings including:
 * - LLM server URLs (de-identification, post-processing)
 * - Remote transcription server URL
 * - Default job options (de-identification, cleanup, output formats)
 */

import { useState } from 'react';
import { useSettingsStore, AppSettings, OUTPUT_FORMATS, OutputFormatKey } from '../store';

interface SettingsDialogProps {
  onClose: () => void;
}

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const settings = useSettingsStore();
  
  // Local state for form (allows cancel without saving)
  const [localSettings, setLocalSettings] = useState<AppSettings>({
    deIdentificationUrl: settings.deIdentificationUrl,
    postProcessingUrl: settings.postProcessingUrl,
    remoteTranscriptionUrl: settings.remoteTranscriptionUrl,
    defaultEnableDeIdentification: settings.defaultEnableDeIdentification,
    defaultEnableCleanup: settings.defaultEnableCleanup,
    defaultOutputFormats: settings.defaultOutputFormats,
    defaultTranscriberProvider: settings.defaultTranscriberProvider,
    defaultTranscriberModel: settings.defaultTranscriberModel,
  });
  
  // Validation state
  const [errors, setErrors] = useState<Partial<Record<keyof AppSettings, string>>>({});
  
  // Validate URL format
  const validateUrl = (url: string): boolean => {
    if (!url) return false;
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      return false;
    }
  };
  
  // Validate all settings
  const validateSettings = (): boolean => {
    const newErrors: Partial<Record<keyof AppSettings, string>> = {};
    
    if (!validateUrl(localSettings.deIdentificationUrl)) {
      newErrors.deIdentificationUrl = 'Please enter a valid URL (e.g., http://192.168.1.100:8080)';
    }
    
    if (!validateUrl(localSettings.postProcessingUrl)) {
      newErrors.postProcessingUrl = 'Please enter a valid URL (e.g., http://192.168.1.100:8080)';
    }
    
    if (!validateUrl(localSettings.remoteTranscriptionUrl)) {
      newErrors.remoteTranscriptionUrl = 'Please enter a valid URL (e.g., http://192.168.1.100:7070)';
    }
    
    if (localSettings.defaultOutputFormats.length === 0) {
      newErrors.defaultOutputFormats = 'Please select at least one output format';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSave = () => {
    if (validateSettings()) {
      settings.updateSettings(localSettings);
      onClose();
    }
  };
  
  const handleOutputFormatToggle = (format: OutputFormatKey) => {
    setLocalSettings(prev => {
      const formats = prev.defaultOutputFormats.includes(format)
        ? prev.defaultOutputFormats.filter(f => f !== format)
        : [...prev.defaultOutputFormats, format];
      return { ...prev, defaultOutputFormats: formats };
    });
  };
  
  const handleResetToDefaults = () => {
    settings.resetToDefaults();
    // Reload local settings from the reset store
    setLocalSettings({
      deIdentificationUrl: 'http://0.0.0.0:8080',
      postProcessingUrl: 'http://0.0.0.0:8080',
      remoteTranscriptionUrl: 'http://0.0.0.0:7070',
      defaultEnableDeIdentification: true,
      defaultEnableCleanup: false,
      defaultOutputFormats: ['turns-json', 'timestamped-txt'],
      defaultTranscriberProvider: 'granite',
      defaultTranscriberModel: 'granite-8b',
    });
    setErrors({});
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto py-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl mx-4 my-auto max-h-[calc(100vh-2rem)] overflow-y-auto">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-800 z-10">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Settings</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-6">
          {/* Server URLs Section */}
          <section>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
              </svg>
              Server Configuration
            </h3>
            
            <div className="space-y-4">
              {/* Remote Transcription URL */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Remote Transcription Server URL
                </label>
                <input
                  type="text"
                  value={localSettings.remoteTranscriptionUrl}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, remoteTranscriptionUrl: e.target.value }))}
                  placeholder="http://192.168.1.100:7070"
                  className={`w-full px-3 py-2 border rounded-md focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white ${
                    errors.remoteTranscriptionUrl 
                      ? 'border-red-300 dark:border-red-600' 
                      : 'border-gray-300 dark:border-gray-600'
                  }`}
                />
                {errors.remoteTranscriptionUrl && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.remoteTranscriptionUrl}</p>
                )}
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  URL of the server running the transcription model (e.g., Granite)
                </p>
              </div>
              
              {/* De-identification LLM URL */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  De-identification LLM Server URL
                </label>
                <input
                  type="text"
                  value={localSettings.deIdentificationUrl}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, deIdentificationUrl: e.target.value }))}
                  placeholder="http://192.168.1.100:8080"
                  className={`w-full px-3 py-2 border rounded-md focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white ${
                    errors.deIdentificationUrl 
                      ? 'border-red-300 dark:border-red-600' 
                      : 'border-gray-300 dark:border-gray-600'
                  }`}
                />
                {errors.deIdentificationUrl && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.deIdentificationUrl}</p>
                )}
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  URL of the LLM server used for PII detection and de-identification
                </p>
              </div>
              
              {/* Post-processing LLM URL */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Post-processing LLM Server URL
                </label>
                <input
                  type="text"
                  value={localSettings.postProcessingUrl}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, postProcessingUrl: e.target.value }))}
                  placeholder="http://192.168.1.100:8080"
                  className={`w-full px-3 py-2 border rounded-md focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white ${
                    errors.postProcessingUrl 
                      ? 'border-red-300 dark:border-red-600' 
                      : 'border-gray-300 dark:border-gray-600'
                  }`}
                />
                {errors.postProcessingUrl && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.postProcessingUrl}</p>
                )}
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  URL of the LLM server used for transcript cleanup and post-processing
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-200 dark:border-gray-700" />
          
          {/* Default Job Options Section */}
          <section>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Default Job Options
            </h3>
            
            <div className="space-y-4">
              {/* Processing toggles */}
              <div className="space-y-3">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={localSettings.defaultEnableDeIdentification}
                    onChange={(e) => setLocalSettings(prev => ({ ...prev, defaultEnableDeIdentification: e.target.checked }))}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                    Enable de-identification by default
                  </span>
                </label>
                <p className="ml-6 text-xs text-gray-500 dark:text-gray-400">
                  Automatically redact personal information (names, locations, etc.) from transcripts
                </p>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={localSettings.defaultEnableCleanup}
                    onChange={(e) => setLocalSettings(prev => ({ ...prev, defaultEnableCleanup: e.target.checked }))}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                    Enable transcript cleanup by default
                  </span>
                </label>
                <p className="ml-6 text-xs text-gray-500 dark:text-gray-400">
                  Use LLM to clean up transcription artifacts, fix grammar, and improve readability
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-200 dark:border-gray-700" />
          
          {/* Output Formats Section */}
          <section>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Default Output Formats
            </h3>
            
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Select which output formats to generate by default for new jobs
            </p>
            
            {errors.defaultOutputFormats && (
              <p className="mb-3 text-sm text-red-600 dark:text-red-400">{errors.defaultOutputFormats}</p>
            )}
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {(Object.entries(OUTPUT_FORMATS) as [OutputFormatKey, string][]).map(([key, description]) => (
                <label
                  key={key}
                  className={`flex items-start p-3 border rounded-lg cursor-pointer transition-colors ${
                    localSettings.defaultOutputFormats.includes(key)
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={localSettings.defaultOutputFormats.includes(key)}
                    onChange={() => handleOutputFormatToggle(key)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 mt-0.5"
                  />
                  <div className="ml-2">
                    <span className="text-sm font-medium text-gray-900 dark:text-white block">
                      {key}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {description}
                    </span>
                  </div>
                </label>
              ))}
            </div>
          </section>
          
          <hr className="border-gray-200 dark:border-gray-700" />
          
          {/* Transcription Settings Section */}
          <section>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              Transcription Settings
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Default Transcriber
                </label>
                <select
                  value={localSettings.defaultTranscriberProvider}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, defaultTranscriberProvider: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                >
                  <option value="granite">Granite (Local)</option>
                  <option value="whisper">Whisper (Local)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Default Model
                </label>
                <select
                  value={localSettings.defaultTranscriberModel}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, defaultTranscriberModel: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                >
                  {localSettings.defaultTranscriberProvider === 'granite' ? (
                    <>
                      <option value="granite-8b">Granite 8B</option>
                      <option value="granite-3b">Granite 3B</option>
                    </>
                  ) : (
                    <>
                      <option value="whisper-large-v3">Whisper Large v3</option>
                      <option value="whisper-medium">Whisper Medium</option>
                      <option value="whisper-small">Whisper Small</option>
                    </>
                  )}
                </select>
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 sticky bottom-0 bg-white dark:bg-gray-800 flex justify-between items-center">
          <button
            onClick={handleResetToDefaults}
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            Reset to defaults
          </button>
          
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
