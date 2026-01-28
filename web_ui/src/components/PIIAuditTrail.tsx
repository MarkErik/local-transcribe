/**
 * PII Audit Trail component.
 * 
 * Displays a chronological list of all PII replacements for a job,
 * with filtering, navigation, and export capabilities.
 */

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getPIIReplacements, PIIReplacement } from '../api/client';
import { useDeIdentificationStore } from '../store';
import { RestorePIIButton } from './RedactionTool';

interface PIIAuditTrailProps {
  jobId: string;
  onJumpToLocation?: (turnId: number, wordIndex: number) => void;
  className?: string;
}

type FilterType = 'all' | 'first-pass' | 'second-pass' | 'manual' | 'overrides';

export function PIIAuditTrail({ jobId, onJumpToLocation, className = '' }: PIIAuditTrailProps) {
  const { setPIIReplacements } = useDeIdentificationStore();
  const [filter, setFilter] = useState<FilterType>('all');
  const [speakerFilter, setSpeakerFilter] = useState<string>('');
  
  // Fetch PII replacements
  const { data, isLoading, error } = useQuery({
    queryKey: ['piiReplacements', jobId],
    queryFn: () => getPIIReplacements(jobId),
    enabled: !!jobId,
  });
  
  // Update store when data changes
  React.useEffect(() => {
    if (data) {
      setPIIReplacements(data.replacements);
    }
  }, [data, setPIIReplacements]);
  
  // Get unique speakers for filter dropdown
  const speakers = useMemo(() => {
    if (!data) return [];
    const uniqueSpeakers = new Set(data.replacements.map(r => r.speaker).filter(Boolean));
    return Array.from(uniqueSpeakers) as string[];
  }, [data]);
  
  // Filter replacements
  const filteredReplacements = useMemo(() => {
    if (!data) return [];
    
    let filtered = data.replacements;
    
    // Apply type filter
    switch (filter) {
      case 'first-pass':
        filtered = filtered.filter(r => r.pass_number === 1 && !r.is_override);
        break;
      case 'second-pass':
        filtered = filtered.filter(r => r.pass_number === 2 && !r.is_override);
        break;
      case 'manual':
        filtered = filtered.filter(r => r.is_manual && !r.is_override);
        break;
      case 'overrides':
        filtered = filtered.filter(r => r.is_override);
        break;
    }
    
    // Apply speaker filter
    if (speakerFilter) {
      filtered = filtered.filter(r => r.speaker === speakerFilter);
    }
    
    return filtered;
  }, [data, filter, speakerFilter]);
  
  // Group by date
  const groupedReplacements = useMemo(() => {
    const groups: Record<string, PIIReplacement[]> = {};
    
    filteredReplacements.forEach(r => {
      const date = r.created_at.split('T')[0];
      if (!groups[date]) groups[date] = [];
      groups[date].push(r);
    });
    
    return groups;
  }, [filteredReplacements]);
  
  // Export as CSV
  const handleExportCSV = () => {
    if (!data) return;
    
    const headers = ['ID', 'Speaker', 'Original', 'Replacement', 'Pass', 'Type', 'Turn ID', 'Word Index', 'Created At'];
    const rows = data.replacements.map(r => [
      r.id,
      r.speaker || '',
      r.original_text,
      r.replacement_text,
      r.pass_number || 'Manual',
      r.is_override ? 'Override' : (r.is_manual ? 'Manual' : 'Automatic'),
      r.turn_id || '',
      r.word_index || '',
      r.created_at,
    ]);
    
    const csv = [headers, ...rows].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `pii-audit-${jobId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Export as JSON
  const handleExportJSON = () => {
    if (!data) return;
    
    const json = JSON.stringify(data.replacements, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `pii-audit-${jobId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const getTypeLabel = (r: PIIReplacement): string => {
    if (r.is_override) return 'Restored';
    if (r.is_manual) return 'Manual';
    if (r.pass_number === 1) return 'Pass 1';
    if (r.pass_number === 2) return 'Pass 2';
    return 'Unknown';
  };
  
  const getTypeBadgeClass = (r: PIIReplacement): string => {
    if (r.is_override) return 'bg-gray-100 text-gray-600';
    if (r.is_manual) return 'bg-red-100 text-red-700';
    if (r.pass_number === 2) return 'bg-orange-100 text-orange-700';
    return 'bg-yellow-100 text-yellow-700';
  };
  
  if (isLoading) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600"></div>
        <span className="ml-2 text-gray-600">Loading audit trail...</span>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className={`p-4 bg-red-50 text-red-600 rounded ${className}`}>
        Failed to load audit trail: {error.message}
      </div>
    );
  }
  
  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-gray-900">PII Audit Trail</h3>
          <span className="text-sm text-gray-500">{data?.total || 0} total entries</span>
        </div>
      </div>
      
      {/* Filters */}
      <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-3 flex-wrap">
        {/* Type filter */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500">Type:</span>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as FilterType)}
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-purple-500 focus:border-purple-500"
          >
            <option value="all">All</option>
            <option value="first-pass">First Pass</option>
            <option value="second-pass">Second Pass</option>
            <option value="manual">Manual</option>
            <option value="overrides">Restored</option>
          </select>
        </div>
        
        {/* Speaker filter */}
        {speakers.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500">Speaker:</span>
            <select
              value={speakerFilter}
              onChange={(e) => setSpeakerFilter(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="">All</option>
              {speakers.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        )}
        
        {/* Export buttons */}
        <div className="flex-1"></div>
        <button
          onClick={handleExportCSV}
          className="text-xs text-purple-600 hover:text-purple-700 flex items-center gap-1"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          CSV
        </button>
        <button
          onClick={handleExportJSON}
          className="text-xs text-purple-600 hover:text-purple-700 flex items-center gap-1"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          JSON
        </button>
      </div>
      
      {/* Audit list */}
      <div className="max-h-96 overflow-y-auto">
        {filteredReplacements.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            No PII replacements found matching the current filters.
          </div>
        ) : (
          Object.entries(groupedReplacements).map(([date, replacements]) => (
            <div key={date}>
              {/* Date header */}
              <div className="px-4 py-2 bg-gray-50 text-xs font-medium text-gray-500 sticky top-0">
                {new Date(date).toLocaleDateString('en-US', { 
                  weekday: 'long', 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </div>
              
              {/* Entries for this date */}
              {replacements.map((r) => (
                <div
                  key={r.id}
                  className="px-4 py-3 border-b border-gray-100 hover:bg-gray-50 flex items-start gap-3"
                >
                  {/* Type badge */}
                  <span className={`px-2 py-0.5 text-xs rounded ${getTypeBadgeClass(r)}`}>
                    {getTypeLabel(r)}
                  </span>
                  
                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`font-mono text-sm ${r.is_override ? 'line-through text-gray-400' : 'text-red-600'}`}>
                        {r.original_text}
                      </span>
                      <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                      <span className="font-mono text-sm text-purple-600">
                        {r.replacement_text}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                      {r.speaker && <span>Speaker: {r.speaker}</span>}
                      {r.turn_id !== undefined && <span>Turn: {r.turn_id}</span>}
                      <span>{new Date(r.created_at).toLocaleTimeString()}</span>
                    </div>
                  </div>
                  
                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {onJumpToLocation && r.turn_id !== undefined && r.word_index !== undefined && (
                      <button
                        onClick={() => onJumpToLocation(r.turn_id!, r.word_index!)}
                        className="text-blue-600 hover:text-blue-700"
                        title="Jump to location"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>
                    )}
                    {!r.is_override && (
                      <RestorePIIButton
                        jobId={jobId}
                        replacement={r}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
      
      {/* Summary footer */}
      <div className="px-4 py-3 border-t border-gray-200 bg-gray-50 text-xs text-gray-500 flex items-center gap-4">
        <span>
          First pass: {data?.replacements.filter(r => r.pass_number === 1 && !r.is_override).length || 0}
        </span>
        <span>
          Second pass: {data?.replacements.filter(r => r.pass_number === 2 && !r.is_override).length || 0}
        </span>
        <span>
          Manual: {data?.replacements.filter(r => r.is_manual && !r.is_override).length || 0}
        </span>
        <span>
          Restored: {data?.replacements.filter(r => r.is_override).length || 0}
        </span>
      </div>
    </div>
  );
}

export default PIIAuditTrail;
