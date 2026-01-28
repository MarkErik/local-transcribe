/**
 * Name List Review component for de-identification.
 * 
 * Displays discovered names from first pass and allows user to:
 * - Include/exclude names before second pass
 * - Add manually discovered names
 * - Remove incorrectly identified names
 */

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getDiscoveredNames,
  updateNameList,
  runSecondPass,
  DiscoveredName,
} from '../api/client';
import { useDeIdentificationStore } from '../store';

export interface NameListReviewProps {
  jobId: string;
  onComplete?: () => void;
  onCancel?: () => void;
}

export function NameListReview({ jobId, onComplete, onCancel }: NameListReviewProps) {
  const queryClient = useQueryClient();
  const {
    discoveredNames,
    setDiscoveredNames,
    updateNameInclusion,
    addDiscoveredName,
    removeDiscoveredName,
    setSecondPassComplete,
    setLoading,
  } = useDeIdentificationStore();
  
  const [newName, setNewName] = useState('');
  const [newSpeaker, setNewSpeaker] = useState('');
  
  // Fetch discovered names
  const { data: nameListData, isLoading: namesLoading } = useQuery({
    queryKey: ['discoveredNames', jobId],
    queryFn: () => getDiscoveredNames(jobId),
    enabled: !!jobId,
  });
  
  // Initialize store from API data
  useEffect(() => {
    if (nameListData) {
      // Use reviewed names if available, otherwise discovered names
      const names = nameListData.reviewed_names || nameListData.discovered_names;
      setDiscoveredNames(names);
    }
  }, [nameListData, setDiscoveredNames]);
  
  // Update name list mutation
  const updateMutation = useMutation({
    mutationFn: (names: DiscoveredName[]) => updateNameList(jobId, names),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discoveredNames', jobId] });
    },
  });
  
  // Run second pass mutation
  const secondPassMutation = useMutation({
    mutationFn: () => runSecondPass(jobId),
    onSuccess: () => {
      setSecondPassComplete(true);
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      onComplete?.();
    },
  });
  
  const handleToggleInclude = (name: string, include: boolean) => {
    updateNameInclusion(name, include);
  };
  
  const handleAddName = () => {
    if (!newName.trim()) return;
    
    const newDiscoveredName: DiscoveredName = {
      name: newName.trim(),
      source_speaker: newSpeaker.trim() || undefined,
      occurrences: 0, // Manual addition
      include: true,
    };
    
    addDiscoveredName(newDiscoveredName);
    setNewName('');
    setNewSpeaker('');
  };
  
  const handleRemoveName = (name: string) => {
    removeDiscoveredName(name);
  };
  
  const handleApprove = async () => {
    setLoading(true);
    try {
      // Save reviewed name list
      await updateMutation.mutateAsync(discoveredNames);
      // Run second pass
      await secondPassMutation.mutateAsync();
    } catch (error) {
      console.error('Failed to complete de-identification:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const includedCount = discoveredNames.filter(n => n.include).length;
  const totalCount = discoveredNames.length;
  
  if (namesLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading names...</span>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg shadow-lg max-w-2xl mx-auto">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">Review Discovered Names</h2>
        <p className="mt-1 text-sm text-gray-500">
          The following names were discovered in the transcript. Uncheck any that should NOT be redacted.
        </p>
      </div>
      
      {/* Name List */}
      <div className="px-6 py-4 max-h-96 overflow-y-auto">
        {discoveredNames.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No names were discovered in the transcript.
          </div>
        ) : (
          <table className="min-w-full">
            <thead>
              <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                <th className="pb-2 w-8">Include</th>
                <th className="pb-2">Name</th>
                <th className="pb-2">Speaker</th>
                <th className="pb-2 text-right">Occurrences</th>
                <th className="pb-2 w-8"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {discoveredNames.map((name, idx) => (
                <tr key={idx} className={!name.include ? 'opacity-50' : ''}>
                  <td className="py-2">
                    <input
                      type="checkbox"
                      checked={name.include}
                      onChange={(e) => handleToggleInclude(name.name, e.target.checked)}
                      className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                    />
                  </td>
                  <td className="py-2 font-medium text-gray-900">{name.name}</td>
                  <td className="py-2 text-gray-500 text-sm">
                    {name.source_speaker || '—'}
                  </td>
                  <td className="py-2 text-right text-gray-500 text-sm">
                    {name.occurrences}
                  </td>
                  <td className="py-2">
                    <button
                      onClick={() => handleRemoveName(name.name)}
                      className="text-red-500 hover:text-red-700"
                      title="Remove name"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      {/* Add Name Form */}
      <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Add a name..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
          />
          <input
            type="text"
            value={newSpeaker}
            onChange={(e) => setNewSpeaker(e.target.value)}
            placeholder="Speaker (optional)"
            className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={handleAddName}
            disabled={!newName.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add
          </button>
        </div>
      </div>
      
      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
        <div className="text-sm text-gray-500">
          {includedCount} of {totalCount} names will be redacted
        </div>
        <div className="flex gap-3">
          {onCancel && (
            <button
              onClick={onCancel}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
          )}
          <button
            onClick={handleApprove}
            disabled={secondPassMutation.isPending || updateMutation.isPending}
            className="px-4 py-2 bg-green-600 text-white rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {(secondPassMutation.isPending || updateMutation.isPending) && (
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            )}
            Approve & Continue
          </button>
        </div>
      </div>
      
      {/* Error Display */}
      {(updateMutation.isError || secondPassMutation.isError) && (
        <div className="px-6 py-3 bg-red-50 border-t border-red-200">
          <p className="text-sm text-red-600">
            {updateMutation.error?.message || secondPassMutation.error?.message || 'An error occurred'}
          </p>
        </div>
      )}
    </div>
  );
}

export default NameListReview;
