/**
 * Component exports
 */

export { AudioPlayer } from './AudioPlayer';
export type { AudioPlayerProps, AudioPlayerRef } from './AudioPlayer';

export { DualTrackPlayer } from './DualTrackPlayer';
export type { DualTrackPlayerProps, DualTrackPlayerRef } from './DualTrackPlayer';

export { CombinedAudioControl } from './CombinedAudioControl';
export type { CombinedAudioControlProps, CombinedAudioControlRef } from './CombinedAudioControl';

export { TranscriptView } from './TranscriptView';
export type { TranscriptViewProps } from './TranscriptView';

export { StageSelector, StageBadges } from './StageSelector';
export type { Stage, StageSelectorProps } from './StageSelector';

export { WordEditor, EditableTurn } from './WordEditor';
export type { WordEditorProps } from './WordEditor';

export { EditToolbar } from './EditToolbar';

export { StaleStageWarning } from './StaleStageWarning';

// Phase 4 components
export { AnnotationMenu } from './AnnotationMenu';
export type { AnnotationMenuProps } from './AnnotationMenu';

export { TurnActions } from './TurnActions';
export type { TurnActionsProps } from './TurnActions';

export { FindReplace } from './FindReplace';
export type { FindReplaceProps, FindReplaceMatch } from './FindReplace';

export { EditHistory } from './EditHistory';
export type { EditHistoryProps } from './EditHistory';

export { KeyboardShortcuts, useKeyboardShortcuts, ShortcutsHelpModal, DEFAULT_SHORTCUTS } from './KeyboardShortcuts';
export type { KeyboardShortcutsProps, ShortcutConfig } from './KeyboardShortcuts';

// Phase 5 components - De-identification
export { NameListReview } from './NameListReview';
export type { NameListReviewProps } from './NameListReview';

export { PIIHighlightMode } from './PIIHighlightMode';
export type { PIIHighlightModeProps } from './PIIHighlightMode';

export { RedactionTool, RestorePIIButton, REDACTION_OPTIONS } from './RedactionTool';
export type { RedactionToolProps } from './RedactionTool';

export { PIIAuditTrail } from './PIIAuditTrail';

// Phase 6 components - Export & Polish
export { ExportDialog } from './ExportDialog';

export { PrintView } from './PrintView';

export { ComparisonView } from './ComparisonView';

export { VirtualizedTranscriptView } from './VirtualizedTranscriptView';
export type { VirtualizedTranscriptViewProps } from './VirtualizedTranscriptView';
export type { PIIAuditTrailProps } from './PIIAuditTrail';

export { ErrorBoundary, withErrorBoundary } from './ErrorBoundary';
