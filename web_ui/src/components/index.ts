/**
 * Component exports
 */

export { AudioPlayer } from './AudioPlayer';
export type { AudioPlayerProps, AudioPlayerRef } from './AudioPlayer';

export { DualTrackPlayer } from './DualTrackPlayer';
export type { DualTrackPlayerProps, DualTrackPlayerRef } from './DualTrackPlayer';

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
