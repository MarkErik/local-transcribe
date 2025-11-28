#!/usr/bin/env python3
"""
Run turn building test on first 10 minutes of sample transcript.
"""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, '.')

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.processing.turn_building.split_audio_llm_turn_builder import SplitAudioTurnBuilder

LLM_URL = 'http://100.84.208.72:8083'

# Load data
sample_dir = Path('transcript-sample')
with open(sample_dir / 'interviewer_word_segments_deidentified.json') as f:
    interviewer_data = json.load(f)
with open(sample_dir / 'participant_word_segments_deidentified.json') as f:
    participant_data = json.load(f)

def to_word_segment(d):
    return WordSegment(text=d['text'], start=d['start'], end=d['end'], speaker=d.get('speaker', 'Unknown'))

all_words = [to_word_segment(w) for w in interviewer_data['words']] + [to_word_segment(w) for w in participant_data['words']]

# First 10 minutes (600 seconds)
test_words = [w for w in all_words if w.end <= 600.0]
print(f'Testing with {len(test_words)} words (first 10 minutes)')
print(f'LLM endpoint: {LLM_URL}')
print()

builder = SplitAudioTurnBuilder(llm_url=LLM_URL)

start_time = time.time()
transcript_flow = builder.build_turns(
    test_words,
    llm_url=LLM_URL
)
elapsed = time.time() - start_time

print()
print('='*60)
print('RESULTS:')
print('='*60)
print(f'Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)')
print(f'Total turns: {transcript_flow.total_turns}')
print(f'Total interjections: {transcript_flow.total_interjections}')
print(f'LLM stats: {builder.llm_stats}')

# Save to JSON file
output = {
    'metadata': {
        'test_duration_seconds': 600,
        'total_words': len(test_words),
        'processing_time_seconds': elapsed,
        'llm_url': LLM_URL,
        'llm_stats': builder.llm_stats,
    },
    'summary': {
        'total_turns': transcript_flow.total_turns,
        'total_interjections': transcript_flow.total_interjections,
        'conversation_metrics': transcript_flow.conversation_metrics,
    },
    'turns': []
}

for turn in transcript_flow.turns:
    turn_data = {
        'turn_id': turn.turn_id,
        'primary_speaker': turn.primary_speaker,
        'start': turn.start,
        'end': turn.end,
        'duration': turn.duration,
        'word_count': turn.word_count,
        'text': turn.text,
        'turn_type': turn.turn_type,
        'flow_continuity': turn.flow_continuity,
        'interjections': []
    }
    for ij in turn.interjections:
        turn_data['interjections'].append({
            'speaker': ij.speaker,
            'start': ij.start,
            'end': ij.end,
            'text': ij.text,
            'interjection_type': ij.interjection_type,
            'confidence': ij.confidence,
            'classification_method': ij.classification_method,
        })
    output['turns'].append(turn_data)

output_file = Path('transcript-sample/turn_building_test_10min.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'')
print(f'Output saved to: {output_file}')
