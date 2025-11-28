#!/usr/bin/env python3
"""Debug trace of the smart grouping algorithm."""

import json
from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.providers.turn_builders.split_audio_base import merge_word_streams
from local_transcribe.providers.turn_builders.split_audio_data_structures import TurnBuilderConfig

# Load sample data
with open('transcript-sample/participant_word_segments_deidentified.json') as f:
    participant = json.load(f)
with open('transcript-sample/interviewer_word_segments_deidentified.json') as f:
    interviewer = json.load(f)

# Convert to WordSegment objects
words = []
for w in participant['words']:
    words.append(WordSegment(text=w['text'], start=w['start'], end=w['end'], speaker=w['speaker']))
for w in interviewer['words']:
    words.append(WordSegment(text=w['text'], start=w['start'], end=w['end'], speaker=w['speaker']))

merged = merge_word_streams(words)
config = TurnBuilderConfig()

# Focus on the transition around 'comfortable' at 80s
# The Interviewer says "and what tells you that you feel" ending around 76s
# Then Participant speaks from ~73s onwards (overlapping!)

print('Detailed trace around 73-85s:')
print('='*60)

region_words = [w for w in merged if 73 <= w.start <= 85]
for i, w in enumerate(region_words):
    print(f'{i:3}: [{w.speaker:12}] {w.start:6.2f}-{w.end:6.2f}s "{w.text}"')

print('\n\nNow check: what primary segment contains the timestamp 80.76s?')
from local_transcribe.providers.turn_builders.split_audio_base import smart_group_with_interjection_detection

primary_segments, pending_interjections = smart_group_with_interjection_detection(merged, config)

for i, seg in enumerate(primary_segments):
    if seg.start <= 80.76 <= seg.end:
        print(f'Segment {i}: [{seg.speaker}] {seg.start:.2f}-{seg.end:.2f}s')
        print(f'  Word count: {seg.word_count}, Duration: {seg.duration:.2f}s')
        print(f'  Text: "{seg.text[:100]}..."')

# Show segments 3-10
print('\n\nPrimary segments 3-10:')
for i in range(3, min(11, len(primary_segments))):
    seg = primary_segments[i]
    text_preview = seg.text[:60] + '...' if len(seg.text) > 60 else seg.text
    print(f'{i:3}: [{seg.speaker:12}] {seg.start:6.2f}-{seg.end:6.2f}s ({seg.word_count:3} words) "{text_preview}"')

# Simplified manual trace of the algorithm around 80-85s
print('Manual trace around 80-85s:')
print('='*60)

region_words = [w for w in merged if 79 <= w.start <= 85]

primary_speaker = None
primary_words = []
other_speaker = None
other_speaker_buffer = []

def is_likely_interjection(buffer_words, text, duration):
    word_count = len(buffer_words)
    if word_count > config.max_interjection_words:
        return False
    if duration > config.max_interjection_duration:
        return False
    if word_count <= 2:
        return True
    return False

for i, word in enumerate(region_words):
    speaker = word.speaker
    
    if primary_speaker is None:
        primary_speaker = speaker
        primary_words = [word]
        print(f'[{i}] Init primary: {speaker} "{word.text}"')
        continue
    
    if speaker == primary_speaker:
        if other_speaker_buffer:
            other_text = ' '.join(w.text for w in other_speaker_buffer)
            other_duration = other_speaker_buffer[-1].end - other_speaker_buffer[0].start
            is_ij = is_likely_interjection(other_speaker_buffer, other_text, other_duration)
            print(f'[{i}] Primary returns ({speaker}), other_buffer="{other_text}" ({len(other_speaker_buffer)} words, {other_duration:.2f}s) -> interjection={is_ij}')
            if not is_ij:
                print(f'     !!! NOT INTERJECTION - becomes new primary')
            other_speaker = None
            other_speaker_buffer = []
        primary_words.append(word)
        continue
    
    # Different speaker
    if other_speaker is None:
        other_speaker = speaker
        other_speaker_buffer = [word]
        print(f'[{i}] Other speaker starts: {speaker} "{word.text}" at {word.start:.2f}s')
    elif speaker == other_speaker:
        other_speaker_buffer.append(word)
        print(f'[{i}] Other speaker continues: {speaker} "{word.text}"')
    else:
        print(f'[{i}] THIRD SPEAKER: {speaker} "{word.text}"')

print('\n\nChecking first few single-word segments:')
from local_transcribe.providers.turn_builders.split_audio_base import smart_group_with_interjection_detection

primary_segments, pending_interjections = smart_group_with_interjection_detection(merged, config)

# Show where 'comfortable' ended up
print('\nLooking for "comfortable" in segments:')
for seg in primary_segments:
    if 'comfortable' in seg.text.lower():
        print(f'  PRIMARY: [{seg.speaker}] {seg.start:.2f}-{seg.end:.2f}s "{seg.text}"')
for ij in pending_interjections:
    text = ' '.join(w.text for w in ij.words)
    if 'comfortable' in text.lower():
        print(f'  INTERJECTION: [{ij.speaker}] {ij.start:.2f}-{ij.end:.2f}s "{text}"')

print('\nLooking for single-word "to" from Interviewer:')
for seg in primary_segments:
    if seg.text.strip().lower() == 'to' and seg.speaker == 'Interviewer':
        print(f'  PRIMARY: [{seg.speaker}] {seg.start:.2f}-{seg.end:.2f}s "{seg.text}"')
for ij in pending_interjections:
    text = ' '.join(w.text for w in ij.words)
    if text.strip().lower() == 'to' and ij.speaker == 'Interviewer':
        print(f'  INTERJECTION: [{ij.speaker}] {ij.start:.2f}-{ij.end:.2f}s "{text}"')

# Check all interjections around 80-100s
print('\nAll pending interjections between 80-100s:')
for ij in pending_interjections:
    if 80 <= ij.start <= 100:
        text = ' '.join(w.text for w in ij.words)
        print(f'  [{ij.speaker}] {ij.start:.2f}-{ij.end:.2f}s "{text}" (during {ij.detected_during_turn_of})')

# Count unique interjections
unique_texts = set()
for ij in pending_interjections:
    text = ' '.join(w.text for w in ij.words)
    unique_texts.add(f'{ij.speaker}: {text}')
print(f'\nTotal pending interjections: {len(pending_interjections)}')
print(f'Unique interjection texts: {len(unique_texts)}')

# Check first few interjections to see where they start
print('\nFirst 10 pending interjections:')
for ij in pending_interjections[:10]:
    text = ' '.join(w.text for w in ij.words)
    print(f'  [{ij.speaker}] {ij.start:.2f}-{ij.end:.2f}s "{text}" (during {ij.detected_during_turn_of})')

# Check who are the interjections from
participant_ij = [ij for ij in pending_interjections if ij.speaker == 'Participant']
interviewer_ij = [ij for ij in pending_interjections if ij.speaker == 'Interviewer']
print(f'\nParticipant interjections: {len(participant_ij)}')
print(f'Interviewer interjections: {len(interviewer_ij)}')
