// DOM manipulation and UI updates

// Import utility functions
import { escapeHtml } from './utils.js';

// Import global state variables from app.js
import {
    filesLoaded,
    scriptFilesLoaded,
    turnComparisons,
    turnTimeline,
    currentPlayingTurnIndex,
    diffSegments,
    currentDiffIndex
} from './app.js';

export function updateCompareButton(filesLoaded) {
    const btnCompare = document.getElementById('btn-compare');
    if (btnCompare) {
        btnCompare.disabled = !(filesLoaded.a && filesLoaded.b);
    }
}

export function updateScriptCompareButton(scriptFilesLoaded, btnCompareScripts) {
    if (btnCompareScripts) {
        btnCompareScripts.disabled = !(scriptFilesLoaded.raw && scriptFilesLoaded.cleaned);
    }
}

export function displayResults(data, diffSegments, currentDiffIndex) {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
        
        // Statistics
        document.getElementById('stat-words-a').textContent = data.statistics.total_words_a;
        document.getElementById('stat-words-b').textContent = data.statistics.total_words_b;
        document.getElementById('stat-matching').textContent = data.statistics.matching_words;
        document.getElementById('stat-similarity').textContent = data.statistics.similarity_ratio + '%';
        document.getElementById('stat-wer').textContent = data.statistics.word_error_rate + '%';
        
        const totalDiffs = data.statistics.inserted_words +
                          data.statistics.deleted_words +
                          data.statistics.replaced_words_a;
        document.getElementById('stat-diffs').textContent = totalDiffs;
        
        // Transcript content
        document.getElementById('content-a').innerHTML = data.html_a;
        document.getElementById('content-b').innerHTML = data.html_b;
        document.getElementById('count-a').textContent = `${data.statistics.total_words_a} words`;
        document.getElementById('count-b').textContent = `${data.statistics.total_words_b} words`;
        
        // Diff segments for navigation
        diffSegments.length = 0;
        diffSegments.push(...data.diff_segments);
        currentDiffIndex.value = -1;
        document.getElementById('total-diffs').textContent = diffSegments.length;
        document.getElementById('current-diff').textContent = '0';
        
        // Analysis lists
        displaySubstitutions(data.common_substitutions);
        displaySimilarPairs(data.similar_word_pairs);
        displayUniqueWords('unique-a-list', data.unique_to_a, 'danger');
        displayUniqueWords('unique-b-list', data.unique_to_b, 'success');
        
        // Detailed analysis
        displayDetailedAnalysis(data);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

export function displayScriptResults(data, turnTimeline, currentPlayingTurnIndex, turnComparisons) {
    const scriptResultsSection = document.getElementById('script-results-section');
    if (scriptResultsSection) {
        scriptResultsSection.classList.remove('hidden');
        
        // Store timeline data for audio sync
        turnTimeline.length = 0;
        turnTimeline.push(...(data.turn_timeline || []));
        currentPlayingTurnIndex.value = -1;
        
        // LLM Improvement Score
        const score = data.llm_metrics.overall_improvement_score;
        document.getElementById('improvement-score').textContent = score.score;
        document.getElementById('improvement-label').textContent = score.label;
        document.getElementById('improvement-desc').textContent = score.description;
        
        // LLM Metrics
        const filler = data.llm_metrics.filler_reduction;
        document.getElementById('metric-filler-reduction').textContent = filler.reduction;
        document.getElementById('metric-filler-detail').textContent = `${filler.raw_count} → ${filler.cleaned_count} (${filler.reduction_percentage}% reduction)`;
        
        const stutter = data.llm_metrics.stutter_reduction;
        document.getElementById('metric-stutter-reduction').textContent = stutter.reduction;
        document.getElementById('metric-stutter-detail').textContent = `${stutter.raw_count} → ${stutter.cleaned_count}`;
        
        const turns = data.llm_metrics.turn_consolidation;
        document.getElementById('metric-turns-merged').textContent = turns.turns_merged;
        document.getElementById('metric-turns-detail').textContent = `${turns.raw_turns} → ${turns.cleaned_turns} turns`;
        
        const mods = data.llm_metrics.modification_summary;
        document.getElementById('metric-modification-rate').textContent = mods.modification_rate + '%';
        document.getElementById('metric-mod-detail').textContent = `+${mods.words_added} / -${mods.words_removed} / ~${mods.words_replaced}`;
        
        // Statistics
        document.getElementById('script-stat-words-raw').textContent = data.statistics.total_words_raw;
        document.getElementById('script-stat-words-cleaned').textContent = data.statistics.total_words_cleaned;
        document.getElementById('script-stat-turns-raw').textContent = data.statistics.total_turns_raw;
        document.getElementById('script-stat-turns-cleaned').textContent = data.statistics.total_turns_cleaned;
        document.getElementById('script-stat-similarity').textContent = data.statistics.similarity_ratio + '%';
        document.getElementById('script-stat-wer').textContent = data.statistics.word_error_rate + '%';
        
        // Transcript content
        document.getElementById('script-content-raw').innerHTML = data.html_raw;
        document.getElementById('script-content-cleaned').innerHTML = data.html_cleaned;
        document.getElementById('script-count-raw').textContent = `${data.statistics.total_words_raw} words`;
        document.getElementById('script-count-cleaned').textContent = `${data.statistics.total_words_cleaned} words`;
        
        // Turn-by-turn comparison
        turnComparisons.length = 0;
        turnComparisons.push(...data.turn_comparisons);
        displayTurnComparisons(turnComparisons);
        
        // Analysis lists
        displayScriptSubstitutions(data.common_substitutions);
        displayScriptSimilarPairs(data.similar_word_pairs);
        displayScriptUniqueWords('script-unique-raw-list', data.unique_to_raw, 'danger');
        displayScriptUniqueWords('script-unique-cleaned-list', data.unique_to_cleaned, 'success');
        
        // Scroll to results
        scriptResultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

export function displaySubstitutions(substitutions) {
    const list = document.getElementById('substitutions-list');
    list.innerHTML = substitutions.map(s => `
        <li>
            <div class="substitution-item">
                <span class="word-a">${escapeHtml(s.word_a)}</span>
                <span class="arrow">→</span>
                <span class="word-b">${escapeHtml(s.word_b)}</span>
            </div>
            <span>${s.count}x</span>
        </li>
    `).join('');
}

export function displaySimilarPairs(pairs) {
    const list = document.getElementById('similar-pairs-list');
    if (!pairs || pairs.length === 0) {
        list.innerHTML = '<li style="color: var(--text-secondary);">No similar word pairs found</li>';
        return;
    }
    
    list.innerHTML = pairs.map(p => {
        const simClass = p.similarity >= 80 ? 'high' : 'medium';
        const typeLabel = {
            'single_char': '1 char',
            'double_char': '2 chars',
            'suffix_diff': 'suffix',
            'prefix_diff': 'prefix',
            'homophone': 'homophone',
            'similar_sound': 'sound',
            'other': ''
        }[p.likely_type] || '';
        
        return `
            <li>
                <div class="substitution-item">
                    <span class="word-a">${escapeHtml(p.word_a)}</span>
                    <span class="arrow">↔</span>
                    <span class="word-b">${escapeHtml(p.word_b)}</span>
                    <span class="similarity-badge ${simClass}">${p.similarity}%</span>
                    ${typeLabel ? `<span class="type-badge">${typeLabel}</span>` : ''}
                </div>
                <span>${p.count}x</span>
            </li>
        `;
    }).join('');
}

export function displayUniqueWords(listId, words, colorClass) {
    const list = document.getElementById(listId);
    list.innerHTML = words.map(w => `
        <li>
            <span>${w.word}</span>
            <span>${w.count}x</span>
        </li>
    `).join('');
}

export function displayScriptSubstitutions(substitutions) {
    const list = document.getElementById('script-substitutions-list');
    list.innerHTML = substitutions.map(s => `
        <li>
            <div class="substitution-item">
                <span class="word-a">${escapeHtml(s.word_a)}</span>
                <span class="arrow">→</span>
                <span class="word-b">${escapeHtml(s.word_b)}</span>
            </div>
            <span>${s.count}x</span>
        </li>
    `).join('');
}

export function displayScriptSimilarPairs(pairs) {
    const list = document.getElementById('script-similar-pairs-list');
    if (!pairs || pairs.length === 0) {
        list.innerHTML = '<li style="color: var(--text-secondary);">No similar word pairs found</li>';
        return;
    }
    
    list.innerHTML = pairs.map(p => {
        const simClass = p.similarity >= 80 ? 'high' : 'medium';
        return `
            <li>
                <div class="substitution-item">
                    <span class="word-a">${escapeHtml(p.word_a)}</span>
                    <span class="arrow">↔</span>
                    <span class="word-b">${escapeHtml(p.word_b)}</span>
                    <span class="similarity-badge ${simClass}">${p.similarity}%</span>
                </div>
                <span>${p.count}x</span>
            </li>
        `;
    }).join('');
}

export function displayScriptUniqueWords(listId, words, colorClass) {
    const list = document.getElementById(listId);
    if (!words || words.length === 0) {
        list.innerHTML = '<li style="color: var(--text-secondary);">None</li>';
        return;
    }
    list.innerHTML = words.map(w => `
        <li>
            <span>${w.word}</span>
            <span>${w.count}x</span>
        </li>
    `).join('');
}

export function displayTurnComparisons(turns) {
    const container = document.getElementById('turn-list');
    container.innerHTML = turns.map((turn, index) => {
        const similarity = turn.turn_similarity || 100;
        const simClass = similarity >= 95 ? '' : (similarity >= 80 ? 'medium' : 'low');
        const cardClass = turn.has_match && similarity < 95 ? 'changed' : (turn.has_match ? 'unchanged' : 'changed');
        const scriptAudioPlayer = document.getElementById('script-audio-player');
        const hasAudio = scriptAudioPlayer && scriptAudioPlayer.src && scriptAudioPlayer.src !== window.location.href;
        
        return `
            <div class="turn-card ${cardClass}" 
                 data-turn-index="${index}"
                 data-timestamp="${turn.timestamp}"
                 data-similarity="${similarity}" 
                 data-changed="${similarity < 95}"
                 onclick="seekAudioToTurn(${turn.timestamp})"
                 title="Click to jump to ${turn.timestamp.toFixed(2)}s in audio">
                <div class="turn-header">
                    <div>
                        <span class="turn-speaker">${escapeHtml(turn.speaker)}</span>
                        <span class="turn-timestamp">(${turn.timestamp.toFixed(2)}s)</span>
                    </div>
                    <div class="audio-indicator">
                        <span class="play-icon">${hasAudio ? '▶' : ''}</span>
                    </div>
                    ${turn.has_match ? `<span class="turn-similarity ${simClass}">${similarity}% match</span>` : '<span class="turn-similarity low">No match</span>'}
                </div>
                <div class="turn-content">
                    <div class="turn-raw">
                        <div class="turn-label">🔴 Raw</div>
                        <div class="turn-text">${turn.html_raw || escapeHtml(turn.raw_text)}</div>
                    </div>
                    <div class="turn-cleaned">
                        <div class="turn-label">✨ Cleaned</div>
                        <div class="turn-text">${turn.html_cleaned || (turn.cleaned_text ? escapeHtml(turn.cleaned_text) : '<em style="color: var(--text-secondary)">No matching turn</em>')}</div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

export function filterTurns(filter) {
    const cards = document.querySelectorAll('.turn-card');
    cards.forEach(card => {
        const isChanged = card.dataset.changed === 'true';
        if (filter === 'all') {
            card.style.display = '';
        } else if (filter === 'changed') {
            card.style.display = isChanged ? '' : 'none';
        } else if (filter === 'unchanged') {
            card.style.display = isChanged ? 'none' : '';
        }
    });
}

export function displayDetailedAnalysis(data) {
    const analysisA = data.analysis_a;
    const analysisB = data.analysis_b;
    const comparison = data.analysis_comparison;
    
    // Display repetitions
    displayRepetitions(analysisA, 'a');
    displayRepetitions(analysisB, 'b');
    
    // Display filler words
    displayFillers(analysisA, 'a');
    displayFillers(analysisB, 'b');
    
    // Update filler comparison stats
    document.getElementById('filler-pct-a').textContent = analysisA.filler_percentage + '%';
    document.getElementById('filler-pct-b').textContent = analysisB.filler_percentage + '%';
    document.getElementById('filler-total-a').textContent = analysisA.total_filler_count;
    document.getElementById('filler-total-b').textContent = analysisB.total_filler_count;
    
    // Display word frequency
    displayFrequency(analysisA, 'a');
    displayFrequency(analysisB, 'b');
    
    // Update frequency comparison stats
    document.getElementById('unique-count-a').textContent = analysisA.unique_words;
    document.getElementById('unique-count-b').textContent = analysisB.unique_words;
    document.getElementById('vocab-richness-a').textContent = analysisA.vocabulary_richness + '%';
    document.getElementById('vocab-richness-b').textContent = analysisB.vocabulary_richness + '%';
    document.getElementById('avg-length-a').textContent = analysisA.avg_word_length;
    document.getElementById('avg-length-b').textContent = analysisB.avg_word_length;
    
    // Display phrases
    displayPhrases(analysisA, 'a');
    displayPhrases(analysisB, 'b');
    
    // Display comparison stats
    document.getElementById('cmp-words-a').textContent = analysisA.total_words;
    document.getElementById('cmp-words-b').textContent = analysisB.total_words;
    document.getElementById('cmp-chars-a').textContent = analysisA.total_characters.toLocaleString();
    document.getElementById('cmp-chars-b').textContent = analysisB.total_characters.toLocaleString();
    document.getElementById('cmp-stutters-a').textContent = comparison.repetition_comparison.stutters_a;
    document.getElementById('cmp-stutters-b').textContent = comparison.repetition_comparison.stutters_b;
    document.getElementById('cmp-phrases-a').textContent = comparison.repetition_comparison.repeated_phrases_a;
    document.getElementById('cmp-phrases-b').textContent = comparison.repetition_comparison.repeated_phrases_b;
    
    // Generate summary insights
    displaySummaryInsights(data);
}

export function displayRepetitions(analysis, which) {
    // Consecutive repetitions (stutters)
    const stuttersList = document.getElementById(`stutters-${which}`);
    if (analysis.consecutive_repetitions.length > 0) {
        stuttersList.innerHTML = analysis.consecutive_repetitions.map(r => `
            <li class="repetition-item stutter">
                <span class="repetition-text">"${escapeHtml(r.text)}" × ${r.count}</span>
                <span class="repetition-badge">pos ${r.position}</span>
            </li>
        `).join('');
    } else {
        stuttersList.innerHTML = '<li style="color: var(--text-secondary); padding: 10px;">No stutters detected</li>';
    }
    
    // Repeated words
    const repeatedList = document.getElementById(`repeated-words-${which}`);
    if (analysis.repeated_words.length > 0) {
        repeatedList.innerHTML = analysis.repeated_words.slice(0, 10).map(r => `
            <li class="repetition-item">
                <span class="repetition-text">"${escapeHtml(r.text)}"</span>
                <span class="repetition-badge">${r.count}×</span>
            </li>
        `).join('');
    } else {
        repeatedList.innerHTML = '<li style="color: var(--text-secondary); padding: 10px;">No notable repetitions</li>';
    }
    
    // Update count
    const totalReps = analysis.consecutive_repetitions.length + analysis.repeated_words.length;
    document.getElementById(`repetition-count-${which}`).textContent = `${totalReps} found`;
}

export function displayFillers(analysis, which) {
    const container = document.getElementById(`fillers-${which}`);
    const fillers = Object.entries(analysis.filler_words);
    
    if (fillers.length > 0) {
        container.innerHTML = fillers.map(([word, count]) => `
            <div class="filler-chip">
                <span class="word">${escapeHtml(word)}</span>
                <span class="count">${count}×</span>
            </div>
        `).join('');
    } else {
        container.innerHTML = '<p style="color: var(--text-secondary); padding: 10px;">No filler words detected</p>';
    }
}

export function displayFrequency(analysis, which) {
    const container = document.getElementById(`frequency-${which}`);
    const maxCount = analysis.word_frequency.length > 0 ? analysis.word_frequency[0].count : 1;
    
    container.innerHTML = analysis.word_frequency.slice(0, 15).map(item => {
        const percentage = (item.count / maxCount) * 100;
        return `
            <div class="frequency-bar">
                <span class="frequency-word">${escapeHtml(item.word)}</span>
                <div class="frequency-bar-container">
                    <div class="frequency-bar-fill" style="width: ${percentage}%"></div>
                </div>
                <span class="frequency-count">${item.count}</span>
            </div>
        `;
    }).join('');
}

export function displayPhrases(analysis, which) {
    // Bigrams
    const bigramsList = document.getElementById(`bigrams-${which}`);
    if (analysis.bigrams.length > 0) {
        bigramsList.innerHTML = analysis.bigrams.map(p => `
            <li class="phrase-item">
                <span class="phrase-text">"${escapeHtml(p.phrase)}"</span>
                <span class="phrase-count">${p.count}×</span>
            </li>
        `).join('');
    } else {
        bigramsList.innerHTML = '<li style="color: var(--text-secondary); padding: 10px;">No repeated bigrams</li>';
    }
    
    // Trigrams
    const trigramsList = document.getElementById(`trigrams-${which}`);
    if (analysis.trigrams.length > 0) {
        trigramsList.innerHTML = analysis.trigrams.map(p => `
            <li class="phrase-item">
                <span class="phrase-text">"${escapeHtml(p.phrase)}"</span>
                <span class="phrase-count">${p.count}×</span>
            </li>
        `).join('');
    } else {
        trigramsList.innerHTML = '<li style="color: var(--text-secondary); padding: 10px;">No repeated trigrams</li>';
    }
}

export function displaySummaryInsights(data) {
    const container = document.getElementById('summary-insights');
    const stats = data.statistics;
    const analysisA = data.analysis_a;
    const analysisB = data.analysis_b;
    const comparison = data.analysis_comparison;
    
    const insights = [];
    
    // Similarity insight
    if (stats.similarity_ratio >= 90) {
        insights.push(`<p>✅ <strong>High similarity</strong> (${stats.similarity_ratio}%) - The transcripts are very similar.</p>`);
    } else if (stats.similarity_ratio >= 70) {
        insights.push(`<p>⚠️ <strong>Moderate similarity</strong> (${stats.similarity_ratio}%) - Some notable differences exist.</p>`);
    } else {
        insights.push(`<p>🔴 <strong>Low similarity</strong> (${stats.similarity_ratio}%) - The transcripts differ significantly.</p>`);
    }
    
    // Word count difference
    const wordDiff = Math.abs(stats.total_words_a - stats.total_words_b);
    const wordDiffPct = ((wordDiff / Math.max(stats.total_words_a, stats.total_words_b)) * 100).toFixed(1);
    if (wordDiff > 50) {
        const longer = stats.total_words_a > stats.total_words_b ? 'A' : 'B';
        insights.push(`<p>📊 Transcript ${longer} has ${wordDiff} more words (${wordDiffPct}% difference).</p>`);
    }
    
    // Filler word comparison
    const fillerDiff = Math.abs(comparison.filler_comparison.percentage_a - comparison.filler_comparison.percentage_b);
    if (fillerDiff > 1) {
        const moreFiller = comparison.filler_comparison.percentage_a > comparison.filler_comparison.percentage_b ? 'A' : 'B';
        insights.push(`<p>🗣️ Transcript ${moreFiller} has more filler words (${comparison.filler_comparison.percentage_a}% vs ${comparison.filler_comparison.percentage_b}%).</p>`);
    }
    
    // Stutter detection
    const totalStuttersA = comparison.repetition_comparison.stutters_a;
    const totalStuttersB = comparison.repetition_comparison.stutters_b;
    if (totalStuttersA > 0 || totalStuttersB > 0) {
        insights.push(`<p>🔁 Consecutive repetitions (stutters): A has ${totalStuttersA}, B has ${totalStuttersB}.</p>`);
    }
    
    // Vocabulary richness
    const richnessDiff = Math.abs(comparison.vocabulary_comparison.richness_a - comparison.vocabulary_comparison.richness_b);
    if (richnessDiff > 5) {
        const richer = comparison.vocabulary_comparison.richness_a > comparison.vocabulary_comparison.richness_b ? 'A' : 'B';
        insights.push(`<p>📚 Transcript ${richer} has richer vocabulary diversity.</p>`);
    }
    
    // Error analysis
    if (stats.word_error_rate > 20) {
        insights.push(`<p>⚠️ <strong>High word error rate</strong> (${stats.word_error_rate}%) suggests significant transcription differences.</p>`);
    }
    
    // Substitution patterns
    if (data.common_substitutions.length > 0) {
        const topSub = data.common_substitutions[0];
        insights.push(`<p>🔄 Most common substitution: "${topSub.word_a}" → "${topSub.word_b}" (${topSub.count} times).</p>`);
    }
    
    if (insights.length === 0) {
        insights.push('<p style="color: var(--text-secondary);">No significant patterns detected.</p>');
    }
    
    container.innerHTML = insights.join('');
}