"""Rank review windows from a broadcast's Pass 1 commentary cues."""

import re
from bisect import bisect_left

from .core import uid


SIGNALS = {
    'reasoning': r'\b(?:because|that(?:\x27s| is) why|which means|so that|in order to|allows? (?:him|her|them|you)|forces? (?:him|her|them))\b',
    'alternatives': r'\b(?:instead|otherwise|could have|should have|if they|if you|had to choose|trade[- ]off)\b',
    'execution': r'\b(?:timing|positioning|flank(?:ing)?|rotat\w*|cooldowns?|vision|engage|disengage|peel|spacing|setup|set up|power spike|win condition)\b',
    'league': r'\b(?:draft|ban phase|first pick|team\s?fight|lane(?:s| priority)?|minion wave|wave clear|baron|elder|drake|dragon|herald|grubs|inhibitor|turret|teleport|itemization|back\s?line)\b',
    'review': r'\b(?:here you can see|take a look|look at how|look at why|what happened|as we see|watch how)\b',
}
PATTERNS = {key: re.compile(pattern, re.I) for key, pattern in SIGNALS.items()}


def rank_commentary(cues, source_id, *, duration, length=30, stride=15,
                    interval=None, vocabulary=None):
    """Return explainable candidates with source time and original cue IDs."""
    cues = sorted(cues, key=lambda row: (row['start'], row['end']))
    starts = [row['start'] for row in cues]
    candidates = []
    lower, upper = interval or (0, duration)
    patterns = PATTERNS.copy()
    if vocabulary is not None:
        terms = sorted(set(vocabulary), key=lambda term: (-len(term), term))
        patterns['league'] = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, terms)) + r')\b' if terms else r'(?!)', re.I
        )
    for offset in range(0, int(upper - lower) + 1, stride):
        start = lower + offset
        end = min(start + length, upper)
        group = cues[bisect_left(starts, start):bisect_left(starts, end)]
        # Keep text in the retained cue artifact rather than duplicating it in every window.
        text = ' '.join(row['text'] for row in group).replace('’', "'")
        matches = {name: sorted({part.lower() for part in pattern.findall(text)})
                   for name, pattern in patterns.items()}
        domain = len(matches['league']) + len(matches['execution'])
        explanation = len(matches['reasoning']) + len(matches['alternatives'])
        if len(text.split()) < 20 or not domain or not explanation:
            continue
        score = 3 * min(explanation, 4) + min(domain, 6) + min(len(matches['review']), 2)
        candidates.append({'id': uid('broadcast-commentary', source_id, start, end),
                           'start': start, 'end': end, 'score': score,
                           'signals': {key: values for key, values in matches.items() if values},
                           'cue_ids': [row['id'] for row in group], 'review_status': 'unreviewed'})
    return sorted(candidates, key=lambda row: (-row['score'], row['start']))


def shortlist(candidates, limit, *, per_half_hour=3):
    """Diversify highly overlapping windows and avoid one game taking every slot."""
    selected, counts = [], {}
    for row in candidates:
        bucket = row['start'] // 1800
        if counts.get(bucket, 0) >= per_half_hour or any(
            item['start'] < row['end'] and row['start'] < item['end'] for item in selected
        ):
            continue
        selected.append(row)
        counts[bucket] = counts.get(bucket, 0) + 1
        if len(selected) == limit:
            break
    return selected
