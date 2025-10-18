import pandas as pd
import oral_interview_utils

def segment_speakers(testimonies):
    testimonies['lines'] = testimonies['content'].apply(oral_interview_utils.get_speaker_lines)
    testimonies = testimonies.drop(columns=['content'])
    testimonies = testimonies.explode('lines')
    testimonies[['speakers_segment', 'speaker', 'line']] = testimonies['lines'].apply(lambda x: pd.Series(x))
    testimonies = testimonies.drop(columns=['lines'])
    return testimonies[['testimony_id', 'speakers_segment', 'speaker', 'line']].reset_index(drop=True)

def segment_micro(testimony, min_words=250):
    seg_ix = 0
    segments = []
    segments_txt = ''
    for i in range(0, len(testimony), 2):
        if i+1 >= len(testimony):
            segments += [seg_ix]
            break

        segments += [seg_ix] * 2
        q = testimony.iloc[i]
        a = testimony.iloc[i+1]
        segments_txt += f"{q['line']}\n{a['line']}\n"
        if len(segments_txt.strip().split()) >= min_words:
            seg_ix += 1
            segments_txt = ''

    return segments

def segment_micro_testimonies(testimonies, min_words=250):
    return (
        testimonies
        .groupby('testimony_id', group_keys=False)
        .apply(lambda g: pd.Series(segment_micro(g, min_words=min_words), index=g.index), include_groups=False)
    )

def segment_macro(base_segments, num_segments=15):
    max_segment = max(1, base_segments.max())
    pos_prec = base_segments.apply(lambda x: x / max_segment)
    pos_prec.apply(lambda x: 1 if str(x) == 'nan' else x)
    segments = pd.cut(pos_prec, bins=num_segments, labels=False)
    return segments.to_list()

def segment_macro_testimonies(testimonies, base_segment_col, num_segments=15):
    return (
        testimonies
        .groupby('testimony_id', group_keys=False)[base_segment_col]
        .apply(lambda g: pd.Series(segment_macro(g, num_segments=num_segments), index=g.index), include_groups=False)
    )