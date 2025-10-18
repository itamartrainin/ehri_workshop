import re
import pandas as pd

from tqdm import tqdm
from litellm import completion, token_counter, get_max_tokens

import segmentation
import oral_interview_utils

tqdm.pandas()

NUM_LLM_RETIRES = 10

get_topic_prompt = """
You are a Holocaust researcher. You will be presented with a short text snippet from a conversation between an interviewer and a Holocaust survivor.
'INT' represents the interviewer and 'SUBJECT' or '<survivor_name>' represents the survivor.

Given the text snippet please generate a short title describing the most prominent topic in the text.
Make sure that the title is short and limited to a few words.
Make sure that the title is comprehensive, specific, interpretable, and short.
Make sure that the title captures only a single topic.

Output format:
Title: "<title>"
Reason: "<reason>"

Text snippet:
"{text_snippet}"
"""

get_common_topics_prompt = """
You are a Holocaust researcher. You will be presented with a set of titles representing topics extracted from Holocaust survivor interviews.

Title Set:
{title_set}

Your task is:
- Generate {num_topics} distinct titles that best describe the most common and prominent titles in set.
- Titles must be concise (maximum of a few words), specific, interpretable, and distinct.
- Do NOT combine multiple topics into a single title or use conjunctions like “and”.

Desired output format:
{output_format}

The common titles are:
1.
"""

def get_single_topic(text, model_name='gpt-5'):
    """
    Get a single title for a given text snippet.
    """
    for _ in range(NUM_LLM_RETIRES):
        try:
            prompt = get_topic_prompt.format(text_snippet=text)
            resp = completion(
                model=model_name,
                messages= [{'role': 'user', 'content': prompt}]
            )
            resp = resp['choices'][0]['message']['content']
            topic = re.findall('Title: "(.*)"', resp)

            if len(topic) == 0:
                raise Exception('No title found.')

            topic = topic[0]

            return topic

        except Exception as e:
            print(f"Failed to get titles: {e} ; retrying...")
            continue

    return None

def get_common_topics(topics, num_output_topics, model_name='gpt-5'):
    """
    Given a list of q/a titles, map-reduces the list to the number of num_output_titles COMMON titles.
    """
    depth = 0
    while len(topics) > num_output_topics:
        print(f"Depth: {depth} ; Num titles: {len(topics)} ; Num output titles: {num_output_topics}")

        buckets = split_into_buckets(model_name, topics, num_output_topics)

        reduced_titles = []
        for bucket in buckets:
            common_titles = get_common_topics_from_bucket(model_name, topics, num_output_topics)

            if common_titles:
                reduced_titles += common_titles
            else:
                print(f"Failed to get common topics for bucket: {bucket}\nReturning None for set.")
                return None

        topics = reduced_titles
        depth += 1

    return topics

def has_exceeded_context_size(model_name, messages, reserve_for_output=5096):
    n_tokens = token_counter(model=model_name, messages=messages)
    max_tokens = get_max_tokens(model_name) - reserve_for_output
    return n_tokens > max_tokens

def compose_common_topics_prompt(topics, num_output_topics):
    titles_str = '\n'.join([f'{i + 1}. "{title}"' for i, title in enumerate(topics)])
    output_format = '\n'.join([f'{i + 1}. "<title{i + 1}>"' for i in range(num_output_topics)])
    prompt = get_common_topics_prompt.format(title_set=titles_str, num_topics=num_output_topics, output_format=output_format)
    return prompt

def split_into_buckets(model_name, topics, num_output_topics):
    """
    Given a list of q/a titles, this method splits the list into buckets that fit into the model's context size.
    """
    buckets = []
    current_bucket = []
    for title in tqdm(topics, desc=f'Splitting into buckets ({num_output_topics})'):
        has_exceeded = has_exceeded_context_size(
            model_name=model_name,
            messages=[{
                'role': 'user',
                'content': compose_common_topics_prompt(current_bucket + [title], num_output_topics)
            }]
        )
        if has_exceeded:
            buckets.append(current_bucket)
            current_bucket = [title]
        else:
            current_bucket.append(title)
    buckets.append(current_bucket)
    return buckets

def get_common_topics_from_bucket(model_name, topics, num_output_topics):
    for _ in range(NUM_LLM_RETIRES):
        try:
            prompt = compose_common_topics_prompt(topics, num_output_topics)
            resp = completion(
                model=model_name,
                messages= [{'role': 'user', 'content': prompt}]
            )
            resp = resp['choices'][0]['message']['content']
            output = re.findall(r'[0-9]+. "(.*)"', resp)

            if len(output) != num_output_topics:
                raise Exception('Output is not of the requested length.')

            return output
        except Exception as e:
            print(f"Failed with error: {e} ; retrying...")
            continue

    return None

def extract_topics(testimonies, num_common_topics=15, model_name='gpt-5'):
    qa_topics = (
        testimonies
        .groupby(['testimony_id', 'micro_segment'])
        .progress_apply(
            lambda lines: get_single_topic(
                oral_interview_utils.join_speaker_lines(lines),
                model_name=model_name
            )
        )
        .reset_index()
        .rename(columns={0: 'topic'})
    )

    qa_topics['micro_macro_segment'] = (
        qa_topics
        .groupby('testimony_id', group_keys=False)['micro_segment']
        .apply(lambda g: pd.Series(segmentation.segment_macro(g), index=g.index), include_groups=False)
    )

    common_topics = (
        qa_topics
        .groupby(['micro_macro_segment'])['topic']
        .progress_apply(lambda x: get_common_topics(x, num_common_topics))
    )

    return pd.DataFrame(common_topics.tolist())







