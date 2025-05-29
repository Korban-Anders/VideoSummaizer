'''
pegasus_test.py reads sets of start timestamps, segements of text, and end timestamps
from segmented_output.txt. It uses a distilation of Google's PEGASUS, small
version, to summarize each text segment and then write them, along with their corresponding
timestamps, to summary_output.txt
'''

import os
import argparse
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def chunk_tensor(tensor, chunk_size):
    tensor = tensor.squeeze(0)
    return [tensor[i:i+chunk_size] for i in range(0, tensor.size(0), chunk_size)]
#
def summarize(sum_text, chunk_size=1024):
    '''
    use distilled PEGASUS model to summarize sum_text
    '''
    #tokenize
    tokens = tokenizer(sum_text, return_tensors="pt", truncation=False)
    chunks = chunk_tensor(tokens['input_ids'], chunk_size)

    summaries = []

    for chunk in chunks:
        input_dict = {
            'input_ids': chunk.unsqueeze(0),
            'attention_mask': (chunk != tokenizer.pad_token_id).unsqueeze(0)
        }
        summary_ids = model.generate(**input_dict, no_repeat_ngram_size=5)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

def read_input():
    """
    reads text segments and both start and end timestamps from segmented_output.txt
    """
    with open("segmented_output.txt", 'r', encoding='utf-8') as input_file:
        lines = [line.strip() for line in input_file]

    start_stamps = []
    text_segments = []
    end_stamps = []

    for i in range(0, len(lines), 3):
        start_stamps.append(lines[i])
        text_segments.append(lines[i+1])
        end_stamps.append(lines[i+2])

    return start_stamps, text_segments, end_stamps

start_stamps, text_topics, end_stamps = read_input()

# Path to your local model directory
local_model_path = os.path.dirname(os.path.abspath(__file__))

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained(local_model_path)
model = PegasusForConditionalGeneration.from_pretrained(local_model_path)

summary = []

for topic in text_topics:
    summary.append(summarize(topic))

with open("output/summary_output.txt", 'w', encoding='utf-8') as output_file:
    for i, summ in enumerate(summary):
        output_file.write(f"{start_stamps[i]}\n{summ}\n{end_stamps[i]}\n")
