'''
pegasus_test.py reads sets of start timestamps, segements of text, and end timestamps
from segmented_output.txt. It uses a distilation of Google's PEGASUS-X to summarize each text segment and then write them, along with their corresponding
timestamps, to summary_output.txt
'''

import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def chunk_tensor(tensor, chunk_size):
    tensor = tensor.squeeze(0)
    return [tensor[i:i+chunk_size] for i in range(0, tensor.size(0), chunk_size)]

def summarize(sum_text, chunk_size=1024):
    '''
    use Pegasus to summarize sum_text
    '''
    '''
    #tokenize
    tokens = tokenizer(sum_text, return_tensors="pt", truncation=False, padding = True)
    chunks = chunk_tensor(tokens['input_ids'], chunk_size)

    summaries = []

    for chunk in chunks:
        input_ids = chunk.unsqueeze(0),
        attention_mask = (chunk != tokenizer.pad_token_id).unsqueeze(0)
        
        summary_ids = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            no_repeat_ngram_size=5
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return " ".join(summaries)
    '''
    tokens = tokenizer(sum_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    summary_ids = model.generate(
        input_ids = input_ids,
        attention_mask=attention_mask,
        no_repeat_ngram_size=5
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def read_input():
    """
    reads text segments and both start and end timestamps from segmented_output.txt
    """
    with open("output/segmented_output.txt", 'r', encoding='utf-8') as input_file:
        lines = [line.strip() for line in input_file]

    start_stamps = []
    text_segments = []
    end_stamps = []

    for i in range(0, len(lines), 3):
        start_stamps.append(lines[i])
        text_segments.append(lines[i+1])
        end_stamps.append(lines[i+2])

    return start_stamps, text_segments, end_stamps

# Model that is used in this case it is 
model_name = "google/pegasus-large"

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Read and summarize input
start_stamps, text_topics, end_stamps = read_input()
summary = [summarize(topic) for topic in text_topics]

os.makedirs("output", exist_ok=True)
with open("output/summary_output.txt", 'w', encoding='utf-8') as output_file:
    for i, summ in enumerate(summary):
        output_file.write(f"{start_stamps[i]}\n{summ}\n{end_stamps[i]}\n")
