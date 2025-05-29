'''
This model grabs an audio file argument when run, and uses distilation of
OpenAI's Whisper, small version, to generate a transcript of the spoken English
in that audio file. It saves this transcript in 30 second chunks to a text file
along with timestamps for each segement.
'''
import os
import argparse

import tkinter as tk
from tkinter import filedialog


import re
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import numpy as np

def parse_whisper_segments(segments):
    """
    Parses Whisper segments into a list of timestamped text chunks.

    Args:
        segments (list): List of dictionaries with 'start', 'end', and 'text'.

    Returns:
        list: List of dictionaries with 'start', 'end', and 'text'.
    """
    timestamped_text = []
    for segment in segments:
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()

        timestamped_text.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    return timestamped_text

def format_timestamp(seconds):
    """
    Formats seconds into HH:MM:SS format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in HH:MM:SS format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def save_timestamped_transcript(start_stamp, end_stamp, text, output_path):
    """
    Saves the timestamped transcript to a text file.
    start time, in seconds, of section n is on line 3n-3
    transcript of section n on line 3n-2
    end time, in seconds, of section n is on line 3n-1

    Args:
        start_stamp: timestamp for the start time of text, formatted HH:MM:SS
        ens_stamp: timestamp for the end time of text, formatted HH:MM:SS
        text: string containing a section of the transcript
        output_path (str): Path to save the output file.
    """
    output_path.write(start_stamp + "\n")
    output_path.write(text + "\n")
    output_path.write(end_stamp + "\n")

# Path to local distilWhisper download
model_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(model_path):
    print("Error: The specified model_path does not exist!")

#load model and processor from file
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, local_files_only=True)

#define device properly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get audio file from stdin
input_text_parser = argparse.ArgumentParser()
input_text_parser.add_argument('text', type=str)
input_text_args = input_text_parser.parse_args()
audio_path = input_text_args.text

audio, sr = librosa.load(audio_path, sr=16000)
    #sr is how many samples per second.

#define segment length, start at 30 seconds each, decrease if necessary
segment_length = 30 * sr
num_segments = int(np.ceil(len(audio) / segment_length))    #get the numnber of segments

transcriptions = [] #empty list to fill within the loop

transcript_save_file = open('output/audio_output.txt', 'w', encoding='utf-8')
transcript_save_file.write("")  #empty out file so new transcript can be appended
#process and transcribe each chunk
for i in range(num_segments):

    start = i * segment_length  #beginning of segment i
    end = min((i + 1) * segment_length, len(audio)) #end of segment i
    chunk = audio[start:end]  #extract segment
    #divide start and end by sr to get start and end times in seconds

    #process chunk
    inputs = processor(chunk,   #
                       sampling_rate=16000,
                       return_tensor="pt"
    )

    attention_mask = torch.ones_like(torch.tensor(inputs.input_features))
        #ChatGPT recommendation for dummy attention mask
        #all ones, just here to clear the error for not having one

    input_features = torch.tensor(inputs.input_features).to(device)
        #pytorch tensor type, move to device

    #infer
    with torch.no_grad():
        predicted_ids = model.generate(input_features, attention_mask=attention_mask.to(device))

    #decode output, save to transcripts array
    single_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    single_transcript = re.sub(r"<\|.*?\|>", "", single_transcript).strip()
        #remove string formatted tokens from output

    start_timestamp = format_timestamp(start/sr)
    end_timestamp = format_timestamp(end/sr)
    save_timestamped_transcript(start_timestamp, end_timestamp, single_transcript, transcript_save_file)

transcript_save_file.close()
