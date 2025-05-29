'''
combo_handler.py is no longer used in latest version of the video summarization project.
Keeping this script just in case.

This script calls whisper_test.py to process an audio clip from the
user and turn it into a text transcript.
It then calls pegasus_test.py to turn that transcript into a summary, which is printed to stdout.
'''
import subprocess
import argparse

#get audio file path from stdin
input_text_parser = argparse.ArgumentParser()
input_text_parser.add_argument('text', type=str)
input_text_args = input_text_parser.parse_args()
audio_file_path = input_text_args.text

#run whisper_test.py, get printed transcript from audio
whisper_test_result = subprocess.run(['python', 'models/distilWhisper/whisper_test.py', audio_file_path],
                                      capture_output=True,
                                      text=True,
                                      check=True)
transcript = whisper_test_result.stdout.strip()

#print(f"Audio Transcript:\n{transcript}")

#print(f"Transcript:\n{transcript}\n")

#convert transcript to segmented_transcript: an array of strings that divides transcript at topic changes
paragraph_breaker_results = subprocess.run(['python', 'scripts/paragraph_breaker.py', transcript, '--topical_summary'],
                                           capture_output=True,
                                           text=True,
                                           check=True)
segmented_transcript = paragraph_breaker_results.stdout.strip()

#print(f"Segmented Transcript:\n{segmented_transcript}\n")

#pass segmented_transcript to pegasus_test.py and get summary
pegasus_test_result = subprocess.run(['python', 'models/distilPegasus/pegasus_test.py', segmented_transcript],
                                      check=True,
                                      capture_output=True,
                                      text=True)
summary = pegasus_test_result.stdout.strip()
print(f"Audio Summary: \n {summary}")
