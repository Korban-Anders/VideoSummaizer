'''
paragraph_breaker takes the contents of audio_output.txt and groups together sentences about
the same topic, while keeping the sentences' corresponding timestamps. Timestamps for the same
topic that line up (i.e. 1:30 to 2:00 and 2:00 to 2:30) are ocmbined (1:30 to 2:30). Each topic's
timestamps are sorted by frequency of occurance and all but the first three are dropped.

The purpose is to chunk up large strings such that they do not exceed PEGASUS'
token limit, and to combine sections about similar topics to improve summary clarity.

Output is text segments paired with lists of start and end timestamps, written to
segmented_output.txt.

dependencies:
    pip install sentence-transformers scikit-learn numpy
    pip install -U spacy coreferee
    python -m coreferee install en
'''

import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

def sequential_segmentation(sequential_sentences, sequential_clustering):
    '''
    Segments input string by topic.
    Maintains sentence order, if two sentences have the
    same topic but aren't consecutive, they're in different segments.

    This function is not used in the final version of this project.
    '''
    #loop through sentences' labels to identify topic changes where the cluster IDs change
    prev_label = sequential_clustering.labels_[0]
    sequential_segmented_string=[]
    sequential_segment = [sequential_sentences[0]]
    for i in range(1, len(sequential_sentences)):
        label = sequential_clustering.labels_[i]

        if label != prev_label and label != -1: #topic shift
            sequential_segmented_string.append(" ".join(sequential_segment))
            sequential_segment = []

        sequential_segment.append(sequential_sentences[i])
        prev_label = label

    sequential_segmented_string.append(" ".join(sequential_segment))#add final segment

    return sequential_segmented_string

def topical_segmentation(topical_sentences, topical_clustering):
    '''
    Segments input string by topic.
    topical_senteces is a 2d array:
        [n][0]: the sentence string
        [n][1]: HH:MM:SS start timestamp
        [n][2]: HH:MM:SS end timestamp
    in addition to grouping sentences of the same topic, the timestamps for that topic should be
    saved, with the most repeated timestamps first, and only the top three timestamp sets saved
    by the end
    '''

    topics_num = max(topical_clustering.labels_)
    topics = ["" for _ in range(0, topics_num)]
    topic_timestamps = [[] for _ in range(topics_num)]
    last_start = ""

    index_totals = [0 for _ in range(topics_num)]
    sentences_per_topic = [0 for _ in range(topics_num)]
    average_indices = [0 for _ in range(topics_num)]

    for i, sent in enumerate(topical_sentences):
        topic_index = topical_clustering.labels_[i] - 1
        if topic_index < 0:
            continue
        
        sent[0] += ". "
        print(f"topics lengths:\t{len(topics)}\ttopic_index:\t{topic_index}\n")
        topics[topic_index] += sent[0]  #geting out of bounds issues here

        if topic_timestamps[topic_index] == []:
            topic_timestamps[topic_index].append([topical_sentences[i][1], topical_sentences[i][2], 1])
            last_start = topical_sentences[i][1]
        elif topic_timestamps[topic_index][-1][0] == last_start:
            topic_timestamps[topic_index][-1][2] += 1
        else:
            topic_timestamps[topic_index].append([topical_sentences[i][1], topical_sentences[i][2], 1])
            last_start = topical_sentences[i][1]

        index_totals[topic_index] += i
        sentences_per_topic[topic_index] += 1
            #used to derive what order the topics should go in
    
    average_indices = [
        index_totals[i]/sentences_per_topic[i] if sentences_per_topic[i] != 0 else 0
        for i in range(topics_num)
    ]

    topics_with_ts = [[s, [], []] for s in topics]
    for i,tts in enumerate(topic_timestamps):
        #combine adjacent timestamps, set the end of the earlier one to the end of the later, and sum counts
        if len(tts) >= 2:
            for j in range(len(tts) -2, -1, -1):
                if tts[j][1] == tts[j+1][0]:
                    tts[j][1] = tts[j+1][1]
                    tts[j][2] += tts[j+1][2]
                    del tts[j+1]

        tts = sorted(tts, key=lambda x: x[2], reverse=True)
        tts = tts[:3]
        print(f"topic_timestamps[i] after:\t{tts}\n")
        topics_with_ts[i][1] = [entry[0] for entry in tts]
        topics_with_ts[i][2] = [entry[1] for entry in tts]



    return topics_with_ts

def process_file(input_file):
    '''
    Reads timestamps and transcript sections from input_file and returns segments[],
    which contains a section of transcript corresponding to the 30s rnge defined by
    the adjacent timestamps
    '''
    segments = []

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]

    for i in range(0, len(lines), 3):
        try:
            start = lines[i]
            text = lines[i + 1]
            end = lines[i + 2]

            segments.append({
                "start_timestamp": start,
                "text": text,
                "end_timestamp": end
            })
        except IndexError:
            print(f"Skipping incomplete segment at lines {i}-{i+2}")

    return segments

input_file = "output/audio_output.txt"
segments = process_file(input_file)

#load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')#light SBERT model

#new sentence splitting block, maintain pairing with timestamps
sentences = []
stamped_sentences = []
for seg in segments:
    text = seg["text"]
    metadata = [seg["start_timestamp"], seg["end_timestamp"]]  #store timestamps
    split_sentences = text.split('. ')
    for s in split_sentences:
        s = s.strip()
        sentences.append(s)
        stamped_sentences.append([s] + metadata)

#embed and then cluster sentences
embeddings = model.encode(sentences, convert_to_numpy=True)
clustering = DBSCAN(eps=0.7, min_samples=1, metric='cosine').fit(embeddings)
'''
eps: max distance between points in the same cluster
    smaller eps: smaller clusters, more specific topics
    default: 0.5
    too high: 0.9, only one topic with every sentence
    too low: 0.1, too many sentences in no group, -1
min_samples: minimum number of points to form a region
    lower min_samples: more sensitive to small topic shifts
    increasing min_samples means fewer topics, with more sentences per topic
    default: 2
'''

print("Unique cluster labels:", set(clustering.labels_))
print("Homeless (noise) sentences:", sum(1 for l in clustering.labels_ if l == -1))
print("Total sentences:", len(clustering.labels_))

test_array = []
homeless_sentences = 0
for clust in clustering.labels_:
    if clust == -1:
        homeless_sentences += 1
        continue
    if len(test_array) <= clust:
        test_array.append(1)
    else:
        test_array[clust] += 1

topics = topical_segmentation(stamped_sentences, clustering)

seg_sent_save_file = open("output/segmented_output.txt", 'w', encoding='utf-8')
for top in topics:
    seg_sent_save_file.write(str(top[1]) + "\n")
    seg_sent_save_file.write(top[0] + "\n")
    seg_sent_save_file.write(str(top[2]) + "\n")
seg_sent_save_file.close()
