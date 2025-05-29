# timestamps for video summarizer

def video_model_ocr_with_timestamps(video_path):
    screen_reader = easyocr.Reader(['en']) #specify the language(s) and sets up the reader
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    text_occurrences = {}

# Loops through frames
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        timestamp = frame_count / fps  # timestamp in seconds
        # Text recognition on the frame
        result = screen_reader.readtext(frame)
        frame_count += 1

        for detection in result:
            detected_text = detection[1]
            if detected_text not in text_occurrences:
                text_occurrences[detected_text] = []
            text_occurrences[detected_text].append(timestamp)

    video.release()

    # Filter texts that appear frequently and summarize timestamps
    filtered_texts_with_timestamps = {}
    for text, timestamps in text_occurrences.items():
        if len(timestamps) >= 20:  # need to adjust based on video length
            filtered_texts_with_timestamps[text] = timestamps

    # Pretty print or return structured output
    for text, timestamps in filtered_texts_with_timestamps.items():
        print(f"'{text}' appeared at: {[round(t, 2) for t in timestamps]}")

    return filtered_texts_with_timestamps

# notes to self
# need to work on integration of this with the main and adjust number of timestamps, maybe user adjustable?