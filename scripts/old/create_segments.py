import os
import glob
import time
import librosa
import numbers
import datetime
import pandas as pd
import soundfile as sf

NEMO_ROOT = os.path.dirname(os.path.dirname(__file__))

def _get_seconds_from_time(t):
    x = time.strptime(t, '%H:%M:%S')
    seconds = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    return seconds


def create_segments(file_names):
    df_lengths = 0
    for file_name in file_names:

        audio_path = f'{NEMO_ROOT}/recordings/{file_name}.mp3'
        audio_transcript_path = f'{NEMO_ROOT}/recordings/{file_name}.csv'
        audio_transcript = pd.read_csv(audio_transcript_path, sep=None, engine='python')
        audio_transcript.rename(columns=lambda x: x.strip(), inplace=True)
        audio_name = file_name

        df_lengths += audio_transcript.shape[0]
        y, sr = librosa.load(audio_path)

        rand_num = 0
        # create segments here.
        # firstly combine 
        for idx, row in audio_transcript.iterrows():
            label = row['speaker']

            spk_dir = os.path.join(NEMO_ROOT,'segments', f'{label}_{file_name}')
            os.makedirs(spk_dir, exist_ok=True)
            if not isinstance(row['end'], numbers.Number):
                start = _get_seconds_from_time(row['start'].strip())
                end = _get_seconds_from_time(row['end'].strip())
            else:
                start = row['start']
                end = row['end']
            y_sample = y[int(start * sr): int(end * sr)]

            file_path = f"{spk_dir}/{file_name[0:7]}_{rand_num}.wav"

            sf.write(file_path, y_sample, sr)
            rand_num += 1


if __name__ == '__main__':
    csv_files = glob.glob(f'{NEMO_ROOT}/recordings/*.csv', recursive=True)
    file_names = [name.split('/')[1].split('.csv')[0] for name in csv_files]
    # this file produced mostly empty segments, so I removed it completely
    file_names.remove('7d024765a41e42a89480080bd40cad3c')
    print(file_names)

    create_segments(file_names)
