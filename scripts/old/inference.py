# IN this script,
# I run inference without splitting the timestamps of the input segments
# The output of this script is the DER for each file

import os
import json
import glob
import torch
import librosa
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from copy import deepcopy
from pydub import AudioSegment
from typing import Any, List, Optional, Union

from pytorch_lightning.utilities import rank_zero_only
from pyannote.database.util import load_rttm
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import nemo.collections.asr.parts.utils.speaker_utils as spk_utils
import nemo.collections.asr.parts.utils.offline_clustering as offline_clustering
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.parts.utils.manifest_utils import create_manifest
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object, labels_to_rttmfile
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering, SpectralClustering
from finetune_embedding_model import NEMO_ROOT

rttm_files = glob.glob(f'{NEMO_ROOT}/data/recordings/*.rttm', recursive=True)  # get all (to be used as our)


# utility functions
def get_contiguous_stamps(stamps):
    """
    Return contiguous time stamps
    """
    lines = deepcopy(stamps)
    contiguous_stamps = []
    for i in range(len(lines) - 1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i + 1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i + 1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps


def merge_stamps(lines):
    """
    Merge time stamps of the same speaker.
    """
    stamps = deepcopy(lines)
    overlap_stamps = []
    for i in range(len(stamps) - 1):
        start, end, speaker = stamps[i].split()
        next_start, next_end, next_speaker = stamps[i + 1].split()
        if float(end) == float(next_start) and speaker == next_speaker:
            stamps[i + 1] = ' '.join([start, next_end, next_speaker])
        else:
            overlap_stamps.append(start + " " + end + " " + speaker)

    start, end, speaker = stamps[-1].split()
    overlap_stamps.append(start + " " + end + " " + speaker)

    return overlap_stamps


def generate_cluster_labels(segment_ranges: List[str], cluster_labels: List[int]):
    """
    Generate cluster (speaker labels) from the segment_range list and cluster label list.

    Args:
        segment_ranges (list):
            List containing intervals (start and end timestapms, ranges) of each segment
        cluster_labels (list):
            List containing a cluster label sequence

    Returns:
        diar_hyp (list):
            List containing merged speaker-turn-level timestamps and labels in string format
            Example:
                >>>  diar_hyp = ['0.0 4.375 speaker_1', '4.375 5.125 speaker_0', ...]

        lines (list)
            List containing raw segment-level timestamps and labels in raw digits
                >>>  diar_hyp = ['0.0 0.25 speaker_1', '0.25 0.5 speaker_1', ..., '4.125 4.375 speaker_1']
    """
    lines = []
    for idx, label in enumerate(cluster_labels):
        tag = 'speaker_' + str(label)
        stt, end = segment_ranges[idx]
        lines.append(f"{stt} {end} {tag}")
    cont_lines = get_contiguous_stamps(lines)
    diar_hyp = merge_stamps(cont_lines)
    return diar_hyp, lines
    # return cont_lines

    # perform the embedding and clustering

    # model=EncDecSpeakerLabelModel.restore_from('/kaggle/working/titanet-large-finetune_new.nemo')


def generate_ext_vad():  # put segments into json file
    fp = open(f'{NEMO_ROOT}/outputs/true_ref.json', 'w+')
    for rttm_file in rttm_files:

        true_ref = rttm_to_labels(rttm_file)
        filename = rttm_file.split('/')[-1].split('.')[0]
        audio_path = os.path.join(NEMO_ROOT,'data','recordings', filename + '.wav')

        # manifest_list=[]
        audio_name = filename
        for ref in true_ref:
            line = ref.split(' ')
            offset = float(line[0])
            end = float(line[1])

            duration = end - offset
            uniq_id = audio_name
            vad_data = {
                'audio_filepath': audio_path,
                'offset': offset,
                'duration': duration,
                'label': 'UNK',
                'uniq_id': uniq_id
            }
            if duration > 0.0000:
                json.dump(vad_data, fp)
                fp.write('\n')
    fp.close()

def main():
    warnings.filterwarnings("ignore")

    generate_ext_vad()

    wav_files = glob.glob(f'{NEMO_ROOT}/data/recordings/*.wav', recursive=True)

    data_path = f'{NEMO_ROOT}/outputs/data.txt'
    data_txt_file = open(data_path, 'w+')
    for wav in wav_files:
        filename = wav.split('/')[-1].split('.')[0]
        if (filename !='7d024765a41e42a89480080bd40cad3c'):
            data_txt_file.write(wav + '\n')
    data_txt_file.close()

    manifest_filepath = f'{NEMO_ROOT}/outputs/manifest.json'
    create_manifest(
        wav_path=data_path,  # create segments path for accessing
        manifest_filepath=manifest_filepath,
        add_duration=True
    )

    model = EncDecSpeakerLabelModel.restore_from(f'{NEMO_ROOT}/outputs/titanet-large-finetune.nemo')
    metric = DiarizationErrorRate()
    spectral_model = SpectralClustering(  # define clustering algorith with number of speakers
        n_clusters=2,
        n_random_trials=1,
        cuda=False,
        # device=device
    )

    working_dir = f'{NEMO_ROOT}/outputs'
    output_dir = os.path.join(working_dir, 'predicted_der')
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{NEMO_ROOT}/outputs/true_ref.json', 'r') as vad:
        lines = vad.readlines()
        vad_segments = [json.loads(line) for line in lines]
        all_segments = vad_segments

    # create a loop for all the recordings; get segments from each file and extract embeddings
    with open(f'{NEMO_ROOT}/outputs/manifest.json', 'r') as mani:
        manifest_lines = mani.readlines()
        manifest = [json.loads(line) for line in manifest_lines]
        for line in manifest:
            filename = line['audio_filepath'].split('/')[-1].split('.')[0]

            # print(filename)
            file_segments = list(filter(lambda i: i['uniq_id'] == filename, all_segments))
            # print(len(file_segments))
            timestamps = torch.empty(size=(len(file_segments), 2))
            all_embs = torch.empty(size=(len(file_segments), 192))
            # f = AudioSegment.from_file(line['audio_filepath'])
            f, sr = librosa.load(line['audio_filepath'])
            exp_link = f'{NEMO_ROOT}/outputs/test.wav'
            i = 0
            for seg in file_segments:  # for each segment, generate embeddings

                seg_start = seg['offset']
                seg_dur = seg['duration']
                seg_end = seg_start + seg_dur
                audio_file = seg['audio_filepath']

                y_sample = f[int(seg_start * sr): int(seg_end * sr)]
                # seg_path=f[seg_start * 1000:seg_end * 1000]
                temp_fn = f'{NEMO_ROOT}/outputs/{filename}.wav'
                # seg_path.export(temp_fn,format='wav')
                sf.write(temp_fn, y_sample, sr)
                '''seg_embs=[]
                subsegments=spk_utils.get_subsegments(offset=seg_start,window=1.5,shift=0.75,duration=seg_dur)
                for subsegment in subsegments:     
                    subseg_start=subsegment[0] * 1000
                    subseg_dur=subsegment[1] * 1000
                    subseg_end=subseg_start + subseg_dur
                    sebseg=f[subseg_start : subseg_end]

                    sebseg.export(exp_link)
                    emb=model.get_embedding(exp_link).cpu().detach()

                    seg_embs.append(emb)
                    os.remove(exp_link)
                avg_emb=torch.mean(torch.cat(seg_embs),axis=0)'''
                emb = model.get_embedding(temp_fn).cpu().detach()

                all_embs[i] = emb  # avg_emb#emb

                timestamps[i] = torch.tensor([seg_start, seg_end])
                i = i + 1

                os.remove(temp_fn)
            affinity_mat = offline_clustering.getCosAffinityMatrix(all_embs)  # get affinity matrix for all embeddings
            y = spectral_model.forward(affinity_mat)  # generate clustering labels
            cluster_labels = y.cpu().detach()

            # print(cluster_labels)

            lines, labels = generate_cluster_labels(timestamps, np.array(cluster_labels))

            # save files to output directory
            labels_to_rttmfile(lines, filename, output_dir)

            pred_file = load_rttm(f'{output_dir}/{filename}.rttm')[filename]
            true_file = load_rttm(f'{NEMO_ROOT}/data/recordings/{filename}.rttm')[filename]
            print(f'DER for for {filename}: {metric(true_file, pred_file)}')
        


if __name__ == '__main__':
    main()