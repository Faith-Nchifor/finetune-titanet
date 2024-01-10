# This file calculates the DER of the recordings for the pretrained and finetuned models
# The output is a dataframe showing the result of inference by both models.

import os
import glob
import json
import pandas as pd

from omegaconf import OmegaConf
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

from finetune_embedding_model import NEMO_ROOT

output_dir = os.path.join(NEMO_ROOT,'outputs')
metric = DiarizationErrorRate()

files = glob.glob(f'{NEMO_ROOT}/data/recordings/*.rttm', recursive=True)
file_names = [path.split('/')[-1].split('.')[0] for path in files]


# generate jsonl file from ground truth segments.
def generate_ext_vad():
    fp = open(f'{output_dir}/true_ref.json', 'w+')
    for rttm_file in files:

        true_ref = rttm_to_labels(rttm_file)
        filename = rttm_file.split('/')[-1].split('.')[0]
        audio_path = f'{NEMO_ROOT}/data/recordings/{filename}.wav'

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

            json.dump(vad_data, fp)
            fp.write('\n')
    fp.close()


# function that predicts annotations and returns the mean DER
def predict_annot(cfg):
    clustering_diarizer = ClusteringDiarizer(cfg=cfg)
    result = clustering_diarizer.diarize()
    return result


def calculate_der(predictions_dir='pt_'):
    arr = {}

    for file_name in file_names:
        predicted_rttm = load_rttm(f'{output_dir}/{predictions_dir}/pred_rttms/{file_name}.rttm')
        # predicted_rttm[file_name]
        # true_rttm=generate_annot(input_dir+file_name+'.csv')
        true_rttm_path = os.path.join(f'{NEMO_ROOT}/data/recordings', (f'{file_name}.rttm'))
        true_rttm = load_rttm(true_rttm_path)
        der = metric(true_rttm[file_name], predicted_rttm[file_name])
        arr[file_name] = der
    df = pd.DataFrame({
        'filename': arr.keys(),
        f'{predictions_dir}_der': arr.values()})
    return df


def main():
    # load finetuned model specific checkpoint
    finetuned_model_path = f'{output_dir}/titanet-large-finetune.nemo'

    # put original links to recordings and their rttm  in txt files
    rttm_paths = glob.glob(f'{NEMO_ROOT}/data/recordings/*.rttm', recursive=False)
    audio_paths = glob.glob(f'{NEMO_ROOT}/data/recordings/*.wav', recursive=False)

    rttm_data_file = open(f'{output_dir}/rttms.txt', 'w+')
    for link in rttm_paths:
        rttm_data_file.write(link + '\n')
    rttm_data_file.close()

    audio_data_file = open(f'{output_dir}/data.txt', 'w+')
    for link in audio_paths:
        audio_data_file.write(link + '\n')
    audio_data_file.close()

    # generate manifest file for test
    create_manifest(
        rttm_path=f'{output_dir}/rttms.txt',
        wav_path=f'{output_dir}/data.txt',
        manifest_filepath=f'{output_dir}/infer_manifest.json',
        add_duration=True)

    # Get ground truth segments from rttm files and generate
    generate_ext_vad()

    config = OmegaConf.load(f'{NEMO_ROOT}/conf/diar_infer_telephonic.yaml')  # load default configurations for pipeline inference

    config.diarizer.oracle_vad = False  # we do not want VAD
    config.diarizer.vad.model_path = None
    config.diarizer.vad.external_vad_manifest = f'{output_dir}/true_ref.json'
    config.diarizer.clustering.parameters.oracle_num_speakers = True

    config.diarizer.manifest_filepath = f'{output_dir}/infer_manifest.json'

    # use pipeline to generate predicted annotations for pretrained model
    pt_config = config.copy()
    pt_config.diarizer.out_dir = f'{output_dir}/pt_'
    pt_config.diarizer.speaker_embeddings.model_path = 'titanet_large'
    pt_result = predict_annot(pt_config)

    # do the same for finetuned model
    config_ft = config.copy()
    config_ft.diarizer.ignore_overlap = False
    config_ft.diarizer.out_dir = f'{output_dir}/ft_'
    config_ft.diarizer.speaker_embeddings.model_path = finetuned_model_path
    ft_result = predict_annot(config_ft)

    # now get der of recordings for both models and place in the dataframe
    pretrained_der = calculate_der('pt_')
    finetuned_der = calculate_der('ft_')  # output is a dataframe showing the various DERs for the files

    # result_df=pretrained_der.merge(finetuned_der)
    # result_df.columns=['filename','pretrained_der','finetuned_der']

    print('DER from finetuned model: ')
    print(finetuned_der)

    print('DER from pretrained model: ')
    print(pretrained_der)


if __name__ == '__main__':
    main()