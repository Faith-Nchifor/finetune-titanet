# In this file, we use the cosine similarity to evaluate the performance of the finetuned model in comparism with the pretrained model

import os
import glob
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr

from sklearn.metrics.pairwise import cosine_similarity

from finetune_embedding_model import NEMO_ROOT

# in this file we are going to evaluate the performance of the pretrained titanet model versus the fine-tuned model.
# we will compare use two sets of segments to calculate cosine similarity:
# - the first pair is two lists of segments by the same speaker from the same recording
# - the second pair is two lists of segments by different speakers from the same recording
# 


# NB: Every audio is labeled in format: speaker_label_filename_audio_segment 
# E.g 'agent_2_6c6ef345fb4f4d279cce7d67a9075bc4/6c6ef34_3.wav'
# - agent_2 is the speaker label
# - 6c6ef345fb4f4d279cce7d67a9075bc4 is the filename
# - 6c6ef34 is the audio segment
# - '3' is the segment number from the original transcript
# This is how we can identify speaker label by filename 


# Prerequisite: run the main.py to finetune the model, or else this script will not find the finetuned model


# fine tuned model path will be available only after the fine-tuning process takes place
finetuned_model_path = f'{NEMO_ROOT}/outputs/titanet-large-finetune.nemo'

pretrained_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('titanet_large')

finetuned_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(finetuned_model_path)


def calculate_similiarity(audio_1, audio_2):  # calculate similarity for two audios

    pretrained_embedding_audio_1 = pretrained_model.get_embedding(audio_1).cpu()

    pretrained_embedding_audio_2 = pretrained_model.get_embedding(audio_2).cpu()

    finetuned_embedding_audio_1 = finetuned_model.get_embedding(audio_1).cpu()

    finetuned_embedding_audio_2 = finetuned_model.get_embedding(audio_2).cpu()

    similiarity_score_audio_1_pretrained = \
    cosine_similarity(pretrained_embedding_audio_1, pretrained_embedding_audio_2)[0, 0]

    similiarity_score_audio_1_finetuned = cosine_similarity(finetuned_embedding_audio_1, finetuned_embedding_audio_2)[
        0, 0]
    '''print('evaluating cosine similarity for similar audios\n ')

    print(f' pretrained model: {similiarity_score_audio_1_pretrained}')

    print(f'finetuned model: {similiarity_score_audio_1_finetuned}')

    print('---------------------------------------------------------------')'''
    return (similiarity_score_audio_1_finetuned, similiarity_score_audio_1_pretrained)


# get cosine similarity for dataframe
def get_cosine_similarity(df):
    finetuned_cos_sim = []
    pretrained_cos_sim = []
    for idx, row in df.iterrows():
        ft, pt = calculate_similiarity(row['list_1'], row['list_2'])

        finetuned_cos_sim.append(ft)
        pretrained_cos_sim.append(pt)

    return (finetuned_cos_sim, pretrained_cos_sim)

def main():
    # first pair of segments
    train_dir = f'{NEMO_ROOT}/data/train'
    test_dir = f'{NEMO_ROOT}/data/test'

    train_labels = os.listdir(train_dir)
    sim_train_segments_labels = glob.glob(os.path.join(train_dir, 'Speaker_0_44b508e63911479985b7e79d5491d4d8', '*.wav'),
                                        recursive=True)

    test_labels = os.listdir(test_dir)
    sim_test_segments_labels = glob.glob(os.path.join(test_dir, 'Speaker_0_44b508e63911479985b7e79d5491d4d8', '*.wav'),
                                        recursive=True)

    # second pair of segments
    diff_speakers_test_segment = glob.glob(f'{test_dir}/Speaker_0_0b328b8c97f14805b51bfa504bf4becf/*.wav', recursive=True)
    diff_speakers_train_segment = glob.glob(f'{train_dir}/Speaker_1_0b328b8c97f14805b51bfa504bf4becf/*.wav', recursive=True)


    def compute_cos_sim(list1, list2):
        cosine_similarities_finetuned = []
        cosine_similarities_pretrained = []
        for i in range(len(list2)):
            ft, pt = calculate_similiarity(list1[i], list2[i])
            cosine_similarities_pretrained.append(pt)
            cosine_similarities_finetuned.append(ft)
        return np.mean(cosine_similarities_finetuned), np.mean(cosine_similarities_pretrained)


    cos_sims_same_speaker = compute_cos_sim(sim_train_segments_labels, sim_test_segments_labels)
    cos_sim_diff_speakers = compute_cos_sim(diff_speakers_train_segment, diff_speakers_test_segment)

    print(
        f'cosine similarity for segments from same speaker for pretrained: {cos_sims_same_speaker[1]} \n finetuned: {cos_sims_same_speaker[0]}')

    print(
        f'cosine similarity for segments from two different speakers for pretrained: {cos_sim_diff_speakers[1]} \n finetuned: {cos_sim_diff_speakers[0]}')


if __name__ == '__main__':
    main()