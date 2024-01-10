import os
import glob
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr
from sklearn.metrics.pairwise import cosine_similarity
from finetune_embedding_model import NEMO_ROOT

# In this file, we use the cosine similarity to evaluate the performance of the finetuned model in comparism with the pretrained model






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

def compute_cos_sim(list1, list2):
        '''computes cosine similarity between two lists of items and return their average value, both for pretrained and finetuned'''
        cosine_similarities_finetuned = []
        cosine_similarities_pretrained = []
        if(len(list1)<=len(list2)):
            range_val=range(len(list1))
        else:
            range_val=range(len(list2))
        for i in range_val:
            ft, pt = calculate_similiarity(list1[i], list2[i])
            cosine_similarities_pretrained.append(pt)
            cosine_similarities_finetuned.append(ft)
        print(len(range_val))
        
        
        return np.mean(cosine_similarities_finetuned), np.mean(cosine_similarities_pretrained)

def compute_group_cos_sim(grp):
    '''compute cosine similarity between group of values'''
    similarity_values_pretrained=[]
    similarity_values_finetuned=[]
    for pair in grp:
        list_1=pair[0]
        list_2=pair[1]
        cos_sim= compute_cos_sim(list_1,list_2)
        similarity_values_finetuned.append(cos_sim[0])
        similarity_values_pretrained.append(cos_sim[1])
    return (np.mean(similarity_values_finetuned),np.mean(similarity_values_pretrained))

def main():
    
    data_dir = f'{NEMO_ROOT}/data/LibriSpeech/dev-other'
      
      #segments by same speaker
    sim_pair_1 = glob.glob(os.path.join(data_dir, '116/288048', '*.wav'),recursive=True)    
    sim_pair_2 = glob.glob(os.path.join(data_dir, '116/288045', '*.wav'),recursive=True)

    sim_pair_3= glob.glob(f'{data_dir}/3660/172182/*.wav', recursive=True)
    sim_pair_4= glob.glob(f'{data_dir}/3660/172183/*.wav', recursive=True)

    sim_pair_5=glob.glob(f'{data_dir}/1650/167613/*.wav', recursive=True)
    sim_pair_6=glob.glob(f'{data_dir}/1650/157641/*.wav', recursive=True)

     #  segments by different speakers
    diff_pair_1 = glob.glob(f'{data_dir}/1255/138279/*.wav', recursive=True)
    diff_pair_2 = glob.glob(f'{data_dir}/8288/274162/*.wav', recursive=True)

    diff_pair_3= glob.glob(f'{data_dir}/4570/14911/*.wav', recursive=True)
    diff_pair_4= glob.glob(f'{data_dir}/3660/172183/*.wav', recursive=True)

    diff_pair_5= glob.glob(f'{data_dir}/4153/61735/*.wav', recursive=True)
    diff_pair_6= glob.glob(f'{data_dir}/4515/11057/*.wav', recursive=True)

    same_speaker_set=[(sim_pair_1,sim_pair_2),
                      (sim_pair_3,sim_pair_4),
                      (sim_pair_5,sim_pair_6)]
    
    different_speaker_set=[(diff_pair_1,diff_pair_2),
                           (diff_pair_3,diff_pair_4),
                           (diff_pair_5,diff_pair_6)]

    cos_sims_same_speaker = compute_group_cos_sim(same_speaker_set)
    cos_sim_diff_speakers = compute_group_cos_sim(different_speaker_set)

    print(
        f'cosine similarity for segments from same speaker for pretrained: {cos_sims_same_speaker[1]} \n finetuned: {cos_sims_same_speaker[0]}')

    print(
        f'cosine similarity for segments from two different speakers for pretrained: {cos_sim_diff_speakers[1]} \n finetuned: {cos_sim_diff_speakers[0]}')

if __name__ == '__main__':
    main()