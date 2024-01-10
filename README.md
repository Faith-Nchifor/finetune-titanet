# Fine-tuning Speaker embedding model (Titanet)

This package allows you to finetune titanet_large on librispeech.
## What is in this Repo?
- `train.py` allows you to fine-tune the model. This script should be run first before executing the other scripts

- `scripts/eval_cosine_sim.py` is for evaluating the cosine similarity of some segments from the `data` directory. To run the script, use `python scripts/eval_cosine_sim.py`


## Folders in the Repo
- data is the segmented recordings split into train and test (The smaller dataset). The original recordings are also included in the `recordings` subdirectory alongside their transcripts (in csv format), and annotations in rttm formats (generated from transcripts). 
- scripts folder contains the scripts for this package
Some folders will be created when the package is being run, like 'nemo_experiments' and 'segments'
- scripts/old contain the old scripts for the first training on the small dataset (not librispeech).

## How to run the package
- This package runs with the latest version of Nemo and requires python 3.10.12. Nemo does not support 3.11 at this time.
- install correct version of torch. For instance, run `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- on your terminal: `apt-get update && apt-get install -y libsndfile1 ffmpeg`
- you also need to have soX on your system `sudo apt-get install sox`
- Next run `pip install -r requirements.txt`
- Next, install run `pip install -e .` to install the package in your environment.
- run `python scripts/train.py`
- When the file has ran, you can call the fine-tuned model from the directory using the `restore_from` method of the 'EncDecSpeakerLabelModel' class. The loaded model can be used for inference (the model will be saved in a 'nemo_experiments' folder)