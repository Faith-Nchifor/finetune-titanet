import os
import glob
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from finetune_embedding_model import NEMO_ROOT

from finetune_embedding_model.utils import empty_dir
from finetune_embedding_model.filelist_to_manifest import \
                               main as filelist_to_manifest


MODEL_CONFIG = os.path.join(NEMO_ROOT, 'conf/titanet-finetune.yaml')

files_dir = os.path.join(NEMO_ROOT, 'data')

output_dir = os.path.join(NEMO_ROOT, 'outputs')


def finetune_titanet():
    # prepare dataset. Generate manifest files for train and test dataset
    # get model config file (titanet large)
    # get finetune config file for titanet large
    # modify values of finetune config file. For example, set the train_ds and val_ds_manifest_file for train and validation respectively
    # create pytorch lightening trainer and add finetune config file to it
    # instantiate speaker embedding model and add model  config file to it
    # fit the model to trainer
    # save last checkpoint as model in .nemo format
    # for testing fine-tuned model, load model from checkpoint and use the get_embedding method to get speaker embeddings

    # data processing.
    os.makedirs(files_dir,exist_ok=True)
    os.makedirs(output_dir,exist_ok=True)

    empty_dir('outputs')
    empty_dir('segments')
    """
    data processing
    - First collect the links to segments from train and test dataset and place them in a text folder
    - then use the filelist_to_manifest.py script to 
        - further split the segments into smaller segments where neccesary
        - use segment information to generate the .json format that will be used by the model
        - once you run this script, you can check in output/train.json to view the json file structure
    """

    train_data = glob.glob(f'{NEMO_ROOT}/data/train/**/*.wav', recursive=False)
    test_data = glob.glob(f'{NEMO_ROOT}/data/test/**/*.wav', recursive=False)

    with open(f'{output_dir}/train_all.txt', 'w+') as train_data_file: #copy links of files to the train dataset to a text file
        for link in train_data:
            train_data_file.write(link + '\n')

    with open(f'{output_dir}/test_all.txt', 'w+') as test_data_file: #copy links of files to the train dataset to a text file
        for link in test_data:
            test_data_file.write(link + '\n')

    train_manifest = os.path.join(output_dir, 'train.json') 

    test_manifest = os.path.join(output_dir, 'test.json')

    # generate manifest file, splits it into train.json and test.json.
    # Also trims the audiofiles into segments of length 1~3. This eases the finetuning process and keeps avoids cuda out of memory error
    filelist_to_manifest(filelist=f'{output_dir}/train_all.txt',
                         id=-2,
                         manifest=None,
                         out=train_manifest,
                         create_segments=True)
    
    filelist_to_manifest(filelist=f'{output_dir}/test_all.txt',
                         id=-2,
                         manifest=None,
                         out=test_manifest,
                         create_segments=True
                         )

    # then get config files for fine-tuning titanet_large

    finetune_config = OmegaConf.load(MODEL_CONFIG)

    # edit model configuration data in config file

    finetune_config.model.train_ds.manifest_filepath = train_manifest
    finetune_config.model.validation_ds.manifest_filepath = test_manifest
    finetune_config.model.decoder.num_classes = 8  # represents the total number of speaker labels in the dataset
    finetune_config.model.train_ds.batch_size = 15  
    finetune_config.model.model_defaults.droupout = 0.20
    finetune_config.model.optim.lr = 0.0004
    #finetune_config.model.train_ds.augmentor.speed.prob=0.4


    # create configurations for the trainer

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer_config = OmegaConf.create(dict(
        devices=1,
        accelerator=accelerator,
        max_epochs=80,
        max_steps=-1,  # computed at runtime if not set
        num_nodes=1,

        accumulate_grad_batches=1,
        enable_checkpointing=False,  # Provided by exp_manager
        logger=False,  # Provided by exp_manager
        log_every_n_steps=1,  # Interval of logging.
        val_check_interval=1.0,
    ))
    print(OmegaConf.to_yaml(trainer_config))

    # add trainer config to lightning trainer
    trainer_finetune = pl.Trainer(**trainer_config)

    # use nemo's experiment manager for log outputs
    log_dir_finetune = exp_manager(trainer_finetune, finetune_config.get("exp_manager", None))  # handle our

    # initialize model from pretrained checkpoint of config file
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)

    # fine-tuning time :) 
    trainer_finetune.fit(speaker_model)

    # get checkpoints and save last checkpoint as .nemo format
    checkpoint_dir = os.path.join(log_dir_finetune, 'checkpoints')
    checkpoint_paths = list(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    print('model checkpoints...')


    # save last checkpoint as final checkpoint
    final_checkpoint = list(filter(lambda x: "-last.ckpt" in x, checkpoint_paths))[0]
    print(final_checkpoint)

    # load model from last checkpoint
    restored_model = nemo_asr.models.EncDecSpeakerLabelModel.load_from_checkpoint(final_checkpoint)
    restored_model.save_to(f"{NEMO_ROOT}/outputs/titanet-large-finetune.nemo")

    print(
        'finetuned model can be loaded with "nemo_asr.models.EncDecSpeakerLabelModel.restore_from(outputs/titanet-large-finetune.nemo)"')

    print('training complete')


if __name__ == '__main__':
    finetune_titanet()