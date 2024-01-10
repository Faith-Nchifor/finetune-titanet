import os
import glob
import wget
import shutil
import tarfile
import subprocess


NEMO_ROOT = os.path.dirname(os.path.dirname(__file__))

# some utility functions used in this experiment

def empty_dir(folder):

    folder = os.path.join(NEMO_ROOT,folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_files(file_path):
    '''
        returns list of files within a directory
    '''

    data=  []

    
    for path, subdirs, files in os.walk(file_path):
        for name in files:
            
            if(name.endswith('.wav')):

                data.append(f'{file_path}/{name}')
    return data


def download():
    # download datafile, extract, convert and turn to wav. then delete tar/zip file and all flac files to free up space
    # check for librispeech: if not there check for data donwload
    if not os.path.exists(f'{NEMO_ROOT}/data/LibriSpeech/dev-other'):
        tar_path= os.path.join(NEMO_ROOT,'data','dev-other.tar.gz')
        if not os.path.exists(tar_path):
            dataset_url = 'https://www.openslr.org/resources/12/dev-other.tar.gz'
            wget.download(dataset_url, 'data')
        else:
            print('tar file already exists')

        # extract file
        file = tarfile.open(tar_path)
        # extracting file
        file.extractall('data')
        file.close()
        os.remove(tar_path)

    file_list_path=glob.glob(f'{NEMO_ROOT}/data/LibriSpeech/dev-other/**/*.flac',recursive=True)
    print("len(file_list_path)", len(file_list_path))

    # Convert flac to wav
    for path in file_list_path:    
        new_file_path=path.replace('.flac','.wav')
        cmd = ["sox", path, new_file_path]
        subprocess.run(cmd)
        os.remove(path)
    print('finished conversion and removed flac files')


if __name__ =='__main__':
    download()