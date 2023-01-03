import os
import shutil

import opendatasets as od

# Download the dataset
if os.path.exists('images'):
    shutil.rmtree('images')
os.mkdir('images')
dataset_url = 'https://www.kaggle.com/datasets/dandandooo/images'
od.download(dataset_url, data_dir='')