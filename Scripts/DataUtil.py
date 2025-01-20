#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import json
import os
import pickle
import pathlib
import itertools
import gzip


# In[ ]:


def save_to_csv(data, path, filename):
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)
        
def load_compressed_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with gzip.open(datafile, 'rt', encoding='UTF-8') as zipfile:
            return json.load(zipfile)
        
def load_all_results(path):
    if not os.path.isdir(path):
        return []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for datafile in onlyfiles:
        with open(path + '/' + datafile, 'rb') as file:
            data.append(json.load(file))
    return data

def save_data(data, path, filename, override=True):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        if override:
            try:
                os.remove(datafile)
            except OSError:
                pass
        else:
            return
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)
        
def compress_and_save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    
    with gzip.open(datafile, 'wt', encoding='UTF-8') as zipfile:
        json.dump(data, zipfile)
       