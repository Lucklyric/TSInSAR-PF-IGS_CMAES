#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : generate_ps.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 06.02.2021

import glob
import matplotlib.pyplot as plt
import joblib
import numpy as np
from mrc_insar_common.data import data_reader
import os
import json

# ! Adjust data root
DATA_ROOT = '/quobyte/dev_machine_learning'

def get_mean_cohs(dataname):
    fs = glob.glob('{}/*{}*/ifg_hr/*.natm.filt.coh'.format(DATA_ROOT,dataname))
    fs.sort()

    mean_coh = np.zeros([1500,1500])
    cohs = np.zeros([len(fs), 1500, 1500])
    for i in range(len(fs)):
        coh = data_reader.readBin(fs[i], 1500, 'float')
        mean_coh += coh
        cohs[i] = coh
    mean_coh/=len(fs)
    print('db:{}, len:{}'.format(dataname, len(fs)))
    sample_rslc_path = glob.glob('{}/*{}*/rmli_hr/*.rmli'.format(DATA_ROOT, dataname))[0]
    sample_rslc = data_reader.readBin(sample_rslc_path, 1500, 'float')

    sample_mr_path = glob.glob('{}/*{}*/fit_hr/def_fit_cmpy'.format(DATA_ROOT, dataname))[0]
    mr = data_reader.readBin(sample_mr_path, 1500, 'float')

    sample_he_path = glob.glob('{}/*{}*/fit_hr/hgt_fit_m'.format(DATA_ROOT, dataname))[0]
    he = data_reader.readBin(sample_he_path, 1500, 'float')
    return mean_coh, sample_rslc, mr, he, cohs

def generate_candidates(mean_coh, number, cohs):
    candidates = []
    [n,_,_] = cohs.shape
    while len(candidates) < number:
        x = np.random.randint(3, 997)
        y = np.random.randint(3, 997)
        # patch = cohs[:,x-2:x+2,y-2:y+2]
        patch = cohs[:, x,y]
        if ((patch>0.8).sum() >= (patch.size*0.6) and patch.std()<0.2):
        # if ((patch>0.7).sum() == n):
            candidates.append([x,y])
            # print(len(candidates))
            # print(patch.shape)
            # print(patch.min())
    return candidates

def  generate_ps(db_name, processing_root, output_path):
    db = db_name
    print(f"process {db}")
    mean_coh , sample_rslc, mr, he, cohs = get_mean_cohs(db)
    mean_coh = mean_coh[250:1250, 250:1250]
    cohs = cohs[:, 250:1250, 250:1250]
    sample_rslc = sample_rslc[250:1250, 250:1250]
    mr = mr[250:1250, 250:1250]
    he = he[250:1250, 250:1250]
    
    candidates = generate_candidates(mean_coh, 2000, cohs)

    out_dir = '{}/{}'.format(processing_root,db);

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.imsave('{}/mean_coh.png'.format(out_dir), mean_coh, cmap='gray');
    plt.imsave('{}/sample_rslc.png'.format(out_dir), np.log(np.abs(sample_rslc)+1e-5), cmap='gray');
    plt.imsave('{}/mr.png'.format(out_dir), mr, cmap='rainbow', vmin=-15, vmax=15);
    plt.imsave('{}/he.png'.format(out_dir), he, cmap='rainbow', vmin=-15, vmax=15);
    
    plt.cla()
    plt.clf()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    implot = plt.imshow(np.log(np.abs(sample_rslc)+1e-5), cmap='gray')
    plt.axis('off')
    plt.scatter(x=[c[1] for c in candidates], y=[c[0] for c in candidates], s=6, c=[mr[x[0],x[1]] for x in candidates],cmap='rainbow',vmin=-15, vmax=15,edgecolors='none');
    plt.savefig('{}/ps.mr.png'.format(out_dir),dpi=200,bbox_inches='tight')
    plt.cla()
    plt.clf()

    implot = plt.imshow(np.log(np.abs(sample_rslc)+1e-5), cmap='gray')
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.scatter(x=[c[1] for c in candidates], y=[c[0] for c in candidates], s=6, c=[he[x[0],x[1]] for x in candidates],cmap='rainbow',vmin=-15, vmax=15,edgecolors='none');
    plt.savefig('{}/ps.he.png'.format(out_dir),dpi=200,bbox_inches='tight')
    plt.cla()
    plt.clf()

    print(f"save output to {output_path}")
    joblib.dump(candidates, output_path)
