#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : generate_ps.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 06.02.2021

import glob
import json
import logging
import os

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from mrc_insar_common.data import data_reader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from .utils import gen_sim_mr_he, get_delta_days, wrap
import tqdm

log = logging.getLogger(__name__)

# ! Adjust data root
DATA_ROOT = '/quobyte/dev_machine_learning'

def crop(source):
    return source[250:1250,250:1250]

class DBPsProcessor(object):
    def __init__(self, filt_dir, filt_ext, coh_dir, coh_ext, bperp_dir, bperp_ext, conv1, conv2, width, height,ref_mr_path, ref_he_path, ps_coords_path, rslc_dir, rslc_ext, *args,**kwargs):
        self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.filt_paths = sorted(glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.rslc_paths = sorted(glob.glob('{}/*{}'.format(rslc_dir, rslc_ext)))
        self.height = height
        self.stack_size = len(self.bperp_paths)
        assert len(self.bperp_paths) == len(self.filt_paths) == len(self.coh_paths)
        self.width = width;
        log.info('stack size {}'.format(self.stack_size))
        self.conv1 = conv1
        self.conv2 = conv2
        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path

        self.ps_coords = np.asarray(joblib.load(ps_coords_path))
        self.length = len(self.ps_coords)
        log.info('ps loaded with len {}'.format(len(self.ps_coords)))

        self.pre_fetch_ifgs = np.zeros([self.stack_size, self.width, self.height])
        self.pre_fetch_cohs = np.zeros([self.stack_size, self.width, self.height])
        self.pre_fetch_mr = data_reader.readBin(self.ref_mr_path, self.width, 'float')
        self.pre_fetch_he = data_reader.readBin(self.ref_he_path, self.width, 'float')
        self.sample_rslc = data_reader.readBin(self.rslc_paths[0], self.width, 'float')

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]

            self.pre_fetch_ifgs[idx] = np.angle(data_reader.readBin(self.filt_paths[idx], self.width, 'floatComplex'))
            self.pre_fetch_cohs[idx] = data_reader.readBin(self.coh_paths[idx], self.width, 'float')
        log.info('bp dday loaded')

        log.info('done')

    def recon_phase(self, mr, he):
        return self.ddays* self.conv1 * mr + self.bperps * self.conv2 * he

    def generate_candidates(self,  number):
        candidates = []
        while len(candidates) < number:
            x = np.random.randint(3, 997) + 250
            y = np.random.randint(3, 997) + 250
            # patch = cohs[:,x-2:x+2,y-2:y+2]
            cohs = self.pre_fetch_cohs[:, x,y]
            mr = self.pre_fetch_mr[x,y]
            he = self.pre_fetch_he[x,y]
            ifg = self.pre_fetch_ifgs[:,x,y]
            recon_phase = self.recon_phase(mr, he)
            diff = np.abs(np.angle(np.exp(1j*recon_phase))-np.angle(np.exp(1j*ifg))).mean()
            if ((cohs>0.8).sum() >= (cohs.size*0.6) and diff< 0.9):
                candidates.append([x-250,y-250])
                print(len(candidates))
                
        return candidates

def  generate_ps(db_info,processing_root):
    print(f"process {db_info.name}")
    dbp = DBPsProcessor(**(db_info))
    cohs = dbp.pre_fetch_cohs[:, 250:1250, 250:1250]
    mean_coh= cohs.mean(0)
    sample_rslc = dbp.sample_rslc[250:1250, 250:1250]
    mr = dbp.pre_fetch_mr[250:1250, 250:1250]
    he = dbp.pre_fetch_he[250:1250, 250:1250]
    
    candidates = dbp.generate_candidates(2000)
    out_dir = processing_root

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
    plt.scatter(x=[c[1] for c in candidates], y=[c[0] for c in candidates], s=6, c=[mr[x[0],x[1]] for x in candidates],cmap='rainbow',vmin=-5, vmax=5,edgecolors='none');
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

    return candidates
