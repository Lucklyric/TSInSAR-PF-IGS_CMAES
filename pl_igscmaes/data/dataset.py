import glob
import logging
from numba import jit

import torch
import tqdm
from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .utils import gen_sim_mr_he, get_delta_days, wrap
import json
import joblib

log = logging.getLogger(__name__)


class SimFittingDataset(Dataset):

    def __init__(self, bperp_dir, bperp_ext, conv1, conv2, width, col, sim_signals_path=None, *args, **kwargs):
        self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.stack_size = len(self.bperp_paths)

        log.info('stack size {}'.format(self.stack_size))

        self.conv1 = conv1
        self.conv2 = conv2

        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)

        log.info('start to read data')
        if (sim_signals_path):
            self.sim_signals = np.load(sim_signals_path)
            log.info('sim signals loaded {}'.format(sim_signals_path))
        else:
            self.sim_signals = gen_sim_mr_he(30, 60, -25, 25, -200, 200)
            log.info('sim signals generated with shape {}'.format(self.sim_signals.shape))

        self.length = self.sim_signals.shape[0] # total number of simulated temporal stacks [1, N] N is number of ifgs
        log.info('total sim sampels {}'.format(self.length))

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]
        log.info('data loaded')

    def __len__(self):
        return self.length

    def __getitem__(self, x):
        [mr, he] = self.sim_signals[x]
        ifg = wrap(self.recon_phase(mr, he))
        coh = 1 # hard code sim data
        return ifg, coh, mr, he

    def recon_phase(self, mr, he):
        return self.ddays* self.conv1 * mr + self.bperps * self.conv2 * he

class RealPsFittingDataset(SimFittingDataset):

    def __init__(self, filt_dir, filt_ext, coh_dir, coh_ext, bperp_dir, bperp_ext, conv1, conv2, width, height,ref_mr_path, ref_he_path, ps_coords_path, rslc_dir, rslc_ext, pre_fetch=True, *args,**kwargs):
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
        self.pre_fetch = pre_fetch

        self.ps_coords = np.asarray(joblib.load(ps_coords_path))
        self.length = len(self.ps_coords)
        log.info('ps loaded with len {}'.format(len(self.ps_coords)))

        if (pre_fetch):
            self.pre_fetch_ifgs = np.zeros([self.stack_size, self.width, self.height])
            self.pre_fetch_cohs = np.zeros([self.stack_size, self.width, self.height])
            self.pre_fetch_mr = data_reader.readBin(self.ref_mr_path, self.width, 'float')
            self.pre_fetch_he = data_reader.readBin(self.ref_he_path, self.width, 'float')

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]

            if (pre_fetch):
                # read ifgs 
                self.pre_fetch_ifgs[idx] = np.angle(data_reader.readBin(self.filt_paths[idx], self.width, 'floatComplex'))
                self.pre_fetch_cohs[idx] = data_reader.readBin(self.coh_paths[idx], self.width, 'float')
        log.info('bp dday loaded')

        log.info('done')

    def __getitem__(self, x):

        ifgs = np.zeros([self.stack_size])
        cohs = np.zeros([self.stack_size])
        [row, col] = self.ps_coords[x]
        # shift by 250
        row += 250
        col += 250
        if (self.pre_fetch):
            ifgs = self.pre_fetch_ifgs[:,row,col]
            cohs = self.pre_fetch_cohs[:,row,col]
            ref_mr = self.pre_fetch_mr[row, col].squeeze()
            ref_he = self.pre_fetch_he[row, col].squeeze()
        else:
            for ifg_idx in range(self.stack_size):
                filt_path = self.filt_paths[ifg_idx]
                coh_path = self.coh_paths[ifg_idx]
                ifgs[ifg_idx] = np.angle(data_reader.readBin(filt_path, self.width, 'floatComplex', crop=(row, col, 1, 1)).squeeze())
                cohs[ifg_idx] = data_reader.readBin(coh_path, self.width, 'float', crop=(row, col, 1, 1)).squeeze()

            # read bperp
            ref_mr = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(row, col, 1, 1)).squeeze()
            ref_he = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(row, col, 1, 1)).squeeze()

        return ifgs, cohs, ref_mr, ref_he 

