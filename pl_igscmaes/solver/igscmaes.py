import logging
from functools import partial
from mrc_insar_common.data import data_reader
import shutil

import matplotlib.pyplot as plt
import cma
import numpy as np
import pytorch_lightning as pl
from joblib import Parallel, delayed
from omegaconf import OmegaConf
import json
import joblib 

from ..data.utils import acc_diff_thresh, ri_l1,ri_l2, wrap, neg_coh

log = logging.getLogger(__name__)


def grid_eval(i, j, mr, he, recn_fn, metric_fn, target):
    reward = metric_fn(source=recn_fn(mr, he))
    return i, j, reward


class IGSCMAES(pl.LightningModule):

    def __init__(self, max_iter, mr_bound, he_bound, loss_map_scale,num_of_inits, sample_scale_bound, cma_args, mr_grid_unit, he_grid_unit, loss_thresh,**kwargs):
        super().__init__()
        self.max_iter = max_iter
        self.num_of_inits = num_of_inits
        self.loss_thresh = loss_thresh
        self.mr_bound = mr_bound
        self.he_bound = he_bound
        self.sample_scale_bound = sample_scale_bound
        self.loss_map_scale = loss_map_scale
        self.mr_grid_unit = mr_grid_unit 
        self.he_grid_unit = he_grid_unit

        # this is for bouding the searchign space to [-10, 10]
        self.mr_scale = self.mr_bound[1]/self.loss_map_scale
        self.he_scale = self.he_bound[1]/self.loss_map_scale

        # convert omegeconfg to python dict for later **kwargs for CMAES library
        self.cma_args = OmegaConf.to_container(cma_args)

        log.info("solver inited")

        self.scales = np.arange(sample_scale_bound[1], sample_scale_bound[0], -1)

    def forward(self, x):
        # placeholder beacuse this is not a typical ML model
        return x

    def fit(self, ifg, coh):
        recon_fn = self.test_dataloader().dataset.recon_phase
        metric_fn = ri_l1
        # metric_fn = neg_coh
        # print(coh)
        coh[coh<0.5] = 0
        objfun = partial(metric_fn, target=ifg, weight=coh)
        init_points, scale_iters = self.select_canadiates(objfun, recon_fn, ifg)
        error = np.inf
        ret = (0, 0)
        total_iters = scale_iters
        for s in init_points:
            res = cma.CMAEvolutionStrategy(x0=[s[0], s[1]], **self.cma_args) 
            while not res.stop():
                res.tell(*res.ask_and_eval(lambda x:objfun(source=recon_fn(np.round(x[0] * self.mr_scale, 1), np.round(x[1] * self.he_scale, 2)))))
            result = res.result[0]    # cmaes mean results
            result = result[0], result[1]
            recon_phase = recon_fn(result[0] * self.mr_scale, result[1] * self.he_scale)
            loss = metric_fn(recon_phase, ifg, coh)

            total_iters += res.result[3]
            l = np.round(loss, 4)
            if (l < error):
                ret = (result[0], result[1])
                error = l
            else:
                if (np.abs(l - error) == 0):
                    if ((result[0])**2 + (result[1])**2) < ((ret[0])**2 + (ret[1])**2):
                        ret = (result[0], result[1])
                        error = l
        return ret, total_iters

    def select_canadiates(self, metric_fn, recon_fn, ifg):
        init_points = []
        for scale in self.scales:
            loss_map = self.get_loss_map(scale, recon_fn, metric_fn, ifg)
            init_points = self.get_init_points(loss_map, self.loss_thresh, scale, self.num_of_inits)
            if (len(init_points) >= self.num_of_inits):
                break
        print(scale, len(init_points))
        return init_points, (loss_map.size)

    def get_loss_map(self, scale, recon_fn, metric_fn, target):
        mrc = np.arange(-self.loss_map_scale, self.loss_map_scale, scale * self.mr_grid_unit / self.mr_bound[1])
        hec = np.arange(-self.loss_map_scale, self.loss_map_scale, scale * self.he_grid_unit/ self.he_bound[1])
        num_mr_samples = len(mrc)
        num_he_samples = len(hec)
        loss_map = np.zeros((num_mr_samples, num_he_samples))
        with Parallel(n_jobs=5, temp_folder='/cache/pf/grid') as para:
            ret = para(delayed(grid_eval)(i, j, mrc[i] * 2.5, hec[j] * 20, recon_fn, metric_fn, target) for i in (range(num_mr_samples)) for j in range(num_he_samples))

            # check all losses
            for t in ret:
                loss_map[t[0], t[1]] = t[2]
        return loss_map

    def get_init_points(self, loss_map, loss_thresh, scale, num_of_cma_inits):
        sorted_loss_map = np.sort(loss_map.flatten())
        # pick init point
        # init_points = [(0,0)]
        init_points = []
        # init_points = [(0,0)]
        # only accept when min-loss greater than 0.5
        # if (sorted_loss_map[0] > 0.3 and scale > 2):
        if (sorted_loss_map[0] > 0.6 and scale > 2):
            return init_points
        count = 0
        for l in sorted_loss_map:
            # print(l)
            idx = np.where(loss_map == l)
            candidate_p = [
                idx[0][0] * scale * self.mr_grid_unit / self.mr_bound[1] - self.loss_map_scale,
                idx[1][0] * scale * self.he_grid_unit / self.he_bound[1] - self.loss_map_scale
            ]
            if count == 0:
                init_points.append(candidate_p)
                count += 1
            else:
                for ex_idx in range(count):
                    ex = init_points[ex_idx]
                    dist = np.sqrt((candidate_p[0] - ex[0])**2 + (candidate_p[1] - ex[1])**2)
                    if dist > np.pi:
                        init_points.append(candidate_p)
                        count += 1
                        break
            if count >= num_of_cma_inits:
                return init_points
        return init_points

    def test_step(self, batch, batch_idx):
        recon_fn = self.test_dataloader().dataset.recon_phase
        with Parallel(n_jobs=len(batch),temp_folder='/cache/pf/fit/') as para:
            ret = para(delayed(self.fit)(sample[0], sample[1]) for sample in batch)
            batch_loss_ri_l1 = []
            batch_gt_acc_unw_l1 = []
            batch_total_iters = [] 
            ests = []
            for r, sample in zip(ret, batch):
                est_mr, est_he = np.round(r[0][0] * self.mr_scale, 1), np.round(r[0][1] * self.he_scale, 2)
                est_iters = r[1]
                # calcuate some metric
                recon_phase = recon_fn(est_mr, est_he)
                loss_ri_l1 = ri_l1(recon_phase, sample[0])
                wrap_phase_residuals = np.abs(np.angle(1*np.exp(1j*(wrap(recon_phase)-sample[0])))).mean()

                gt_unwrap = recon_fn(sample[2], sample[3])
                wrap_phase_residuals_ref = np.abs(np.angle(1*np.exp(1j*(wrap(gt_unwrap)-sample[0])))).mean()
                loss_ri_l1_ref = ri_l1(wrap(gt_unwrap), sample[0])

                mr_rmse = np.sqrt(np.mean((est_mr - sample[2])**2))
                he_rmse = np.sqrt(np.mean((est_he - sample[3])**2))

                batch_loss_ri_l1.append(loss_ri_l1)
                acc_unw_l1 = acc_diff_thresh(recon_phase, gt_unwrap, np.pi)
                batch_gt_acc_unw_l1.append(acc_unw_l1)
                batch_total_iters.append(est_iters)
                ests.append((est_mr, est_he, sample[2], sample[3]))
                self.log_dict({'total_iters': est_iters, 
                               'loss_ri_l1': loss_ri_l1, 
                               'acc_unw_l1': acc_unw_l1[1], 
                               'loss_ri_l1_ref': loss_ri_l1_ref, 
                               'wpr': wrap_phase_residuals, 
                               'wpr_ref': wrap_phase_residuals_ref, 
                               'mr_rmse': mr_rmse, 
                               'he_rmse': he_rmse})
                log.info(f"coh_>0.5:{(sample[1]>0.5).mean():.4f}, iters:{est_iters}, est_mr:{est_mr:.4f}, est_he:{est_he:.4f}, ref_mr:{sample[2]:.4f}, ref_he:{sample[3]:.4f}, ri_l1:{loss_ri_l1:.4f}, ri_l1_ref:{loss_ri_l1_ref:.4f}, wpr:{wrap_phase_residuals:.4f}, wpr_ref:{wrap_phase_residuals_ref:.4f}, mr_rmse:{mr_rmse:.4f}, he_rmse:{he_rmse:.4f}")
                # print((sample[1]>0.5).sum(), est_iters, est_mr, est_he, loss_ri_l1, np.abs(np.angle(1*np.exp(1j*(wrap(recon_phase)-sample[0])))).mean(),sample[2], sample[3],loss_ri_l1_ref,np.abs(np.angle(1*np.exp(1j*(wrap(gt_unwrap)-sample[0])))).mean())
        try:
            shutil.rmtree('/cache/pf')
        except Exception as e:
            print(e)
        return ests, batch_loss_ri_l1, batch_gt_acc_unw_l1, batch_total_iters

    def test_epoch_end(self, outputs):
        # aggreate all results
        ret = [x for (r1, r2, r3, r4) in outputs for x in zip(r1, r2, r3, r4)]
        joblib.dump(ret,'{}/raw_out.json'.format(self.trainer.log_dir))

        sample_rslc = self.test_dataloader().dataset.rslc_paths[0]
        rslc = (data_reader.readBin(sample_rslc, 1500, 'float'))[250:1250,250:1250]
        candidates = self.test_dataloader().dataset.ps_coords

        plt.cla()
        plt.clf()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        implot = plt.imshow(np.log(np.abs(rslc)+1e-5), cmap='gray')
        plt.axis('off')
        plt.scatter(x=[c[1] for c in candidates], y=[c[0] for c in candidates], s=6, c=[e[0][0] for e in ret],cmap='rainbow',vmin=-5, vmax=5,edgecolors='none');
        plt.savefig('{}/ps.est.mr.png'.format(self.trainer.log_dir),dpi=200,bbox_inches='tight')

        plt.cla()
        plt.clf()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        implot = plt.imshow(np.log(np.abs(rslc)+1e-5), cmap='gray')
        plt.axis('off')
        plt.scatter(x=[c[1] for c in candidates], y=[c[0] for c in candidates], s=6, c=[e[0][1] for e in ret],cmap='rainbow',vmin=-15, vmax=15,edgecolors='none');
        plt.savefig('{}/ps.est.he.png'.format(self.trainer.log_dir),dpi=200,bbox_inches='tight')


