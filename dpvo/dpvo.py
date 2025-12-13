import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import sys
import os
from datetime import datetime
import cv2  # For visualization

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *
from .debug_utils import *

# Create a logger class to redirect output
class Logger:
    # Class variable to hold the global log file
    _log_file = None

    def __init__(self, log_dir="/home/jerett/Project/DPVO/Debug/Log"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Use a fixed log file name based on date (cleared each run)
        if Logger._log_file is None:
            date_str = datetime.now().strftime("%Y%m%d")
            Logger._log_file = os.path.join(log_dir, f"dpvo_log_{date_str}.txt")

            # Clear the log file at the start of each run
            try:
                with open(Logger._log_file, 'w', encoding='utf-8') as f:
                    # Write a header with run timestamp
                    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n{'='*60}\n")
                    f.write(f"DPVO Log - Run started at: {run_timestamp}\n")
                    f.write(f"{'='*60}\n\n")
            except Exception as e:
                print(f"Warning: Could not initialize log file: {e}")
                Logger._log_file = None

        self.log_file = Logger._log_file

        # Keep original stdout
        self.terminal = sys.stdout

    def write(self, message):
        # Write to both terminal and file
        self.terminal.write(message)

        # Write to file if it exists
        if self.log_file:
            # Add timestamp for important lines (those ending with newline)
            if message.strip() and (message.endswith('\n') or len(message.strip()) > 10):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}] {message}")
                except:
                    pass  # Silently ignore file write errors
            else:
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(message)
                except:
                    pass  # Silently ignore file write errors

    def flush(self):
        # Flush both terminal and file
        self.terminal.flush()
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.flush()
            except:
                pass

# Initialize logger for this process
logger = Logger()
sys.stdout = logger
sys.stderr = logger

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0
        self.update_counter = 0  # Count update calls

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()
    
    def reproject_debug(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        # coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        coords = pops.transform_debug(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        # coords = self.reproject(indicies=(ii, jj, kk))
        coords = self.reproject_debug(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def update(self):
        # Increment update counter
        self.update_counter += 1

        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def update_debug(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject_debug()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]


    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        # Print frame separator with timestamp
        print("\n" + "="*80)
        print(f"[FRAME] Processing Frame #{tstamp}")
        print("="*80)

        # Store current timestamp for use in update()
        self.current_timestamp = tstamp

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())
        # print(f"{tstamp},image: {self.image_.shape},poses: {self.pg.poses_.shape},points: {self.pg.points_.shape},colors: {self.pg.colors_.shape}")

        # Store original image for visualization if this is frame 7 (last initialization frame)
        if self.n == 7:
            # Make a copy before normalization
            self.init_frame_image = image.cpu().numpy()  # Store original image as HWC uint8

        ## image/intrinsics
        # save_image(tstamp,image)
        # save_intrinsics(tstamp,intrinsics)

        image = 2 * (image[None,None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
            
        # print(f"ts: {tstamp},image: {image.shape},fmap: {fmap.shape},gmap: {gmap.shape},imap: {imap.shape},patches: {patches.shape},clr: {clr.shape}")

        # pred_feature = (fmap, gmap, imap, patches, _, clr)
        # save_features(tstamp,pred_feature)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1       
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            # self.update_debug()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
        
        # region debug
        # index = self.n - 1
        # pg_tstamp = self.pg.tstamps_[index]
        # pg_pose = self.pg.poses_[index]
        # print(f"tstamp: {tstamp},index: {index},pg_stamp: {pg_tstamp},pose: {pg_pose}")
        # endregion

        # save_patches(tstamp,self.pg.patches_)
        # print(f"tstamp: {tstamp},patches: {self.pg.patches_.shape}")

        # Print frame statistics using dedicated function
        self.print_frame_statistics(tstamp)

        # Visualize feature points from network output

        # simple_dict = {
        #     "pmem" : self.pg.pmem,
        #     "DIM" : self.pg.DIM,
        #     "n" : self.n,
        #     "m" : self.m,
        #     "M" : self.M,
        #     "N" : self.N
        # }
        # save_simple_dict(tstamp,simple_dict)
        # save_tstamp(tstamp,self.pg.tstamps_)
        # save_poses(tstamp,self.pg.poses_)
        # save_points(tstamp,self.pg.points_)
        # save_colors(tstamp,self.pg.colors_)
        # print(f"tstamp: {tstamp}")
        # print(f"tstamp: {tstamp},n: {self.pg.n},m: {self.pg.m},M: {self.pg.M},N: {self.pg.N},poses: {array_save_poses.shape},points: {array_save_points.shape},colors: {array_save_colors.shape}")

    def print_frame_statistics(self, tstamp):
        """
        Print comprehensive frame statistics including poses, points, and depth information.

        Args:
            tstamp: Timestamp of the current frame
        """
        print(f"\n[FRAME STATS] Timestamp: {tstamp}")

        # 1. Count valid poses
        valid_poses = 0
        if hasattr(self, 'poses') and self.poses is not None:
            for i in range(min(self.n, self.poses.shape[1])):
                pose = self.poses[0, i] if self.poses.dim() == 3 else self.poses[i]
                if i == 0:
                    valid_poses += 1  # First frame is always valid (origin)
                else:
                    # Check if pose is not identity
                    if not torch.allclose(pose[3:7], torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device)):
                        valid_poses += 1

        print(f"  Valid poses: {valid_poses}")
        print(f"  Total points: {self.m}")

        # 2. Process inverse depth and depth statistics
        if hasattr(self, 'patches') and self.patches is not None and self.m > 0:
            # Extract center pixel of each patch
            inv_depths = self.patches[0, :self.m, 2, 1, 1]  # Shape: [m]

            # Count valid 3D points based on inverse depth range
            valid_mask = (inv_depths >= 0.02) & (inv_depths <= 5.0)
            valid_3d_points = valid_mask.sum().item()
            print(f"  Valid 3D points: {valid_3d_points}")

            # Inverse depth statistics
            inv_depth_min = inv_depths.min().item()
            inv_depth_max = inv_depths.max().item()
            inv_depth_mean = inv_depths.mean().item()
            inv_depth_median = inv_depths.median().item()
            inv_depth_std = inv_depths.std().item()

            # Calculate percentiles
            inv_depths_sorted, _ = torch.sort(inv_depths)
            n = len(inv_depths_sorted)
            p25 = inv_depths_sorted[int(n * 0.25)].item()
            p75 = inv_depths_sorted[int(n * 0.75)].item()
            p95 = inv_depths_sorted[int(n * 0.95)].item()

            # Count outliers
            valid_range_mask = (inv_depths >= 0.02) & (inv_depths <= 5.0)
            valid_count = valid_range_mask.sum().item()
            outlier_low_count = (inv_depths < 0.02).sum().item()
            outlier_high_count = (inv_depths > 5.0).sum().item()

            print(f"  Inverse depth stats:")
            print(f"    Range: {inv_depth_min:.6f} - {inv_depth_max:.6f}")
            print(f"    Mean±Std: {inv_depth_mean:.6f}±{inv_depth_std:.6f}")
            print(f"    Median: {inv_depth_median:.6f}")
            print(f"    Percentiles: 25%={p25:.6f}, 75%={p75:.6f}, 95%={p95:.6f}")
            print(f"    Valid range (0.02-5.0): {valid_count}/{n} ({100*valid_count/n:.1f}%)")
            if outlier_low_count > 0 or outlier_high_count > 0:
                print(f"    Outliers: <0.02={outlier_low_count}, >5.0={outlier_high_count}")

            # Depth statistics
            valid_inv_depths = inv_depths[(inv_depths >= 0.02) & (inv_depths <= 5.0)]
            if len(valid_inv_depths) > 0:
                depths = 1.0 / valid_inv_depths

                # Calculate depth statistics
                depth_min = depths.min().item()
                depth_max = depths.max().item()
                depth_mean = depths.mean().item()
                depth_median = depths.median().item()
                depth_std = depths.std().item()

                # Calculate depth percentiles
                depths_sorted, _ = torch.sort(depths)
                n_depth = len(depths_sorted)
                depth_p25 = depths_sorted[int(n_depth * 0.25)].item()
                depth_p75 = depths_sorted[int(n_depth * 0.75)].item()
                depth_p95 = depths_sorted[int(n_depth * 0.95)].item()

                # Count depth outliers
                depth_valid_mask = (depths >= 0.2) & (depths <= 50.0)
                depth_valid_count = depth_valid_mask.sum().item()
                depth_outlier_low = (depths < 0.2).sum().item()
                depth_outlier_high = (depths > 50.0).sum().item()

                print(f"  Depth stats:")
                print(f"    Range: {depth_min:.6f} - {depth_max:.6f}m")
                print(f"    Mean±Std: {depth_mean:.6f}±{depth_std:.6f}m")
                print(f"    Median: {depth_median:.6f}m")
                print(f"    Percentiles: 25%={depth_p25:.6f}m, 75%={depth_p75:.6f}m, 95%={depth_p95:.6f}m")
                print(f"    Valid range (0.2-50m): {depth_valid_count}/{n_depth} ({100*depth_valid_count/n_depth:.1f}%)")
                if depth_outlier_low > 0 or depth_outlier_high > 0:
                    print(f"    Outliers: <0.2m={depth_outlier_low}, >50m={depth_outlier_high}")

    def debug_extract(self, tstamp, image, intrinsics):

        print(f"============= frame_time: {tstamp} ===============")
        # Store original image for visualization
        original_image = image.clone()

        image = 2 * (image[None,None] / 255.0) - 0.5
        patches_per_frame = 96
        centroid_sel_strat = 'RANDOM'

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr, coords = \
                self.network.patchify.debug(image,
                    patches_per_image=patches_per_frame,
                    centroid_sel_strat=centroid_sel_strat,
                    return_color=True)

        # Visualize coordinates on original image and save
        self.visualize_coords_on_image(tstamp, original_image, coords)

    def visualize_coords_on_image(self, tstamp, image, coords):
        """Visualize extracted coordinates on the original image"""
        import os
        import cv2
        import numpy as np

        # Convert original image to proper format for visualization
        # The original image should be in [H, W, 3] format with values 0-255
        if isinstance(image, np.ndarray):
            vis_image = image.copy()
        else:
            # Handle tensor input - convert from CPU tensor to numpy
            vis_image = image.cpu().numpy()

        # Ensure image is in HWC format and correct data type
        if len(vis_image.shape) == 3 and vis_image.shape[0] == 3:  # CHW format
            vis_image = np.transpose(vis_image, (1, 2, 0))  # Convert to HWC

        # Convert to uint8 and ensure proper range
        if vis_image.dtype != np.uint8:
            # If values are in 0-1 range, scale to 0-255
            if vis_image.max() <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = np.clip(vis_image, 0, 255).astype(np.uint8)

        # Get coordinates for the first frame
        frame_coords = coords[0].cpu().numpy()  # Shape: [patches_per_frame, 2]

        # Scale coordinates back to original image resolution
        # The coords are in feature space (1/4 resolution), so scale by 4
        h, w = vis_image.shape[:2]
        frame_coords[:, 0] = frame_coords[:, 0] * 4  # x coordinate
        frame_coords[:, 1] = frame_coords[:, 1] * 4  # y coordinate

        # Draw coordinates as circles on the image
        for i, (x, y) in enumerate(frame_coords):
            if 0 <= x < w and 0 <= y < h:  # Ensure coordinates are within image bounds
                # Draw small circle with different colors for visibility
                cv2.circle(vis_image, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green circles
                # Optional: add a small border around each point
                cv2.circle(vis_image, (int(x), int(y)), 3, (255, 255, 255), 1)  # White border

        # Add text showing the number of feature points
        num_points = len(frame_coords)
        text = f"Feature Points: {num_points}"
        text_position = (10, 30)  # Top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        # Add text with background for better visibility
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(vis_image,
                     (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                     (text_position[0] + text_size[0] + 5, text_position[1] + 5),
                     (0, 0, 0), -1)  # Black background
        cv2.putText(vis_image, text, text_position, font, font_scale, (255, 255, 255), font_thickness)  # White text

        # Add frame timestamp
        timestamp_text = f"Frame: {tstamp}"
        timestamp_position = (10, 70)  # Below the feature points text
        cv2.rectangle(vis_image,
                     (timestamp_position[0] - 5, timestamp_position[1] - text_size[1] - 5),
                     (timestamp_position[0] + cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)[0][0] + 5, timestamp_position[1] + 5),
                     (0, 0, 0), -1)  # Black background
        cv2.putText(vis_image, timestamp_text, timestamp_position, font, font_scale, (255, 255, 255), font_thickness)  # White text

        # Create debug directory if it doesn't exist
        debug_dir = "/home/jerett/Project/DPVO/Debug/coords"
        os.makedirs(debug_dir, exist_ok=True)

        # Save the visualization using RGB format (correct version)
        output_image = vis_image.copy()

        # Create 8-digit padded frame ID
        frame_id = f"{int(tstamp):08d}"
        output_path = os.path.join(debug_dir, f"{frame_id}.png")

        cv2.imwrite(output_path, output_image)

        print(f"Saved coordinate visualization to: {output_path}")
        print(f"Visualized {num_points} coordinates on frame {frame_id}")

    def initialized(self, tstamp, image, intrinsics):
        if self.is_initialized:
            return

        # Store original image for visualization
        original_image = image.clone()

        ### 1.extract feature ###
        image = 2 * (image[None,None] / 255.0) - 0.5
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
        ### 2.update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        ### 3.depth linear ###
        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec

        ### 4.depth initialization ###
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        self.pg.patches_[self.n] = patches

        # Visualize extracted coordinates on the original image
        self.visualize_initialized_coordinates(tstamp, original_image, patches)

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return
        self.n += 1
        self.m += self.M

        ### 5.Add forward and backward factors ###
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        ### 6.Init ###
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True
            print(f"\n[INIT] Starting initialization optimization - Frame {self.n}")
            print(f"[INIT] Will run 12 update iterations for initial bundle adjustment")

            for itr in range(12):
                print(f"\n[INIT] Iteration {itr+1}/12")
                self.update()
                # Save pose and point cloud after each update
                self.save_init_state(itr)
            print(f"\n[INIT] Initialization optimization completed")

    def visualize_initialized_coordinates(self, tstamp, image, patches):
        """Visualize extracted patch coordinates on the original image"""
        print(f"\n[VIS_INIT] Visualizing initialized coordinates for frame {self.n} (timestamp: {tstamp})")

        # Convert patches to coordinates
        # patches shape: [1, N, 3, 1, 1] where last dims are [x, y, depth]
        coords = patches[0, :, :2, 0, 0].cpu().numpy()  # Extract x, y coordinates

        # Scale coordinates to match original image size
        # The patches are extracted from downscaled feature maps (divided by RES)
        coords_scaled = coords * self.RES

        # Convert image to numpy for OpenCV
        img_np = image.cpu().numpy()

        # Handle different image formats
        # Original image is typically in CHW format with values 0-255 (uint8)
        if len(img_np.shape) == 3:
            if img_np.shape[0] == 3:
                # CHW format, convert to HWC
                img_np = np.transpose(img_np, (1, 2, 0))

            # Ensure the image is in uint8 format for proper display
            if img_np.dtype != np.uint8:
                # If float values, scale to 0-255 and convert
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        img_vis = img_np.copy()

        # Draw coordinates on the image
        num_points = len(coords_scaled)
        print(f"[VIS_INIT] Total coordinates: {num_points}")
        print(f"[VIS_INIT] Image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
        print(f"[VIS_INIT] RES: {self.RES}")
        print(f"[VIS_INIT] First 5 coordinates (scaled): {coords_scaled[:5]}")

        # Draw all coordinates but use smaller circles for less clutter
        for i in range(num_points):
            x, y = coords_scaled[i]
            x_int, y_int = int(x), int(y)

            # Check if coordinates are within image bounds
            if 0 <= x_int < img_vis.shape[1] and 0 <= y_int < img_vis.shape[0]:
                # Draw a small red circle (in BGR format)
                cv2.circle(img_vis, (x_int, y_int), 2, (0, 0, 255), -1)

                # Add index number for first 10 points only
                if i < 10:
                    cv2.putText(img_vis, f"{i}", (x_int + 3, y_int - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # Save the visualization
        output_dir = "/home/jerett/Project/DPVO/Debug/Data"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"frame_{tstamp:06d}_init_coords.png"
        output_path = os.path.join(output_dir, filename)

        # Save the image as-is (assume it's already in the correct format)
        cv2.imwrite(output_path, img_vis)

        print(f"[VIS_INIT] Saved initialization visualization to: {output_path}")
        print(f"[VIS_INIT] Visualized {num_points} coordinates on original image")

    def save_init_state(self, iteration):
        """Save poses and point cloud during initialization iterations"""
        import os
        import numpy as np

        print(f"[INIT_SAVE] Saving state for iteration {iteration}")

        # Create output directory
        output_dir = "/home/jerett/Project/DPVO/Debug/Init"
        os.makedirs(output_dir, exist_ok=True)

        # Save poses
        # Convert poses from SE3 to transformation matrix
        poses_se3 = SE3(self.poses[:, :self.n])  # Get poses for initialized frames
        poses_np = poses_se3.matrix().cpu().numpy()  # Convert to numpy [1, N, 4, 4]

        # Save poses as TUM format
        tum_file = os.path.join(output_dir, f"poses_iter_{iteration:02d}.txt")
        with open(tum_file, 'w') as f:
            for i in range(poses_np.shape[1]):  # poses_np shape: [1, N, 4, 4]
                timestamp = self.tlist[i] if i < len(self.tlist) else float(i)
                pose_matrix = poses_np[0, i]  # [4, 4]

                # Extract translation
                tx, ty, tz = pose_matrix[:3, 3]

                # Extract quaternion (x, y, z, w)
                # Convert rotation matrix to quaternion using simple method
                import scipy.spatial.transform as st
                rotation = st.Rotation.from_matrix(pose_matrix[:3, :3])
                quat = rotation.as_quat()  # (x, y, z, w)
                qx, qy, qz, qw = quat

                f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        # Save point cloud
        # Extract 3D points from patches
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points_3d = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3).cpu().numpy()

        # Get colors for points
        colors = self.pg.colors_[:self.m].cpu().numpy().reshape(-1, 3)

        # Save point cloud as PLY format
        ply_file = os.path.join(output_dir, f"points_iter_{iteration:02d}.ply")
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(len(points_3d)):
                x, y, z = points_3d[i]
                r, g, b = colors[i] if i < len(colors) else (255, 0, 0)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

        # Save as JSON for easier processing
        json_file = os.path.join(output_dir, f"state_iter_{iteration:02d}.json")
        import json
        state_data = {
            'iteration': iteration,
            'poses': poses_np[0].tolist(),  # [N, 4, 4]
            'points': points_3d.tolist(),
            'colors': colors.tolist(),
            'timestamps': self.tlist[:self.n],
            'frame_count': self.n,
            'patch_count': self.m
        }

        with open(json_file, 'w') as f:
            json.dump(state_data, f, indent=2)

        print(f"[INIT_SAVE] Saved iteration {iteration}:")
        print(f"  - Poses (TUM): {tum_file}")
        print(f"  - Point cloud (PLY): {ply_file}")
        print(f"  - State (JSON): {json_file}")
        print(f"  - Frames: {self.n}, Points: {len(points_3d)}")

    def get_frame_statistics(self):
        """
        Get comprehensive frame statistics and return valid poses and points

        Returns:
            dict: Dictionary containing statistics and valid data, or None if not available
            {
                'n': int,                    # Total frames processed
                'valid_poses': int,           # Number of valid poses
                'valid_poses_data': list,      # List of valid pose tensors
                'total_points': int,          # Total number of 3D points
                'valid_3d_points': int,        # Number of points with valid depth
                'valid_3d_points_data': numpy.ndarray,  # Valid 3D points [N, 3]
                'valid_3d_points_colors': numpy.ndarray,  # RGB colors for valid points [N, 3]
                'status': str,                # 'INITIALIZING' or 'TRACKING'
                'is_initialized': bool        # Whether SLAM is initialized
            }
        """
        import numpy as np

        if not hasattr(self, 'pg'):
            return None

        # Initialize return data
        stats = {
            'n': self.n,
            'valid_poses': 0,
            'valid_poses_data': [],
            'total_points': 0,
            'valid_3d_points': 0,
            'valid_3d_points_data': None,
            'valid_3d_points_colors': None,
            'status': "INITIALIZING" if not self.is_initialized else "TRACKING",
            'is_initialized': self.is_initialized
        }

        # Calculate total number of 3D points
        stats['total_points'] = self.m  # self.m = n * M (total patches processed)

        # Extract and count valid 3D points
        try:
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m],
                                    self.intrinsics, self.ix[:self.m])

            # points shape should be [N, 3, 3, 4] where N is number of points
            # Extract center patch and convert to 3D coordinates (same as line 443)
            if points.dim() >= 4 and points.shape[-2:] == torch.Size([3, 4]):
                # Extract center pixel (1, 1) from each 3x3 patch
                points_center = points[..., 1, 1, :]  # [N, 4] in homogeneous coordinates

                # Convert from homogeneous to 3D coordinates: xyz / w
                xyz = points_center[:, :3]  # [N, 3]
                w = points_center[:, 3:4]  # [N, 1]

                # Calculate actual depth (z coordinate after division)
                w_safe = torch.clamp(w, min=1e-8)
                points_3d = xyz / w_safe  # [N, 3]
                depth_values = points_3d[:, 2]  # z values (depth)

                # Filter valid depth values (0.2 to 50 meters)
                valid_mask = torch.isfinite(depth_values) & (depth_values > 0.2) & (depth_values < 50.0)
                stats['valid_3d_points'] = valid_mask.sum().item()

                if stats['valid_3d_points'] > 0:
                    stats['valid_3d_points_data'] = points_3d[valid_mask].cpu().numpy()
            else:
                # Fallback for unexpected shape
                stats['valid_3d_points'] = 0
                stats['valid_3d_points_data'] = None
                stats['valid_3d_points_colors'] = None

            # Get colors for valid points
            if stats['valid_3d_points'] > 0 and hasattr(self.pg, 'colors_') and self.pg.colors_ is not None:
                try:
                    valid_colors = []

                    # Get indices of valid points
                    valid_point_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()

                    # Map points to their source frames
                    for frame_idx in range(min(self.n, len(self.pg.colors_))):
                        frame_start = frame_idx * self.M
                        frame_end = min(frame_start + self.m, len(valid_point_indices))

                        # Get valid points that belong to this frame
                        frame_mask = (valid_point_indices >= frame_start) & (valid_point_indices < frame_end)
                        frame_valid_indices = valid_point_indices[frame_mask] - frame_start

                        if len(frame_valid_indices) > 0:
                            colors = self.pg.colors_[frame_idx].cpu().numpy()
                            # Ensure indices don't exceed colors array size
                            valid_frame_indices = frame_valid_indices[frame_valid_indices < colors.shape[0]]
                            valid_colors.extend(colors[valid_frame_indices])

                    # Convert to numpy array
                    if valid_colors:
                        stats['valid_3d_points_colors'] = np.array(valid_colors[:stats['valid_3d_points']])
                    else:
                        stats['valid_3d_points_colors'] = None

                except Exception as e:
                    print(f"[WARNING] Failed to extract colors: {e}")
                    stats['valid_3d_points_colors'] = None

        except Exception as e:
            print(f"[WARNING] Failed to extract valid 3D points: {e}")
            stats['valid_3d_points'] = 0
            stats['valid_3d_points_data'] = None
            stats['valid_3d_points_colors'] = None

        # Count valid poses
        if hasattr(self, 'poses') and self.poses is not None:
            stats['valid_poses_data'] = []

            for i in range(min(self.n, self.poses.shape[1])):
                pose = self.poses[0, i] if self.poses.dim() == 3 else self.poses[i]

                if i == 0:
                    # First frame is always considered valid (origin)
                    stats['valid_poses'] += 1
                    stats['valid_poses_data'].append(pose.clone())
                else:
                    # For other frames, check if they've been estimated (not identity)
                    if not torch.allclose(pose[3:7], torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device)):
                        stats['valid_poses'] += 1
                        stats['valid_poses_data'].append(pose.clone())

        return stats
