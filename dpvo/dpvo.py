import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import sys
import os
from datetime import datetime

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

        # Use a lock file mechanism to ensure only one log file per run
        if Logger._log_file is None:
            lock_file = os.path.join(log_dir, ".dpvo_logger_lock")

            # Try to create lock file (atomic operation)
            try:
                # Try to open file in exclusive mode
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # We got the lock, create log file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                Logger._log_file = os.path.join(log_dir, f"dpvo_log_{timestamp}.txt")
                os.close(fd)
                # Write the log file path to lock file
                with open(lock_file, 'w') as f:
                    f.write(Logger._log_file)
            except FileExistsError:
                # Lock file already exists, read the log file path
                with open(lock_file, 'r') as f:
                    Logger._log_file = f.read().strip()
            except Exception:
                # Fallback: create unique log file with PID
                if Logger._log_file is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pid = os.getpid()
                    Logger._log_file = os.path.join(log_dir, f"dpvo_log_{timestamp}_pid{pid}.txt")

        self.log_file = Logger._log_file

        # Keep original stdout
        self.terminal = sys.stdout

    def write(self, message):
        # Write to both terminal and file
        self.terminal.write(message)
        # Add timestamp for important lines (those ending with newline)
        if message.strip() and (message.endswith('\n') or len(message.strip()) > 10):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}")
        else:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message)

    def flush(self):
        # Flush both terminal and file
        self.terminal.flush()
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.flush()
        except:
            pass

# Initialize logger for this process
logger = Logger()
sys.stdout = logger
sys.stderr = logger

import atexit
def cleanup_lock():
    """Clean up the lock file when program exits"""
    try:
        lock_file = os.path.join("/home/jerett/Project/DPVO/Debug/Log", ".dpvo_logger_lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except:
        pass

# Register cleanup function
atexit.register(cleanup_lock)

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

        # Print current frame timestamp if available
        if hasattr(self, 'current_timestamp'):
            print(f"\n=== Update #{self.update_counter} - Frame {self.n}, Timestamp: {self.current_timestamp} ===")
            print(f"    Initialized: {self.is_initialized}")
        else:
            print(f"\n=== Update #{self.update_counter} - Frame {self.n} ===")
            print(f"    Initialized: {self.is_initialized}")

        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            # Print coords information
            print(f"1. coords:")
            print(f"   - Shape: {coords.shape}")
            if coords.numel() > 0:
                # Debug: print the actual structure
                print(f"   - Debug: self.P = {self.P}")
                print(f"   - Debug: coords[...,self.P//2,self.P//2,:].shape = {coords[...,self.P//2,self.P//2,:].shape}")

                # Get first 5 center coordinates correctly
                # coords shape: [1, N, 2, 3, 3] -> we want first 5 entries of the center pixels
                center_coords = coords[0, :5, self.P//2, self.P//2, :2]  # Take first 5, center pixel, u,v

                # Print first 5 correctly
                print(f"   - First 5 center coordinates (u, v):")
                for i in range(center_coords.shape[0]):
                    u, v = center_coords[i, 0].item(), center_coords[i, 1].item()
                    print(f"     [{i}]: u={u:.2f}, v={v:.2f}")

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            # Print delta information
            print(f"2. delta:")
            print(f"   - Shape: {delta.shape}")
            if delta.numel() > 0:
                # Get first 5 delta values
                delta_flat = delta.flatten(0, 1)[:5]
                print(f"   - First 5 delta values (du, dv):")
                for i, d in enumerate(delta_flat):
                    du, dv = d[0].item(), d[1].item()
                    print(f"     [{i}]: du={du:.3f}, dv={dv:.3f}")

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

            # Print target information
            print(f"3. target:")
            print(f"   - Shape: {target.shape}")
            if target.numel() > 0:
                # Get first 5 target coordinates
                target_flat = target.flatten(0, 1)[:5]
                print(f"   - First 5 target coordinates (u, v):")
                for i, t in enumerate(target_flat):
                    u, v = t[0].item(), t[1].item()
                    print(f"     [{i}]: u={u:.2f}, v={v:.2f}")

            # Calculate and print residual
            # coords has 3 channels, target has 2 channels, so use only first 2 channels of coords
            coords_center = coords[...,self.P//2,self.P//2,:2]  # Take only u,v channels
            residual = target - coords_center
            print(f"4. residual r:")
            print(f"   - Shape: {residual.shape}")
            if residual.numel() > 0:
                # Get first 5 residual values
                residual_flat = residual.flatten(0, 1)[:5]
                print(f"   - First 5 residual values (ru, rv):")
                for i, r in enumerate(residual_flat):
                    ru, rv = r[0].item(), r[1].item()
                    rmag = torch.sqrt(r[0]**2 + r[1]**2).item()
                    print(f"     [{i}]: ru={ru:.3f}, rv={rv:.3f}, |r|={rmag:.3f}")
                print(f"   - Mean residual magnitude: {torch.norm(residual, dim=-1).mean().item():.3f}")

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

        # Store current timestamp for use in update()
        self.current_timestamp = tstamp

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())
        # print(f"{tstamp},image: {self.image_.shape},poses: {self.pg.poses_.shape},points: {self.pg.points_.shape},colors: {self.pg.colors_.shape}")

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
            
        print(f"ts: {tstamp},image: {image.shape},fmap: {fmap.shape},gmap: {gmap.shape},imap: {imap.shape},patches: {patches.shape},clr: {clr.shape}")

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
        print(f"tstamp: {tstamp},patches: {self.pg.patches_.shape}")

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
        ### 1.extract feature ###
        image = 2 * (image[None,None] / 255.0) - 0.5
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
        ### 2.update state attributes ###
        print(f"=== frame,tstamp: {tstamp},n: {self.n},counter: {self.counter} ===")
        print(f"image: {image.shape}")
        print(f"fmap: {fmap.shape}")
        print(f"gmap: {gmap.shape}")
        print(f"imap: {imap.shape}")
        print(f"patches: {patches.shape}")
        print(f"clr: {clr.shape}")
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
                print(f"P1: {P1},P2: {P2}")

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec

        ### 4.depth initialization ###
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
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

            print(f"\n[INIT] Initialization optimization completed")

