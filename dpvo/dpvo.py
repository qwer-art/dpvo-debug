import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import sys
import os
from datetime import datetime
import cv2  # For visualization


def get_jet_color(value):
    """
    Generate matplotlib jet colormap color based on normalized value [0, 1].

    Args:
        value (float): Normalized value between 0 and 1

    Returns:
        tuple: BGR color tuple (blue, green, red)
    """
    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))

    # matplotlib jet colormap: blue -> cyan -> green -> yellow -> red
    # This implementation follows matplotlib's color mapping
    if value < 0.25:
        # Blue to Cyan (0.0 -> 0.25)
        t = value * 4.0
        r = 0
        g = t
        b = 1.0
    elif value < 0.5:
        # Cyan to Green (0.25 -> 0.5)
        t = (value - 0.25) * 4.0
        r = 0
        g = 1.0
        b = 1.0 - t
    elif value < 0.75:
        # Green to Yellow (0.5 -> 0.75)
        t = (value - 0.5) * 4.0
        r = t
        g = 1.0
        b = 0
    else:
        # Yellow to Red (0.75 -> 1.0)
        t = (value - 0.75) * 4.0
        r = 1.0
        g = 1.0 - t
        b = 0

    # Convert to 0-255 range and return as BGR for OpenCV
    return (int(b * 255), int(g * 255), int(r * 255))


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
                with open(Logger._log_file, "w", encoding="utf-8") as f:
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
            if message.strip() and (
                message.endswith("\n") or len(message.strip()) > 10
            ):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{timestamp}] {message}")
                except:
                    pass  # Silently ignore file write errors
            else:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(message)
                except:
                    pass  # Silently ignore file write errors

    def flush(self):
        # Flush both terminal and file
        self.terminal.flush()
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.flush()
            except:
                pass


# Initialize logger for this process
logger = Logger()
sys.stdout = logger
sys.stderr = logger

mp.set_start_method("spawn", True)


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

        self.ht = ht  # image height
        self.wd = wd  # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.nkframe_tstamps = []
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
        self.pmem = self.mem = 36  # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000  # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE  # patch memory

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

        # keyframe pose
        self.last_pose = SE3.Identity(1, device="cuda")
        self.distance = 0.

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
                    new_state_dict[k.replace("module.", "")] = v

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
            self.image_, self.pg.poses_, self.pg.points_, self.pg.colors_, intrinsics_
        )

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N * self.M, 3, 3, 3)

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

    def get_lnkframe_tstamp(self, t):
        if len(self.nkframe_tstamps) == 0:
            return -1
        return self.nkframe_tstamps[-1]

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
        """local correlation volume"""
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """reproject patch k from i -> j"""
        (ii, jj, kk) = (
            indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        )
        coords = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
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
            self.pg.weight_inac = torch.cat(
                (self.pg.weight_inac, self.pg.weight[:, m]), dim=1
            )
            self.pg.target_inac = torch.cat(
                (self.pg.target_inac, self.pg.target[:, m]), dim=1
            )
        self.pg.weight = self.pg.weight[:, ~m]
        self.pg.target = self.pg.target[:, ~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:, ~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """kinda hacky way to ensure enough motion for initialization"""
        kk = torch.arange(self.m - self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:, kk % (self.M * self.pmem)]
            net, (delta, weight, _) = self.network.update(
                net, ctx, corr, None, ii, jj, kk
            )

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5
        )
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k - 1]
            t1 = self.pg.tstamps_[k]
            self.nkframe_tstamps.append(t1)

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i + 1]
                self.pg.colors_[i] = self.pg.colors_[i + 1]
                self.pg.poses_[i] = self.pg.poses_[i + 1]
                self.pg.patches_[i] = self.pg.patches_[i + 1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i + 1]

                self.imap_[i % self.pmem] = self.imap_[(i + 1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i + 1) % self.pmem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0, (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0, (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = (
            self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
        )  # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (
                self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW)
            )
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """Global bundle adjustment
        Includes both active and inactive edges"""
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(
            self.poses,
            self.patches,
            self.intrinsics,
            full_target,
            full_weight,
            lmbda,
            full_ii,
            full_jj,
            full_kk,
            t0,
            self.n,
            M=self.M,
            iterations=2,
            eff_impl=True,
        )
        self.ran_global_ba[self.n] = True

    def update(self):
        # Increment update counter
        self.update_counter += 1

        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = self.network.update(
                    self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk
                )

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[..., self.P // 2, self.P // 2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (
                    self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1
                ).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = (
                        self.n - self.cfg.OPTIMIZATION_WINDOW
                        if self.is_initialized
                        else 1
                    )
                    t0 = max(t0, 1)
                    fastba.BA(
                        self.poses,
                        self.patches,
                        self.intrinsics,
                        target,
                        weight,
                        lmbda,
                        self.pg.ii,
                        self.pg.jj,
                        self.pg.kk,
                        t0,
                        self.n,
                        M=self.M,
                        iterations=2,
                        eff_impl=False,
                    )
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(
                SE3(self.poses),
                self.patches[:, : self.m],
                self.intrinsics,
                self.ix[: self.m],
            )
            points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.pg.points_[: len(points)] = points[:]

    def __edges_forw(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp, image, intrinsics):
        """track new frame"""

        # Print frame separator with timestamp
        print("\n" + "=" * 80)
        print(f"[FRAME] Processing Frame #{tstamp}")
        print("=" * 80)

        # Store current timestamp for use in update()
        self.current_timestamp = tstamp

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"'
            )

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())
        # print(f"{tstamp},image: {self.image_.shape},poses: {self.pg.poses_.shape},points: {self.pg.points_.shape},colors: {self.pg.colors_.shape}")

        # Store original image for visualization
        original_image = image.clone()  # Store before normalization

        # Store original image for initialization if this is frame 7
        if self.n == 7:
            self.init_frame_image = (
                original_image.cpu().numpy()
            )  # Store original image as HWC uint8

        ## image/intrinsics
        # save_image(tstamp,image)
        # save_intrinsics(tstamp,intrinsics)

        image = 2 * (image[None, None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                image,
                patches_per_image=self.cfg.PATCHES_PER_FRAME,
                centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                return_color=True,
            )

        # print(f"ts: {tstamp},image: {image.shape},fmap: {fmap.shape},gmap: {gmap.shape},imap: {imap.shape},patches: {patches.shape},clr: {clr.shape}")

        # pred_feature = (fmap, gmap, imap, patches, _, clr)
        # save_features(tstamp,pred_feature)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == "DAMPED_LINEAR":
                P1 = SE3(self.pg.poses_[self.n - 1])
                P2 = SE3(self.pg.poses_[self.n - 2])

                # To deal with varying camera hz
                *_, a, b, c = [1] * 3 + self.tlist
                fac = (c - b) / (b - a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n - 1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

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
                """Add loop closure factors"""
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
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()

        # save_patches(tstamp,self.pg.patches_)
        # print(f"tstamp: {tstamp},patches: {self.pg.patches_.shape}")

        # Update pose tracking if initialized
        if self.is_initialized and self.n > 1:
            current_pose = SE3(self.pg.poses_[self.n - 1:self.n])  # Keep batch dimension
            motion_distance = (current_pose * self.last_pose.inv()).log().norm().item()
            self.distance += motion_distance
            self.last_pose = current_pose

        # Print frame statistics using dedicated function
        self.print_frame_statistics(tstamp, original_image)

        # Visualize feature points on current frame
        self.visualize_feature_points(tstamp, original_image)

    def print_frame_statistics(self, tstamp, original_image):
        """
        Print comprehensive frame statistics including poses, points, and depth information.

        Args:
            tstamp: Timestamp of the current frame
        """
        print(
            f"\n[FRAME STATS] Timestamp: {tstamp},Distance: {self.distance:.3f},Keyframes/Counter: {self.n}/{self.counter}"
        )
        print(f"[State],init: {self.is_initialized}")
        ### 1. Pose
        # 使用map key的方式去重，获取所有有效时间戳，一堆无效的都聚集在0上
        kframe_trajs = {}
        for i in range(self.n):
            kframe_trajs[self.pg.tstamps_[i]] = self.pg.poses_[i]

        nkframe_tstamp = self.get_lnkframe_tstamp(tstamp)
        print(
            f"[Pose],kframe_size: {len(kframe_trajs)},last_notkey_tstamp: {nkframe_tstamp}"
        )
        # 使用正确的pose获取方式：通过轨迹插值获取当前帧的准确pose
        curr_tstamp = self.pg.tstamps_[self.n - 1]
        curr_pose = self.pg.poses_[self.n - 1]
        print(
            f"[Pose],tstamp: {tstamp},curr_tstamp: {curr_tstamp},curr_pose: {curr_pose}"
        )

        ### 2. Patch
        patch_pixels = self.patches[:, : self.m, :2, 1, 1]
        patch_inv_depths = self.patches[:, : self.m, 2, 1, 1]
        image_bgr = original_image.cpu().permute(1, 2, 0).numpy()

        print(
            f"[Patch],pixels: {patch_pixels.shape},inv_depths: {patch_inv_depths.shape}"
        )
        print(f"[Patch],image_shape: {image_bgr.shape}")
        minu = int(patch_pixels[..., 0].min().item())
        maxu = int(patch_pixels[..., 0].max().item())
        minv = int(patch_pixels[..., 1].min().item())
        maxv = int(patch_pixels[..., 1].max().item())

        print(
            f"[Patch],batch_range,top_left: [{minu},{minv}],bottom_right: [{maxu},{maxv}]"
        )

        inv_depth_min = patch_inv_depths.min().item()
        inv_depth_max = patch_inv_depths.max().item()

        # Calculate depth statistics with protection against division by zero
        depth_min = 1.0 / (inv_depth_max + 1e-6)
        depth_max = 1.0 / (inv_depth_min + 1e-6)

        print(f"[Patch] inv_depth: {inv_depth_min:.3f} - {inv_depth_max:.3f}")
        print(f"[Patch] depth: {depth_min:.3f} - {depth_max:.3f}")

        ### 3. Points
        pc_poses = SE3(self.poses)
        pc_patches = self.patches[:, : self.m]
        pc_intrinsics = self.intrinsics
        pc_ix = self.ix[: self.m]
        points = pops.point_cloud(pc_poses, pc_patches, pc_intrinsics, pc_ix)
        points_3d = (
            (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3).cpu().numpy()
        )
        print(f"[Point],points: {points.shape}")
        print(f"[Point],points_3d: {points_3d.shape}")
        kk_values = self.pg.kk[: self.m]
        ii_values = self.pg.ii[: self.m]
        pixel_coords = self.patches[0, : self.m, :2, 1, 1]
        if self.m <= 10:
            # 如果点数少于等于10个，取所有点
            sample_indices = list(range(self.m))
        else:
            # 均匀采样10个点
            step = self.m // 10
            sample_indices = [i * step for i in range(10)]

        for point_idx in sample_indices:
            point_ii = ii_values[point_idx].item()
            point_kk = kk_values[point_idx].item()
            pixel_u = int(patch_pixels[0, point_idx, 0].item())
            pixel_v = int(patch_pixels[0, point_idx, 1].item())
            point_3d = points_3d[point_idx]
            print(
                f"[Point],idx: {point_idx},ii: {point_ii},kk: {point_kk},pixel: ({pixel_u},{pixel_v}),point: ({point_3d[0]:.2f},{point_3d[1]:.2f},{point_3d[2]:.2f})"
            )

    def visualize_feature_points(self, tstamp, original_image):
        """Single frame visualization with 3 parts: 1) original image, 2) projected patches with depth colors, 3) depth colorbar."""

        image_bgr_left = original_image.cpu().permute(1, 2, 0).numpy()
        image_bgr_right = image_bgr_left.copy()

        # region left
        ## 1.可视化左侧图像
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # 白色文字
        bg_color = (0, 0, 0)  # 黑色背景

        def draw_text_with_idx(img, text, idx, text_color=(255, 255, 255)):
            """Draw text with automatic position calculation based on index"""
            y_offset = 10 + (idx - 1) * 35  # Starting at y=10, 35px spacing
            text_y_offset = 30 + (idx - 1) * 35  # Text position offset

            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            cv2.rectangle(
                img,
                (10, y_offset),
                (10 + text_size[0] + 10, y_offset + text_size[1] + 10),
                bg_color,
                -1,
            )
            cv2.putText(
                img,
                text,
                (15, text_y_offset),
                font,
                font_scale,
                text_color,
                font_thickness,
            )

        # 使用idx控制的不同文本
        draw_text_with_idx(image_bgr_left, f"tstamp: {tstamp}", 1, text_color)
        draw_text_with_idx(image_bgr_left, f"kframe: {self.n}/{self.counter}", 2, text_color)
        draw_text_with_idx(image_bgr_left, f"dist: {self.distance:.2f}", 3, text_color)


        # 可视化关键点
        if (
            self.m > 0
            and hasattr(self, "pg")
            and hasattr(self.pg, "patches_")
            and self.pg.patches_ is not None
        ):
            # 获取当前帧的关键点坐标和深度
            current_frame_idx = (
                self.n - 1 if self.n > 0 and self.n <= len(self.pg.patches_) else 0
            )

            if current_frame_idx < len(self.pg.patches_):
                # 提取当前帧的patches
                current_patches = self.pg.patches_[current_frame_idx]  # [M, 3, 3, 3]

                # 获取关键点坐标 (patches的中心位置)
                # patches格式: [x, y, depth] 在3x3的patch网格中
                keypoints = current_patches[:, :2, 1, 1]  # [M, 2] (x, y坐标)

                # 获取深度信息 (逆深度)
                inv_depths = current_patches[:, 2, 1, 1]  # [M]

                # 转换为深度值（深度 = 1 / 逆深度）
                depths = 1.0 / inv_depths

                # 过滤有效的深度值
                valid_mask = (depths > 0.2) & (depths < 50.0)  # 深度范围0.2m到50m

                # 显示关键点数量
                total_keypoints = len(keypoints)
                valid_keypoints_count = valid_mask.sum().item()
                keypoints_text = f"Keypoints: {valid_keypoints_count}/{total_keypoints}"
                draw_text_with_idx(image_bgr_left, keypoints_text, 4, text_color)

                if valid_keypoints_count > 0:
                    valid_depths = depths[valid_mask]
                    valid_keypoints = keypoints[valid_mask]

                    # 使用深度值而不是逆深度值进行颜色映射
                    depth_min = valid_depths.min().item()
                    depth_max = valid_depths.max().item()
                    depth_range = depth_max - depth_min

                    if depth_range > 0:
                        normalized_depths = (valid_depths - depth_min) / depth_range
                    else:
                        normalized_depths = torch.zeros_like(valid_depths)

                    # 可视化当前帧的所有特征点，点的颜色按照jet的方式映射深度
                    for i, (kp, depth_norm, depth_val) in enumerate(
                        zip(valid_keypoints, normalized_depths, valid_depths)
                    ):
                        x, y = kp.cpu().numpy()

                        # 缩放坐标到原始图像分辨率
                        # patches坐标可能是在下采样后的特征空间中，需要乘以RES缩放因子
                        x_scaled = int(x * self.RES)
                        y_scaled = int(y * self.RES)

                        # 确保坐标在图像范围内
                        if (
                            0 <= x_scaled < image_bgr_left.shape[1]
                            and 0 <= y_scaled < image_bgr_left.shape[0]
                        ):
                            # 使用jet颜色映射（基于深度值）
                            color = get_jet_color(depth_norm.item())

                            # 绘制关键点：黑色背景 + 彩色圆点 + 白色边框
                            cv2.circle(
                                image_bgr_left, (x_scaled, y_scaled), 6, (0, 0, 0), -1
                            )  # 黑色背景
                            cv2.circle(
                                image_bgr_left, (x_scaled, y_scaled), 5, color, -1
                            )  # 彩色圆点
                            cv2.circle(
                                image_bgr_left,
                                (x_scaled, y_scaled),
                                6,
                                (255, 255, 255),
                                1,
                            )  # 白色边框

        # 关键点中均匀采样20个点，写上具体数值
        if (
            self.m > 0
            and hasattr(self, "pg")
            and hasattr(self.pg, "patches_")
            and self.pg.patches_ is not None
        ):
            # 获取当前帧的关键点坐标和深度（复用前面的逻辑）
            current_frame_idx = (
                self.n - 1 if self.n > 0 and self.n <= len(self.pg.patches_) else 0
            )

            if current_frame_idx < len(self.pg.patches_):
                # 提取当前帧的patches
                current_patches = self.pg.patches_[current_frame_idx]  # [M, 3, 3, 3]

                # 获取关键点坐标和深度信息
                keypoints = current_patches[:, :2, 1, 1]  # [M, 2] (x, y坐标)
                inv_depths = current_patches[:, 2, 1, 1]  # [M]
                depths = 1.0 / inv_depths

                # 过滤有效的深度值
                valid_mask = (depths > 0.2) & (depths < 50.0)  # 深度范围0.2m到50m
                valid_keypoints_count = valid_mask.sum().item()

                if valid_keypoints_count > 0:
                    valid_depths = depths[valid_mask]
                    valid_keypoints = keypoints[valid_mask]

                    # 计算采样间隔
                    if valid_keypoints_count <= 20:
                        sample_indices = list(range(valid_keypoints_count))
                    else:
                        step = valid_keypoints_count // 20
                        sample_indices = [i * step for i in range(20)]

                    # 对采样点添加深度值标注
                    for i in sample_indices:
                        if i < len(valid_keypoints):
                            kp = valid_keypoints[i]
                            depth_val = valid_depths[i]

                            x, y = kp.cpu().numpy()
                            x_scaled = int(x * self.RES)
                            y_scaled = int(y * self.RES)

                            # 确保坐标在图像范围内
                            if (
                                0 <= x_scaled < image_bgr_left.shape[1]
                                and 0 <= y_scaled < image_bgr_left.shape[0]
                            ):
                                depth_text = f"{depth_val.item():.1f}m"

                                # 添加黑色背景让文字更清晰
                                (text_w, text_h), _ = cv2.getTextSize(
                                    depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                )
                                cv2.rectangle(
                                    image_bgr_left,
                                    (x_scaled + 10, y_scaled - text_h - 5),
                                    (x_scaled + 10 + text_w, y_scaled + 5),
                                    (0, 0, 0),
                                    -1,
                                )

                                # 用红色加粗文字标注深度值
                                cv2.putText(
                                    image_bgr_left,
                                    depth_text,
                                    (x_scaled + 10, y_scaled),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2,
                                )
        # endregion

        # region right
        curr_pose = self.pg.poses_[self.n - 1]

        pc_poses = SE3(self.poses)
        pc_patches = self.patches[:, : self.m]
        pc_intrinsics = self.intrinsics
        pc_ix = self.ix[: self.m]
        points = pops.point_cloud(pc_poses, pc_patches, pc_intrinsics, pc_ix)
        points_3d = (
            (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3).cpu().numpy()
        )


        ## 2.可视化右侧图像
        # 2.1 显示前7帧关键帧信息
        prev_frames = min(7, self.n)  # 最多显示前7帧

        # 在右侧图像顶部显示信息
        info_text = f"Previous: {prev_frames} / 7"
        text_size_info = cv2.getTextSize(info_text, font, font_scale, font_thickness)[0]
        cv2.rectangle(
            image_bgr_right,
            (10, 10),
            (10 + text_size_info[0] + 10, 10 + text_size_info[1] + 10),
            bg_color,
            -1,
        )
        cv2.putText(
            image_bgr_right,
            info_text,
            (15, 30),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

        # 获取所有3D点云信息
        points = pops.point_cloud(
            SE3(self.poses),
            self.patches[:, : self.m],
            self.intrinsics,
            self.ix[: self.m],
        )
        points_3d = (
            (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3).cpu().numpy()
        )
        print(f"[POINTS] All 3D points shape: {points_3d.shape}")

        # 获取当前帧索引
        current_frame_idx = self.n - 1 if self.n > 0 else 0

        # 获取当前帧的pose
        if self.n > 0:
            current_pose = SE3(
                self.poses[0, current_frame_idx]
                if self.poses.dim() == 3
                else self.poses[current_frame_idx]
            )
            print(
                f"[POSE] Current frame {current_frame_idx} pose:\n{current_pose.data.cpu().numpy()}"
            )

        # 打印前5个点和最后5个点的坐标，共10个点
        if len(points_3d) > 0:
            print(f"[POINTS] First 5 and last 5 points coordinates (x,y,z):")

            # 打印前5个点
            for i in range(min(5, len(points_3d))):
                x, y, z = points_3d[i]
                print(f"  Point {i}: ({x:.3f}, {y:.3f}, {z:.3f})")

            # 如果点数超过5个，打印最后5个点
            if len(points_3d) > 5:
                print("  ...")
                start_idx = max(5, len(points_3d) - 5)
                for i in range(start_idx, len(points_3d)):
                    x, y, z = points_3d[i]
                    print(f"  Point {i}: ({x:.3f}, {y:.3f}, {z:.3f})")

        # 如果有多于一帧，获取前一帧的5个点
        if (
            self.n > 1
            and hasattr(self, "pg")
            and hasattr(self.pg, "patches_")
            and self.pg.patches_ is not None
        ):
            prev_frame_idx = current_frame_idx - 1
            if prev_frame_idx >= 0 and prev_frame_idx < len(self.pg.patches_):
                try:
                    # 获取前一帧的patches范围
                    start_idx = prev_frame_idx * self.M
                    end_idx = min(start_idx + self.M, self.m)

                    if start_idx < self.m:
                        # 获取前一帧的3D点
                        prev_points = pops.point_cloud(
                            SE3(self.poses),
                            self.patches[:, start_idx:end_idx],
                            self.intrinsics,
                            self.ix[start_idx:end_idx],
                        )

                        if prev_points.dim() == 4 and prev_points.shape[1] > 0:
                            prev_points_3d = (
                                (
                                    prev_points[..., 1, 1, :3]
                                    / prev_points[..., 1, 1, 3:]
                                )
                                .reshape(-1, 3)
                                .cpu()
                                .numpy()
                            )
                            print(
                                f"[POINTS] Previous frame {prev_frame_idx} first 5 and last 5 points coordinates:"
                            )

                            # 打印前5个点
                            for i in range(min(5, len(prev_points_3d))):
                                x, y, z = prev_points_3d[i]
                                print(f"  Point {i}: ({x:.3f}, {y:.3f}, {z:.3f})")

                            # 如果点数超过5个，打印最后5个点
                            if len(prev_points_3d) > 5:
                                print("  ...")
                                start_idx = max(5, len(prev_points_3d) - 5)
                                for i in range(start_idx, len(prev_points_3d)):
                                    x, y, z = prev_points_3d[i]
                                    print(f"  Point {i}: ({x:.3f}, {y:.3f}, {z:.3f})")
                except Exception as e:
                    print(f"[POINTS] Error getting previous frame points: {e}")

        # endregion

        frame_bgr = np.hstack([image_bgr_left, image_bgr_right])
        # Save combined visualization
        debug_dir = "/home/jerett/Project/DPVO/Debug/Image"
        output_path = f"{debug_dir}/frame_{tstamp:06d}_visualization.png"
        cv2.imwrite(output_path, frame_bgr)
