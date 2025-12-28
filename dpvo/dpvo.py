import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import sys
import os
import json
from datetime import datetime
import cv2  # For visualization
from kornia.geometry import left_to_right_epipolar_distance


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
        # print(f"[Forward],r: {r},t0: {t0},t1: {t1}")
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        # print(f"[Back],r: {r},t0: {t0},t1: {t1}")
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp, image, intrinsics):
        """track new frame"""

        # Print frame separator with timestamp
        # print("\n" + "=" * 80)
        print(f"==== [FRAME] Processing Frame #{tstamp} ====")
        # print("=" * 80)

        # Store current timestamp for use in update()
        self.tstamp = tstamp

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
        self.original_image = image.clone()  # Store before normalization

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

            for _ in range(12):
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
        # self.print_frame_statistics(tstamp, original_image)

        # # Debug主目录
        # debug_main_dir = "/home/jerett/Project/DPVO/Debug/Movie4082"
        # # 1.保存原始图像
        # self.save_raw_image(tstamp, original_image, debug_main_dir)
        # # 2.将图像保存下来(可视化)
        # self.visualize_feature_points(tstamp, original_image, debug_main_dir)
        # # 3.保存相机参数和分辨率
        # self.save_camera_params(tstamp, debug_main_dir)
        # # 4.保存关键帧pose
        # self.save_keyframe_poses(tstamp, debug_main_dir)
        # # 5.保存当前帧点
        # self.save_current_frame_points(tstamp, debug_main_dir)
        # # 6.保存全部点
        # self.save_all_points(tstamp, debug_main_dir)

    def print_frame_statistics(self, tstamp, original_image):
        """
        Print comprehensive frame statistics including poses, points, and depth information.

        Args:
            tstamp: Timestamp of the current frame
        """
        print(
            f"\n[STATS] Timestamp: {tstamp},Distance: {self.distance:.3f},Keyframes/Counter: {self.n}/{self.counter}"
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

    def visualize_feature_points(self, tstamp, original_image, debug_main_dir):
        """Single frame visualization with 3 parts: 1) original image, 2) projected patches with depth colors, 3) depth colorbar."""
        print(
            f"\n[Visual] Timestamp: {tstamp},Distance: {self.distance:.3f},Keyframes/Counter: {self.n}/{self.counter}"
        )

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

        # 可视化文字
        draw_text_with_idx(image_bgr_left, f"tstamp: {tstamp}", 1, (255, 255, 255))
        draw_text_with_idx(image_bgr_left, f"kframe: {self.n}/{self.counter}", 2, (255, 255, 255))
        draw_text_with_idx(image_bgr_left, f"dist: {self.distance:.2f}", 3, (255, 255, 255))

        # 可视化关键点
        current_patches = self.pg.patches_[self.n - 1]
        print(f"[VPatch]: {current_patches.shape}")
        current_pixels = (current_patches[:,:2,1,1] * self.RES).cpu().numpy().astype(int)
        current_inv_depths = current_patches[:, 2,1,1].cpu().numpy()
        current_depths = 1.0 / current_inv_depths
        valid_mask = (current_depths > 0.2) & (current_depths < 50.0)

        draw_text_with_idx(image_bgr_left, f"patch: {len(valid_mask)}/{len(current_patches)}", 4, (255, 255, 255))


        valid_indices = np.where(valid_mask)[0]
        valid_pixels = current_pixels[valid_indices]
        valid_depths = current_depths[valid_indices]
        if len(valid_indices) > 0:
            # 深度排序
            sort_indices = np.argsort(valid_depths)
            sorted_depths = valid_depths[sort_indices]
            sorted_pixels = valid_pixels[sort_indices]

            # 深度采样
            num_samples = min(20, len(sorted_depths))
            if num_samples > 2:
                sample_indices = {0, len(sorted_depths) - 1}  # 最小值和最大值索引
                remaining_samples = num_samples - 2
                if remaining_samples > 0 and len(sorted_depths) > 2:
                    middle_indices = np.linspace(1, len(sorted_depths) - 2, remaining_samples, dtype=int)
                    sample_indices.update(middle_indices)
                sample_indices = sorted(list(sample_indices))
            elif num_samples == 2:
                sample_indices = [0, len(sorted_depths) - 1]  # 最小值和最大值
            else:
                sample_indices = [0]

            depth_min = sorted_depths.min()
            depth_max = sorted_depths.max()
            depth_range = depth_max - depth_min

            # 有效点可视化
            if depth_range > 0:
                for pixel, depth_val in zip(sorted_pixels, sorted_depths):
                    u, v = pixel[0], pixel[1]
                    if (0 <= u < image_bgr_left.shape[1] and 0 <= v < image_bgr_left.shape[0]):
                        depth_norm = (depth_val - depth_min) / depth_range
                        cv2.circle(image_bgr_left, (u, v), 5, get_jet_color(depth_norm), -1)

            # 加上深度数值
            for i, sample_idx in enumerate(sample_indices):
                pixel = sorted_pixels[sample_idx]
                depth_val = sorted_depths[sample_idx]
                u, v = pixel[0], pixel[1]

                if (0 <= u < image_bgr_left.shape[1] and 0 <= v < image_bgr_left.shape[0]):
                    depth_text = f"{depth_val:.2f}"

                    text_size = cv2.getTextSize(depth_text, font, 0.5, 1)[0]

                    # 确保文字不超出图像边界
                    text_x = u + 5
                    text_y = v - 5

                    # 绘制文字背景
                    cv2.rectangle(
                        image_bgr_left,
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0),
                        -1
                    )
                    text_color = (0, 255, 255)  # 黄色
                    cv2.putText(
                        image_bgr_left,
                        depth_text,
                        (text_x, text_y),
                        font,
                        0.5,
                        text_color,
                        2
                    )
        # endregion

        # region right
        ## 2.可视化右侧图像 - 投影最后8帧的3D点
        # 获取当前帧pose和内参
        curr_pose = self.pg.poses_[self.n - 1]
        intr = self.intrinsics[0, self.n - 1]

        # 使用已存储的点云数据
        points_3d = self.pg.points_[:self.m]
        proj_size = min(8 * self.M, len(points_3d))
        proj_points_3d = points_3d[-proj_size:]  # 取最后8帧的点

        print(f"[VPose],curr_pose: {curr_pose}")
        print(f"[VIntr],intr: {intr}")
        print(f"[VPoint],total_points: {points_3d.shape}, proj_points: {proj_points_3d.shape}")

        draw_text_with_idx(image_bgr_right, f"points: {len(points_3d)}", 1 , (255, 255, 255))
        # 投影到当前帧 - 使用矩阵操作优化
        if len(proj_points_3d) > 0:
            curr_pose_se3 = SE3(curr_pose)
            curr_pose_matrix = curr_pose_se3.matrix().squeeze().cpu().numpy()
            fx, fy, cx, cy = intr.cpu().numpy()

            # 批量转换为齐次坐标 (N, 4)
            points_3d_np = proj_points_3d.cpu().numpy()
            points_homo = np.hstack([points_3d_np, np.ones((len(points_3d_np), 1))])

            # 批量转换到当前帧坐标系 (N, 4) = (4, 4) @ (4, N).T
            points_cam = (curr_pose_matrix @ points_homo.T).T

            # 检查点是否在相机前方 (深度 > 0)
            valid_mask = points_cam[:, 2] > 0

            if valid_mask.any():
                # 只对有效点进行投影
                valid_points_cam = points_cam[valid_mask]

                # 批量投影到图像平面
                depths = valid_points_cam[:, 2]
                u_proj = ((fx * valid_points_cam[:, 0] / depths) + cx) * self.RES
                v_proj = ((fy * valid_points_cam[:, 1] / depths) + cy) * self.RES

                # 整数坐标和深度过滤
                u_int = u_proj.astype(int)
                v_int = v_proj.astype(int)

                # 图像边界和深度范围过滤
                image_h, image_w = image_bgr_right.shape[:2]
                boundary_mask = (
                    (u_int >= 0) & (u_int < image_w) &
                    (v_int >= 0) & (v_int < image_h) &
                    (depths > 0.2) & (depths < 50.0)
                )

                if boundary_mask.any():
                    # 最终有效的投影点
                    projected_pixels = np.column_stack([
                        u_int[boundary_mask],
                        v_int[boundary_mask]
                    ])
                    projected_depths = depths[boundary_mask]

                    # 显示投影点数量
                    draw_text_with_idx(image_bgr_right, f"proj_points: {len(projected_pixels)}", 2, (255, 255, 255))

                    # 按深度排序
                    sort_indices = np.argsort(projected_depths)
                    sorted_depths = projected_depths[sort_indices]
                    sorted_pixels = projected_pixels[sort_indices]

                    # 深度采样，确保包含最小值和最大值
                    num_samples = min(20, len(sorted_depths))
                    if num_samples > 2:
                        sample_indices = {0, len(sorted_depths) - 1}  # 最小值和最大值索引
                        remaining_samples = num_samples - 2
                        if remaining_samples > 0 and len(sorted_depths) > 2:
                            middle_indices = np.linspace(1, len(sorted_depths) - 2, remaining_samples, dtype=int)
                            sample_indices.update(middle_indices)
                        sample_indices = sorted(list(sample_indices))
                elif num_samples == 2:
                    sample_indices = [0, len(sorted_depths) - 1]
                else:
                    sample_indices = [0]

                depth_min = sorted_depths.min()
                depth_max = sorted_depths.max()
                depth_range = depth_max - depth_min

                # 可视化所有有效投影点（按深度着色）
                if depth_range > 0:
                    for pixel, depth_val in zip(sorted_pixels, sorted_depths):
                        u, v = pixel[0], pixel[1]
                        depth_norm = (depth_val - depth_min) / depth_range
                        cv2.circle(image_bgr_right, (u, v), 5, get_jet_color(depth_norm), -1)

                # 为采样的20个点添加深度数值标签
                for i, sample_idx in enumerate(sample_indices):
                    pixel = sorted_pixels[sample_idx]
                    depth_val = sorted_depths[sample_idx]
                    u, v = pixel[0], pixel[1]

                    # 添加深度数值标签
                    depth_text = f"{depth_val:.2f}"
                    text_size = cv2.getTextSize(depth_text, font, 0.5, 1)[0]

                    # 确保文字不超出图像边界
                    text_x = u + 5
                    text_y = v - 5
                    if text_x + text_size[0] > image_bgr_right.shape[1]:
                        text_x = u - text_size[0] - 5
                    if text_y < text_size[1]:
                        text_y = v + text_size[1] + 5

                    # 绘制文字背景
                    cv2.rectangle(
                        image_bgr_right,
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0),
                        -1
                    )

                    text_color = (0, 255, 255)
                    cv2.putText(
                        image_bgr_right,
                        depth_text,
                        (text_x, text_y),
                        font,
                        0.5,
                        text_color,
                        2
                    )
            else:
                draw_text_with_idx(image_bgr_right, "proj_points: 0", 2, (255, 255, 255))
        else:
            draw_text_with_idx(image_bgr_right, "proj_points: 0", 2, (255, 255, 255))

        # endregion

        frame_bgr = np.hstack([image_bgr_left, image_bgr_right])
        # Save combined visualization
        debug_dir = f"{debug_main_dir}/image"
        os.makedirs(debug_dir, exist_ok=True)
        output_path = f"{debug_dir}/{tstamp:06d}.png"
        cv2.imwrite(output_path, frame_bgr)

    def save_raw_image(self, tstamp, original_image, debug_main_dir):
        """保存原始图像"""
        debug_dir = f"{debug_main_dir}/raw_image"
        os.makedirs(debug_dir, exist_ok=True)
        output_path = f"{debug_dir}/{tstamp:06d}.png"

        # 转换图像格式从 CHW (PyTorch) 到 HWC (OpenCV)
        image_bgr = original_image.cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(output_path, image_bgr)

    def save_camera_params(self, tstamp, debug_main_dir):
        """保存相机参数和分辨率信息"""
        debug_dir = f"{debug_main_dir}/param"
        os.makedirs(debug_dir, exist_ok=True)
        output_path = f"{debug_dir}/{tstamp:06d}.json"

        # 获取当前帧的内参 [fx, fy, cx, cy]
        current_intrinsics = self.pg.intrinsics_[self.n - 1].cpu().numpy() if self.n > 0 else np.zeros(4)
        fx, fy, cx, cy = current_intrinsics

        # 创建参数字典
        params_data = {
            'timestamp': tstamp,
            'frame_id': self.n - 1 if self.n > 0 else 0,
            'resolution': self.RES,
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'intrinsics': [float(fx), float(fy), float(cx), float(cy)],  # [fx, fy, cx, cy]
            'image_height': int(self.ht),
            'image_width': int(self.wd)
        }

        # 保存为JSON格式
        with open(output_path, 'w') as f:
            json.dump(params_data, f, indent=2)

    def save_keyframe_poses(self, tstamp, debug_main_dir):
        """保存关键帧pose信息"""
        # 保存所有关键帧pose到kframe_pose文件夹
        kframe_pose_dir = f"{debug_main_dir}/kframe_pose"
        os.makedirs(kframe_pose_dir, exist_ok=True)

        # 保存关键帧时间戳到kframe_time文件夹
        kframe_time_dir = f"{debug_main_dir}/kframe_time"
        os.makedirs(kframe_time_dir, exist_ok=True)

        # 创建pose矩阵 (n, 7) 格式: [tx, ty, tz, qx, qy, qz, qw]
        poses_matrix = np.zeros((self.n, 7))
        timestamps_array = np.zeros((self.n,))

        # 收集所有关键帧的pose信息
        for i in range(self.n):
            timestamps_array[i] = self.pg.tstamps_[i].item()
            pose = self.pg.poses_[i].cpu().numpy()
            poses_matrix[i] = [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]]

        # 保存pose矩阵到kframe_pose文件夹
        pose_output_path = f"{kframe_pose_dir}/{tstamp:06d}.npy"
        np.save(pose_output_path, poses_matrix)

        # 保存时间戳数组到kframe_time文件夹
        time_output_path = f"{kframe_time_dir}/{tstamp:06d}.npy"
        np.save(time_output_path, timestamps_array)

    def save_current_frame_points(self, tstamp, debug_main_dir):
        """保存当前帧的点云信息"""
        debug_dir = f"{debug_main_dir}/current_points"
        os.makedirs(debug_dir, exist_ok=True)
        output_path = f"{debug_dir}/{tstamp:06d}.npy"

        if self.m > 0 and self.n > 0:
            # 获取当前帧的patches
            current_patches = self.pg.patches_[self.n - 1]  # 当前帧patches

            # 获取像素坐标 (需要乘以RES得到原始图像坐标)
            pixel_coords = current_patches[:, :2, 1, 1] * self.RES  # [M, 2] - u, v
            inv_depths = current_patches[:, 2, 1, 1]  # [M] - 逆深度

            # 获取当前帧的内参和pose
            current_intrinsics = self.pg.intrinsics_[self.n - 1]  # [4] - fx, fy, cx, cy
            current_pose = self.pg.poses_[self.n - 1]  # [7] - pose

            # 提取内参
            fx, fy, cx, cy = current_intrinsics.cpu().numpy()

            # 转换像素坐标到相机坐标系下的3D点 (z=1/inv_depth)
            u = pixel_coords[:, 0].cpu().numpy()  # 像素u坐标
            v = pixel_coords[:, 1].cpu().numpy()  # 像素v坐标
            inv_d = inv_depths.cpu().numpy()      # 逆深度
            depth = 1.0 / inv_d                  # 深度

            # 相机坐标系下的3D点
            x_cam = (u - cx) * depth / fx
            y_cam = (v - cy) * depth / fy
            z_cam = depth

            # 当前帧点云直接保存在相机坐标系下，不进行世界坐标变换
            points_3d = np.column_stack([x_cam, y_cam, z_cam])

            # 输出调试信息
            print(f"[DEBUG] Camera coordinate system points (first 3): {points_3d[:3] if len(points_3d) > 0 else 'None'}")

            # 过滤有效的深度范围
            valid_mask = (depth > 0.2) & (depth < 50.0)
            points_3d = points_3d[valid_mask]

            print(f"[DEBUG] Valid points count in camera coords: {len(points_3d)}")

            # 保存为numpy矩阵
            np.save(output_path, points_3d)
        else:
            # 如果没有点或未初始化，保存空矩阵
            empty_matrix = np.zeros((0, 3))
            np.save(output_path, empty_matrix)

    def save_all_points(self, tstamp, debug_main_dir):
        """保存所有点云信息"""
        debug_dir = f"{debug_main_dir}/all_points"
        os.makedirs(debug_dir, exist_ok=True)
        output_path = f"{debug_dir}/{tstamp:06d}.npy"

        if self.m > 0:
            # 获取所有3D点坐标 (n, 3)
            points_3d = self.pg.points_[:self.m].cpu().numpy()

            # 保存为numpy矩阵
            np.save(output_path, points_3d)
        else:
            # 如果没有点，保存空矩阵
            empty_matrix = np.zeros((0, 3))
            np.save(output_path, empty_matrix)
