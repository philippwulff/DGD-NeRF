from matplotlib import use
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchsearchsorted import searchsorted
# supported in pytorch now: https://github.com/aliutkus/torchsearchsorted/issues/24
from torch import searchsorted
import math
import imageio
import os

from utils.metrics import MSE, PSNR, SSIM, LPIPS, RMSE

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to8d = lambda x : (255*(x/np.max(x))).astype(np.uint8)


def depth2mse(depth, target_depth):
    """Calculate MSE loss over valid (non-zero) depth pixels."""
    # TODO possibly scale mse loss by valid_pixels/all_pixels
    inds_nonzero = target_depth > 0.1
    return torch.mean((depth[inds_nonzero] - target_depth[inds_nonzero]) ** 2)


def depth2gnll(depth, target_depth, depth_std, target_depth_std=0.015): #TODO J: find out target_depth_std and in which unit depth is given
    """
    Calculate Gaussian Negative Log Likelihood Loss over valid depth rays.
    Calculate only if 
    predicted depth - ground truth depth > sensor depth standard deviation
    predicted depth standard deviation > sensor depth standard deviation
    """
    inds_nonzero = target_depth > 0.1
    inds_depth_prediction = (depth - target_depth).abs() > target_depth_std
    inds_depth_std_prediction = depth_std > target_depth_std
    inds_valid = torch.logical_or(inds_depth_prediction, inds_depth_std_prediction)
    inds_valid = torch.logical_and(inds_nonzero, inds_valid)

    depth_valid = depth[inds_valid] 
    target_depth_valid = target_depth[inds_valid] 
    depth_std_valid = depth_std[inds_valid]
    depth_var_valid = depth_std_valid**2

    f = nn.GaussianNLLLoss(full=True,eps=1e-5)

    return f(depth_valid, target_depth_valid, depth_var_valid)


# Positional encoding (section 5.1)
class Embedder:
    """Positional encoding."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        """Create positional encoding functions."""
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)          # encoded vector includes input # TODO P: im paper steht, dass das nicht den input beinhaltet
            out_dim += d
        
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']          # e.g., when N_freqs=10, then max_freq=9
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))   
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    """Returns function to do positional encoding.
    Args:
        multires (int): log2 of the max frequency.
        input_dims (int): input dimensionality.
        i (int, optional): set 0 for default positional encoding, -1 for none. Defaults to 0.
    Returns:
        embed: func. Function that returns embedding of vector.
        embedder_obj.out_dim: int. The dimension of the embedding.
    """
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x) 
    return embed, embedder_obj.out_dim


# Model
class DirectTemporalNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True, use_rigidity_network=False,
                use_latent_codes_as_time=False, ray_bending_latent_size=None):
        super(DirectTemporalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time if not use_latent_codes_as_time else ray_bending_latent_size
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical                    # if the scene at t=0 is the canonical configuration
        self.use_rigidity_network = use_rigidity_network
        self.use_latent_codes_as_time = use_latent_codes_as_time

        self._occ = NeRFOriginal(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views,
                                 input_ch_time=input_ch_time, output_ch=output_ch, skips=skips,
                                 use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
        self._time, self._time_out = self.create_time_net()
        if self.use_rigidity_network:
            self._rigidity = self.create_rigidity_net()
    
    def create_rigidity_net(self):
        """The Rigidity Network"""
        ### code from https://github.com/facebookresearch/nonrigid_nerf/blob/main/run_nerf_helpers.py

        self.rigidity_activation_function = F.relu  # F.relu, torch.sin
        self.rigidity_hidden_dimensions = 32  # 32
        self.rigidity_network_depth = 3  # 3 # at least 2: input -> hidden -> output
        self.rigidity_skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = True
        self.rigidity_tanh = nn.Tanh()

        self.rigidity_network = nn.ModuleList(
            [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)]
            + [
                nn.Linear(
                    self.input_ch + self.rigidity_hidden_dimensions,
                    self.rigidity_hidden_dimensions,
                )
                if i + 1 in self.rigidity_skips
                else nn.Linear(
                    self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions
                )
                for i in range(self.rigidity_network_depth - 2)
            ]
            + [
                nn.Linear(
                    self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias
                )
            ]
        )

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.rigidity_network[:-1]):
                if self.rigidity_activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.rigidity_activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights
            self.rigidity_network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.rigidity_network[-1].bias.data *= 0.0
            
        return self.rigidity_network

    def query_rigidity(self, input_pts):
        """Predicts Rigidity from given encoded location"""
        h = input_pts
        for i, layer in enumerate(self.rigidity_network):
            h = layer(h)

            # SIREN
            if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                h *= 30.0

            if i != len(self.rigidity_network) - 1:
                h = self.rigidity_activation_function(h)

            if i in self.rigidity_skips:
                h = torch.cat([input_pts, h], -1)
        rigidity_mask = (
            self.rigidity_tanh(h) + 1
        ) / 2  # close to 1 for nonrigid, close to 0 for rigid

        return rigidity_mask

    def create_time_net(self):
        """The deformation network."""
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]    # input is encoded position + time/latent code
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:                                 # skip connection from input to ith layer
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]              
        return nn.ModuleList(layers), nn.Linear(self.W, 3)      # final layer outputs (delta x,delta y,delta z)

    def query_time(self, new_pts, t, net, net_final):
        """Predict deformation from given encoded location + time."""
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts):
        """Predict canonical sample.
        Args:
            x (Tensor): shape (-1, encoding_size). Embedded position (+ viewing direction).
            ts (list): Time stamps of the rays in the batch.
                if self.use_latent_codes_as_time:   list of two identical Tensors with shape (-1, ray_bending_latent_size)
                else:                               list of two identical Tensors with shape (-1, encoding_size).
        Returns:
            out: RGB and density
            dx: deformations
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = ts[0] 
        if not self.use_latent_codes_as_time:
            assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
            cur_time = t[0, 0]

        if not self.use_latent_codes_as_time and cur_time == 0. and self.zero_canonical:
            dx = torch.zeros_like(input_pts[:, :3])                             # no deformation in canonical configuration
        else:
            dx = self.query_time(input_pts, t, self._time, self._time_out)      # deformation of given point at time t
            if self.use_rigidity_network:
                rigidity_mask = self.query_rigidity(input_pts)
                dx = rigidity_mask * dx
            input_pts_orig = input_pts[:, :3]
            input_pts = self.embed_fn(input_pts_orig + dx)
        out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)      # predict RGB + density
        return out, dx


class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("[Config] NeRF type selected: %s" % type)

        if type == "original":
            if kwargs.use_latent_codes_as_time or kwargs.use_rigity_network:
                raise NotImplementedError("Naive NeRF cannot be used with latent deformation codes or rigidity network.")
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model


class NeRFOriginal(nn.Module):
    """See architecture in the original paper in Fig. 7. This is the canonical network."""
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        super(NeRFOriginal, self).__init__()
        self.D = D                              # depth
        self.W = W                              # width
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs        # use full 5D input (position + direction). Else, ignore viewing direction (3D input).

        ### First MLP with encoded position input (x,y,z) -> output is latent vector + density
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] +
        #     [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips:                 
                in_channels += input_ch     # add skip connection from input to ith layer

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])   # 2nd MLP: viewing dir + feature vector input

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)               # latent vector -> 2nd MLP input
            self.alpha_linear = nn.Linear(W, 1)                 # latent vector -> density 
            self.rgb_linear = nn.Linear(W//2, output_color_ch)  # second MLP output -> RGB
        else:
            self.output_linear = nn.Linear(W, output_ch)        # latent vector -> RGB + density

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):                # pass position through 1st MLP
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:                                   # 2nd MLP
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:                                                   # 2nd MLP is a single layer
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def hsv_to_rgb(h, s, v):
    '''
    h,s,v in range [0,1]
    '''
    hi = torch.floor(h * 6)
    f = h * 6. - hi
    p = v * (1. - s)
    q = v * (1. - f * s)
    t = v * (1. - (1. - f) * s)

    rgb = torch.cat([hi, hi, hi], -1) % 6
    rgb[rgb == 0] = torch.cat((v, t, p), -1)[rgb == 0]
    rgb[rgb == 1] = torch.cat((q, v, p), -1)[rgb == 1]
    rgb[rgb == 2] = torch.cat((p, v, t), -1)[rgb == 2]
    rgb[rgb == 3] = torch.cat((p, q, v), -1)[rgb == 3]
    rgb[rgb == 4] = torch.cat((t, p, v), -1)[rgb == 4]
    rgb[rgb == 5] = torch.cat((v, p, q), -1)[rgb == 5]
    return rgb


# Ray helpers
def get_rays(H, W, focal_x, focal_y, c2w):
    """Returns ray directions and origins (per ray) in the world frame.

    Args:
        H (int): Image height in pixels.
        W (int): Image width in pixels.
        focal_x (float): Focal length of the virtual camera.
        focal_y (float): Focal length of the virtual camera.
        c2w (torch.Tensor): 3x4 Tensor. Horizontal stack of the camera-to-world rotation matrix and translation vector.

    Returns:
        rays_o: Tensor
        rays_d: Tensor
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing="ij")
    i = i.t()   # transposes 2D tensor
    j = j.t()
    # The ray directions in the camera coordinate system. 
    # They reach respective pixel on the image (scaled by 1/focal_length) after travelling 1 unit.
    dirs = torch.stack([(i-W*.5)/focal_x, -(j-H*.5)/focal_y, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal_x, focal_y, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal_x, -(j-H*.5)/focal_y, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    #inds = searchsorted(cdf, u, side='right')
    inds = searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


# Volume Rendering helpers
def compute_weights(raw, z_vals, rays_d, device=None, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights


def raw2depth(raw, z_vals, rays_d, device=None):
    """Computes depth and standard deviations from network output."""
    weights = compute_weights(raw, z_vals, rays_d, device)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def compute_samples_around_depth(raw, z_vals, rays_d, N_samples, perturb, lower_bound, near, far, device=None):
    """Computes samples within 3 sigma from the predicted depth."""
    sampling_depth, sampling_std = raw2depth(raw, z_vals, rays_d, device)
    sampling_std = sampling_std.clamp(min=lower_bound)
    # IMPORTANT: Maybe hardcode the std here
    sampling_std = torch.full_like(sampling_std, 0.03)
    depth_min = sampling_depth - 3. * sampling_std
    depth_max = sampling_depth + 3. * sampling_std
    return sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far, device)


def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far, device=None):
    """Samples N values from within 3 sigma. Clipped at near and far boundaries."""
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)


def comp_quadratic_samples(near, far, num_samples):
    """normal parabola between 0.1 and 1, shifted and scaled to have y range between near and far"""
    start = 0.1
    x = torch.linspace(0, 1, num_samples)
    c = near
    a = (far - near)/(1. + 2. * start)
    b = 2. * start * a
    return a * x.pow(2) + b * x + c


def comp_depth_sampling(depth, stds):
    """Computes ranges to sample depth locations from. 
    Min and max values are also computed for invalid depth values.

    Args:
        depth (torch.Tensor): [N_rand, 1]. Depth values.
        stds (torch.Tensor): [N_rand, 1]. Standard deviations.

    Returns:
        torch.Tensor: [N_rand, 3]. Sampling range (depth, depth_min, depth_max) for each ray.
    """
    depth_min = depth - 3. * stds   
    depth_max = depth + 3. * stds
    return torch.stack((depth, depth_min, depth_max), 1).squeeze()

def estim_error(estim, gt, estim_depth, gt_depth):
    errors = dict()
    metric = MSE()
    errors["mse"] = metric(estim, gt).item()
    metric = PSNR()
    errors["psnr"] = metric(estim, gt).item()
    metric = SSIM()
    errors["ssim"] = metric(estim, gt).item()
    metric = LPIPS()
    errors["lpips"] = metric(estim, gt).item()
    metric = RMSE()
    errors["depth_rmse"] = metric(estim_depth, gt_depth).item()
    
    return errors

def save_error(errors, save_dir):
    save_path = os.path.join(save_dir, "metrics.txt")
    f = open(save_path,"w")
    f.write( str(errors) )
    f.close()

def compute_metrics(files_dir, estim,gt,estim_depth,gt_depth):
    estim = np.transpose(estim, (0, 3, 1, 2))
    gt = np.transpose(gt, (0, 3, 1, 2))
    estim = torch.Tensor(estim).cuda()
    gt = torch.Tensor(gt).cuda()
    estim_depth = torch.Tensor(estim_depth).cuda()
    gt_depth = torch.Tensor(gt_depth).cuda()

    errors = estim_error(estim, gt, estim_depth, gt_depth)
    save_error(errors, files_dir)
