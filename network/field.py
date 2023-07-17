import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import nvdiffrast.torch as dr
import mcubes

from utils.base_utils import az_el_to_points, sample_sphere
from utils.raw_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 sdf_activation='none',
                 layer_activation='softplus'):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif layer_activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[..., :1].detach(), gradients.detach()


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act == 'exp':
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act == 'linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act == 'square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

    def density(self, input_pts):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module


def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:, :, :3], poses[:, :, 3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:, :, None] + t)[..., 0]  # pn,3
    dirs_ = (R @ dirs[:, :, None])[..., 0]  # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.


def IPE(mean, var, min_deg, max_deg):
    scales = 2 ** torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
                        torch.concat([scaled_var] * 2, dim=-1))


def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask) > 0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points


def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts * dirs, dim=-1, keepdim=True)  # rn,1
    xtx = torch.sum(pts ** 2, dim=-1, keepdim=True)  # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist < 0) == 0
    dist = -dtx + torch.sqrt(dist + 1e-6)  # rn,1
    return dist


# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)  # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask] = -1.0
    return weights, mid_sdf


def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999  # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1 - 1])
    hit_weights = torch.zeros([pn, sn1 - 1])
    hit_sdf = -torch.ones([pn, sn1 - 1])
    if torch.sum(inside_mask) > 0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs)  # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0)  # sn0
            z_vals = max_dist * z_vals.unsqueeze(0)  # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs)  # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True)  # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs)  # pn,sn1-1
            z_vals_mid = (z_vals_new[:, 1:] + z_vals_new[:, :-1]) * 0.5

        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf


class AppShadingNetwork(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
        else:
            direct_light = self.outer_light(ref_roughness)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
        occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness = torch.ones([normals.shape[0], 1])
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective, roughness,
                                                                                             human_poses, step)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        specular_color = specular_ref * specular_light

        # integrated together
        color = diffuse_color + specular_color

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)
        color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
        }

        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
            }
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo


class MaterialFeatsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, input_dim = get_embedder(8, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
        )
        self.module1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        input = x
        x = self.module0(x)
        return self.module1(torch.cat([x, input], -1))


def saturate_dot(v0, v1):
    return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)


class MCShadingNetwork(nn.Module):
    default_cfg = {
        'diffuse_sample_num': 512,
        'specular_sample_num': 256,
        'human_lights': True,
        'light_exp_max': 5.0,
        'inner_light_exp_max': 5.0,
        'outer_light_version': 'direction',
        'geometry_type': 'schlick',

        'reg_change': True,
        'change_eps': 0.05,
        'change_type': 'gaussian',
        'reg_lambda1': 0.005,
        'reg_min_max': True,

        'random_azimuth': True,
        'is_real': False,
    }

    def __init__(self, cfg, ray_trace_fun):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()

        # material part
        self.feats_network = MaterialFeatsNetwork()
        self.metallic_predictor = make_predictor(256 + 3, 1)
        self.roughness_predictor = make_predictor(256 + 3, 1)
        self.albedo_predictor = make_predictor(256 + 3, 3)

        # light part
        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(8, 3)
        if self.cfg['outer_light_version'] == 'direction':
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        else:
            raise NotImplementedError
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
        if self.cfg['human_lights']:
            self.human_light = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light[-2].bias, np.log(0.02))
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=self.cfg['inner_light_exp_max'])
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))

        # predefined diffuse sample directions
        az, el = sample_sphere(self.cfg['diffuse_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(
            self.diffuse_direction_samples.astype(np.float32)).cuda()  # [dn0,2]

        az, el = sample_sphere(self.cfg['specular_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.specular_direction_samples = np.stack([az, el], -1)
        self.specular_direction_samples = torch.from_numpy(
            self.specular_direction_samples.astype(np.float32)).cuda()  # [dn1,2]

        az, el = sample_sphere(8192, 0)
        light_pts = az_el_to_points(az, el)
        self.register_buffer('light_pts', torch.from_numpy(light_pts.astype(np.float32)))
        self.ray_trace_fun = ray_trace_fun

    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1)  # pn,1
        otho0 = torch.cat([y, -x, torch.zeros_like(x)], -1)
        otho1 = torch.cat([-z, torch.zeros_like(x), x], -1)
        mask0 = torch.norm(otho0, dim=-1) > torch.norm(otho1, dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho

    def sample_diffuse_directions(self, normals, is_train):
        # normals [pn,3]
        z = normals  # pn,3
        x = self.get_orthogonal_directions(normals)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        # y = torch.cross(z, x, dim=-1) # pn,3

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples, 1, dim=1)  # sn,1
        el, az = el.unsqueeze(0), az.unsqueeze(0)
        az = az * np.pi * 2
        el_sqrt = torch.sqrt(el + 1e-7)
        if is_train and self.cfg['random_azimuth']:
            az = (az + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)  # pn,sn,3
        return directions

    def sample_specular_directions(self, reflections, roughness, is_train):
        # roughness [pn,1]
        z = reflections  # pn,3
        x = self.get_orthogonal_directions(reflections)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        a = roughness  # we assume the predicted roughness is already squared

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = np.pi * 2 * az  # sn,1
        a, el = a.unsqueeze(1), el.unsqueeze(0)  # [pn,1,1] [1,sn,1]
        cos_theta = torch.sqrt((1.0 - el + 1e-6) / (1.0 + (a ** 2 - 1.0) * el + 1e-6) + 1e-6)  # pn,sn,1
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + 1e-6)  # pn,sn,1

        phi = phi.unsqueeze(0)  # 1,sn,1
        if is_train and self.cfg['random_azimuth']:
            phi = (phi + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_x = torch.cos(phi) * sin_theta  # pn,sn,1
        coeff_y = torch.sin(phi) * sin_theta  # pn,sn,1
        coeff_z = cos_theta  # pn,sn,1

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)  # pn,sn,3
        return directions

    def get_inner_lights(self, points, view_dirs, normals):
        pos_enc = self.pos_enc(points)
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        dir_enc = self.sph_enc(reflections, 0)
        return self.inner_light(torch.cat([pos_enc, dir_enc], -1))

    def get_human_light(self, points, directions, human_poses):
        inter, dists, hits = get_camera_plane_intersection(points, directions, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits

        var = torch.zeros_like(mean)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_outer_lights(self, points, directions):
        if self.cfg['outer_light_version'] == 'direction':
            outer_enc = self.sph_enc(directions, 0)
            outer_lights = self.outer_light(outer_enc)
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            outer_dirs = directions
            outer_pts = points
            outer_enc = self.sph_enc(outer_dirs, 0)
            mask = torch.norm(outer_pts, dim=-1) > 0.999
            if torch.sum(mask) > 0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[mask] *= 0.999  # shrink this point a little bit
            dists = get_sphere_intersection(outer_pts, outer_dirs)
            sphere_pts = outer_pts + outer_dirs * dists
            sphere_pts = self.sph_enc(sphere_pts, 0)
            outer_lights = self.outer_light(torch.cat([outer_enc, sphere_pts], -1))
        else:
            raise NotImplementedError
        return outer_lights

    def get_lights(self, points, directions, human_poses):
        # trace
        shape = points.shape[:-1]  # pn,sn
        eps = 1e-5
        inters, normals, depth, hit_mask = self.ray_trace_fun(points.reshape(-1, 3) + directions.reshape(-1, 3) * eps,
                                                              directions.reshape(-1, 3))
        inters, normals, depth, hit_mask = inters.reshape(*shape, 3), normals.reshape(*shape, 3), depth.reshape(*shape,
                                                                                                                1), hit_mask.reshape(
            *shape)
        miss_mask = ~hit_mask

        # hit_mask
        lights = torch.zeros(*shape, 3)
        human_lights, human_weights = torch.zeros([1, 3]), torch.zeros([1, 1])
        if torch.sum(miss_mask) > 0:
            outer_lights = self.predict_outer_lights(points[miss_mask], directions[miss_mask])
            if self.cfg['human_lights']:
                human_lights, human_weights = self.get_human_light(points[miss_mask], directions[miss_mask],
                                                                   human_poses[miss_mask])
            else:
                human_lights, human_weights = torch.zeros_like(outer_lights), torch.zeros(outer_lights.shape[0], 1)
            lights[miss_mask] = outer_lights * (1 - human_weights) + human_lights * human_weights

        if torch.sum(hit_mask) > 0:
            lights[hit_mask] = self.get_inner_lights(inters[hit_mask], -directions[hit_mask], normals[hit_mask])

        near_mask = (depth > eps).float()
        lights = lights * near_mask  # very near surface does not bring lights
        return lights, human_lights * human_weights, inters, normals, hit_mask

    def fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0) ** 5.0

    def fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = (view_dirs + directions)  # [pn,sn0,3]
        H = F.normalize(H, dim=-1)
        HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0)  # [pn,sn0,1]
        fresnel = self.fresnel_schlick(F0, HoV)  # [pn,sn0,1]
        return fresnel, H, HoV

    def geometry_schlick_ggx(self, NoV, roughness):
        a = roughness  # a = roughness**2: we assume the predicted roughness is already squared

        k = a / 2
        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def geometry_schlick(self, NoV, NoL, roughness):
        ggx2 = self.geometry_schlick_ggx(NoV, roughness)
        ggx1 = self.geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
        def fun(alpha2, cos_theta):
            # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
            cos_theta2 = cos_theta ** 2
            tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
            return 0.5 * torch.sqrt(1 + alpha2 * tan_theta2) - 0.5

        alpha_sq = roughness ** 2
        return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))

    def predict_materials(self, pts):
        feats = self.feats_network(pts)
        metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
        roughness = self.roughness_predictor(torch.cat([feats, pts], -1))
        rmax, rmin = 1.0, 0.04 ** 2
        roughness = roughness * (rmax - rmin) + rmin
        albedo = self.albedo_predictor(torch.cat([feats, pts], -1))
        return metallic, roughness, albedo

    def distribution_ggx(self, NoH, roughness):
        a = roughness
        a2 = a ** 2
        NoH2 = NoH ** 2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom ** 2 + 1e-4)

    def geometry(self, NoV, NoL, roughness):
        if self.cfg['geometry_type'] == 'schlick':
            geometry = self.geometry_schlick(NoV, NoL, roughness)
        elif self.cfg['geometry_type'] == 'ggx_smith':
            geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
        else:
            raise NotImplementedError
        return geometry

    def shade_mixed(self, pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train):
        F0 = 0.04 * (1 - metallic) + metallic * albedo  # [pn,1]

        # sample diffuse directions
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)  # [pn,sn0,3]
        point_num, diffuse_num, _ = diffuse_directions.shape
        # sample specular directions
        specular_directions = self.sample_specular_directions(reflections, roughness, is_train)  # [pn,sn1,3]
        specular_num = specular_directions.shape[1]

        # diffuse sample prob
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1))
        diffuse_probability = NoL_d / np.pi * (diffuse_num / (specular_num + diffuse_num))

        # specualr sample prob
        H_s = (view_dirs.unsqueeze(1) + specular_directions)  # [pn,sn0,3] half vector
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(view_dirs.unsqueeze(1), H_s)
        specular_probability = self.distribution_ggx(NoH_s, roughness.unsqueeze(1)) * NoH_s / (4 * VoH_s + 1e-5) * (
                    specular_num / (specular_num + diffuse_num))  # D * NoH / (4 * VoH)

        # combine
        directions = torch.cat([diffuse_directions, specular_directions], 1)
        probability = torch.cat([diffuse_probability, specular_probability], 1)
        sn = diffuse_num + specular_num

        # specular
        fresnel, H, HoV = self.fresnel_schlick_directions(F0.unsqueeze(1), view_dirs.unsqueeze(1), directions)
        NoV = saturate_dot(normals, view_dirs).unsqueeze(1)  # pn,1,3
        NoL = saturate_dot(normals.unsqueeze(1), directions)  # pn,sn,3
        geometry = self.geometry(NoV, NoL, roughness.unsqueeze(1))
        NoH = saturate_dot(normals.unsqueeze(1), H)
        distribution = self.distribution_ggx(NoH, roughness.unsqueeze(1))
        human_poses = human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
        pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
        lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses)  # pn,sn,3
        specular_weights = distribution * geometry / (4 * NoV * probability + 1e-5)
        specular_lights = lights * specular_weights
        specular_colors = torch.mean(fresnel * specular_lights, 1)
        specular_weights = specular_weights * fresnel

        # diffuse only consider diffuse directions
        kd = (1 - metallic.unsqueeze(1))
        diffuse_lights = lights[:, :diffuse_num]
        diffuse_colors = albedo.unsqueeze(1) * kd[:, :diffuse_num] * diffuse_lights
        diffuse_colors = torch.mean(diffuse_colors, 1)

        colors = diffuse_colors + specular_colors
        colors = linear_to_srgb(colors)

        outputs = {}
        outputs['albedo'] = albedo
        outputs['roughness'] = roughness
        outputs['metallic'] = metallic
        outputs['human_lights'] = hl.reshape(-1, 3)
        outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1)
        outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(specular_lights, dim=1)), min=0, max=1)
        diffuse_colors = torch.clamp(linear_to_srgb(diffuse_colors), min=0, max=1)
        specular_colors = torch.clamp(linear_to_srgb(specular_colors), min=0, max=1)
        outputs['diffuse_color'] = diffuse_colors
        outputs['specular_color'] = specular_colors
        outputs['approximate_light'] = torch.clamp(
            linear_to_srgb(torch.mean(kd[:, :diffuse_num] * diffuse_lights, dim=1) + specular_colors), min=0, max=1)
        return colors, outputs

    def forward(self, pts, view_dirs, normals, human_poses, step, is_train):
        view_dirs, normals = F.normalize(view_dirs, dim=-1), F.normalize(normals, dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        metallic, roughness, albedo = self.predict_materials(pts)  # [pn,1] [pn,1] [pn,3]
        return self.shade_mixed(pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses,
                                is_train)

    def env_light(self, h, w, gamma=True):
        azs = torch.linspace(1.0, 0.0, w) * np.pi * 2 - np.pi / 2
        els = torch.linspace(1.0, -1.0, h) * np.pi / 2

        els, azs = torch.meshgrid(els, azs)
        if self.cfg['is_real']:
            x = torch.cos(els) * torch.cos(azs)
            y = torch.cos(els) * torch.sin(azs)
            z = torch.sin(els)
        else:
            z = torch.cos(els) * torch.cos(azs)
            x = torch.cos(els) * torch.sin(azs)
            y = torch.sin(els)
        xyzs = torch.stack([x, y, z], -1)  # h,w,3
        xyzs = xyzs.reshape(h * w, 3)
        # xyzs = xyzs @ torch.from_numpy(np.asarray([[0,0,1],[0,1,0],[-1,0,0]],np.float32)).cuda()

        batch_size = 8192
        lights = []
        for ri in range(0, h * w, batch_size):
            with torch.no_grad():
                light = self.predict_outer_lights_pts(xyzs[ri:ri + batch_size])
            lights.append(light)
        if gamma:
            lights = linear_to_srgb(torch.cat(lights, 0)).reshape(h, w, 3)
        else:
            lights = (torch.cat(lights, 0)).reshape(h, w, 3)
        return lights

    def predict_outer_lights_pts(self, pts):
        if self.cfg['outer_light_version'] == 'direction':
            return self.outer_light(self.sph_enc(pts, 0))
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            return self.outer_light(torch.cat([self.sph_enc(pts, 0), self.sph_enc(pts, 0)], -1))
        else:
            raise NotImplementedError

    def get_env_light(self):
        return self.predict_outer_lights_pts(self.light_pts)

    def material_regularization(self, pts, normals, metallic, roughness, albedo, step):
        # metallic, roughness, albedo = self.predict_materials(pts)
        reg = 0

        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1)
            x = self.get_orthogonal_directions(normals)
            y = torch.cross(normals, x)
            ang = torch.rand(pts.shape[0], 1) * np.pi * 2
            if self.cfg['change_type'] == 'constant':
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg['change_eps']
            elif self.cfg['change_type'] == 'gaussian':
                eps = torch.normal(mean=0.0, std=self.cfg['change_eps'], size=[x.shape[0], 1])
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            m0, r0, a0 = self.predict_materials(pts + change)
            reg = reg + torch.mean(
                (torch.abs(m0 - metallic) + torch.abs(r0 - roughness) + torch.abs(a0 - albedo)) * self.cfg[
                    'reg_lambda1'], dim=1)

        if self.cfg['reg_min_max'] and step is not None and step < 2000:
            # sometimes the roughness and metallic saturate with the sigmoid activation in the early stage
            reg = reg + torch.sum(torch.clamp(roughness - 0.98 ** 2, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 ** 2 - roughness, min=0))
            reg = reg + torch.sum(torch.clamp(metallic - 0.98, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 - metallic, min=0))

        return reg


def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts, dim=-1) >= 1.0
                    val[outside_mask] = outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles
