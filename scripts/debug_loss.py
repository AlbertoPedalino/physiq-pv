import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, '.')
from main import _normalize_dataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.dataset import PVDataset
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from torch.utils.data import DataLoader

ds = xr.open_dataset('data/real_data_dataset.nc')
ds = _normalize_dataset(ds)
n_steps = ds.sizes["time"]
fit_time_mask = np.arange(n_steps) < int(0.8 * n_steps)
_qs, m_components = compute_qs(
    ds,
    debug=True,
    fit_time_mask=fit_time_mask,
)

dataset = PVDataset(
    ds,
    m_components,
    seq_len=24,
    fit_time_mask=fit_time_mask,
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
(
    x,
    y_poa,
    y_pv,
    pr_proxy,
    poa_cs,
    poa_scale,
    pv_target_valid,
    _pv_lag_valid,
) = next(iter(loader))

print('y_pv  :', y_pv.min().item(), '->', y_pv.max().item())
print('y_poa :', y_poa.min().item(), '->', y_poa.max().item())
print('pr    :', pr_proxy.min().item(), '->', pr_proxy.max().item())
print('x nan :', torch.isnan(x).any().item())

lats = ds['lat'].values
lons = ds['lon'].values
ei, ew = build_graph(lats, lons)

model = STGNN(
    n_nodes=ds.sizes["plant"], n_features=dataset.feats.shape[-1], seq_len=24,
    d_model=128,
    gat_dim=96, gat_heads=4, gat_layers=1,
).cuda()

x     = x.cuda()
y_poa = y_poa.cuda()
y_pv  = y_pv.cuda()
pr_proxy = pr_proxy.cuda()
poa_cs = poa_cs.cuda()
poa_scale = poa_scale.cuda()
pv_target_valid = pv_target_valid.cuda()
ei    = ei.cuda()
ew    = ew.cuda()

pred_poa, pp = model(x, ei, ew, poa_cs)
print('pred_poa:', pred_poa.min().item(), '->', pred_poa.max().item(), '| nan:', torch.isnan(pred_poa).any().item())
print('pred_pv :', pp.min().item(), '->', pp.max().item(), '| nan:', torch.isnan(pp).any().item())

loss, breakdown = physics_loss_full(
    pred_poa,
    pp,
    y_poa,
    y_pv,
    pr_proxy,
    poa_scale,
    pv_valid=pv_target_valid,
)
print('loss    :', loss.item())
print('breakdown:', breakdown)
