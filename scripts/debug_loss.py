import torch, xarray as xr, numpy as np, sys
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
_qs, m_components = compute_qs(ds, debug=True)

dataset = PVDataset(ds, m_components, seq_len=120)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
x, y_ghi, y_pv, eta = next(iter(loader))

print('y_pv  :', y_pv.min().item(), '->', y_pv.max().item())
print('y_ghi :', y_ghi.min().item(), '->', y_ghi.max().item())
print('eta   :', eta.min().item(), '->', eta.max().item())
print('x nan :', torch.isnan(x).any().item())

lats = ds['lat'].values
lons = ds['lon'].values
ei, ew = build_graph(lats, lons)

model = STGNN(
    n_nodes=ds.sizes["plant"], n_features=dataset.feats.shape[-1], seq_len=120,
    patch_len=16, stride=8, d_model=128,
    gat_dim=256, gat_heads=4, gat_layers=2,
).cuda()

x     = x.cuda()
y_ghi = y_ghi.cuda()
y_pv  = y_pv.cuda()
eta   = eta.cuda()
ei    = ei.cuda()
ew    = ew.cuda()

pg, pp = model(x, ei, ew)
print('pred_ghi:', pg.min().item(), '->', pg.max().item(), '| nan:', torch.isnan(pg).any().item())
print('pred_pv :', pp.min().item(), '->', pp.max().item(), '| nan:', torch.isnan(pp).any().item())

loss, breakdown = physics_loss_full(pg, pp, y_ghi, y_pv, eta)
print('loss    :', loss.item())
print('breakdown:', breakdown)
