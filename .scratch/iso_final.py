import itertools
from importlib.machinery import SourceFileLoader

m = SourceFileLoader("cnn_count", ".scratch/cnn_count.py")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mod = m.load_module()

BASE_G, BASE_D, K = 63171, 36769, 3
rf = lambda L, w: 1 + sum((w - 1) * 2 ** i for i in range(L))

rows = []
for cs, ct, L, w, H in itertools.product(
        (8, 16, 24, 32, 48, 64), (16, 24, 32, 48, 64), (5, 6, 7), (5, 7), (32, 64)):
    if rf(L, w) < 168:
        continue
    g = mod.n_params(mod.CNNGenerator(spatial_channels=cs, temporal_channels=ct,
                                      layers=L, width=w, hidden_size=H, k=K))
    rows.append((abs(g - BASE_G) / BASE_G, cs, ct, L, w, H, g, rf(L, w)))
rows.sort()
print("generator configs with receptive field >= 168 h, sorted by parameter gap")
print(f"{'Cs':>4}{'Ct':>5}{'L':>3}{'w':>3}{'H':>5}{'RF':>6}{'G':>9}{'dG%':>7}")
for dg, cs, ct, L, w, H, g, r in rows[:10]:
    print(f"{cs:>4}{ct:>5}{L:>3}{w:>3}{H:>5}{r:>6}{g:>9}{100*dg:>7.2f}")
