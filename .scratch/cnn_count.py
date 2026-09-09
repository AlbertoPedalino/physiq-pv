import itertools, torch
from torch import nn

F_IN, TIME_FEAT = 3, 31

class TemporalCNN(nn.Module):
    """Dilated 1-D conv stack over the 168-hour trend of the target cell."""
    def __init__(self, channels, layers, width):
        super().__init__()
        blocks, in_c = [], F_IN
        for layer in range(layers):
            dilation = 2 ** layer
            blocks += [
                nn.Conv1d(in_c, channels, width, dilation=dilation,
                          padding=(width - 1) * dilation),
                nn.ReLU(),
            ]
            in_c = channels
        self.net = nn.Sequential(*blocks)

class CNNGenerator(nn.Module):
    def __init__(self, *, spatial_channels, temporal_channels, layers, width,
                 hidden_size, k):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(F_IN, spatial_channels, k, padding=k // 2), nn.ReLU())
        self.temporal = TemporalCNN(temporal_channels, layers, width)
        self.time_projection = nn.Sequential(
            nn.Linear(TIME_FEAT, hidden_size), nn.ReLU())
        self.head = nn.Conv2d(
            spatial_channels + temporal_channels + hidden_size, F_IN, 1)

class CNNDiscriminator(nn.Module):
    def __init__(self, *, spatial_channels, hidden_size, k):
        super().__init__()
        self.sequence_encoder = nn.Sequential(
            nn.Conv2d(F_IN, spatial_channels, k, padding=k // 2), nn.ReLU())
        self.sequence_projection = nn.Sequential(
            nn.Linear(k * k * spatial_channels, hidden_size), nn.ReLU())
        self.current = nn.Sequential(
            nn.Conv2d(F_IN, hidden_size, k, padding=k // 2), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid())

def n_params(module):
    return sum(p.numel() for p in module.parameters())

BASE_G, BASE_D = 63171, 36769
K = 3
rows = []
for cs, ct, layers, width, hidden in itertools.product(
        (16, 32, 64), (32, 48, 64, 96), (2, 3, 4), (3, 5, 7), (32, 64)):
    g = n_params(CNNGenerator(spatial_channels=cs, temporal_channels=ct,
                              layers=layers, width=width, hidden_size=hidden, k=K))
    d = n_params(CNNDiscriminator(spatial_channels=cs, hidden_size=hidden, k=K))
    rows.append((abs(g - BASE_G) / BASE_G, abs(d - BASE_D) / BASE_D,
                 cs, ct, layers, width, hidden, g, d))

rows.sort(key=lambda r: max(r[0], r[1]))
print(f"baseline  G={BASE_G}  D={BASE_D}")
print(f"{'Cs':>4}{'Ct':>5}{'L':>3}{'w':>3}{'H':>5}{'G':>9}{'dG%':>7}{'D':>8}{'dD%':>7}")
for r in rows[:12]:
    dg, dd, cs, ct, layers, width, hidden, g, d = r
    print(f"{cs:>4}{ct:>5}{layers:>3}{width:>3}{hidden:>5}"
          f"{g:>9}{100*dg:>7.1f}{d:>8}{100*dd:>7.1f}")

print()
print("discriminator sweep (own spatial channels)")
print(f"{'Cs_d':>5}{'H':>5}{'D':>9}{'dD%':>7}")
best = []
for cs in range(8, 129, 4):
    for hidden in (32, 48, 64, 96):
        d = n_params(CNNDiscriminator(spatial_channels=cs, hidden_size=hidden, k=K))
        best.append((abs(d - BASE_D) / BASE_D, cs, hidden, d))
best.sort()
for dd, cs, hidden, d in best[:8]:
    print(f"{cs:>5}{hidden:>5}{d:>9}{100*dd:>7.1f}")
