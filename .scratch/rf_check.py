import itertools
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("m", ".scratch/cnn_count.py")

def receptive_field(layers, width):
    return 1 + sum((width - 1) * 2 ** layer for layer in range(layers))

print(f"{'L':>3}{'w':>3}{'RF(h)':>8}  covers 168h")
for layers, width in itertools.product((2, 3, 4, 5, 6), (3, 5, 7)):
    rf = receptive_field(layers, width)
    print(f"{layers:>3}{width:>3}{rf:>8}  {'yes' if rf >= 168 else 'no'}")
