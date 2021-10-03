import struct
# Define colours here
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
BLACK = "#000000"
WHITE = "#FFFFFF"
GREY = "#888888"

# Evil hacky global stuff to generate floating point tuples from the colours above.
# On import, if XXX is a hexcode, then XXX_F is the corresponding float tuple
colors = [var for var, val in globals().items() if isinstance(val, str) and var[0] != "_"]
_g = globals()
for var in colors:
    _g[var + "_F"] = tuple(i / 255 for i in struct.unpack('BBB', bytes.fromhex(_g[var][1:])))

