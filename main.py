import numpy as np
import pandas as pd

TYPES = ["ILE","SEI","ESE","LII","EIE","LSI","SLE","IEI",
         "SEE","ILI","LIE","ESI","LSE","EII","IEE","SLI"]

# rows 0..32767 ; we'll treat the row index as the 15-bit "other" mask
masks = np.arange(32768, dtype=np.uint16)
bits = np.zeros((32768, 16), dtype=np.uint8)
bits[:, 0] = 1                                 # ILE column: always 1
for i in range(15):
    bits[:, i+1] = (masks >> i) & 1            # bit i -> column i+1 (SEI..SLI)

df = pd.DataFrame(bits, columns=TYPES)
df["popcount"] = bits.sum(axis=1)
df["is_octad"] = df["popcount"] == 8
df["valid"]   = df["popcount"] < 16            # drop the degenerate all-1 row
# df.to_csv("Data/dichotomies.csv", index=False)

items = pd.read_csv("Data/items.csv")
bool_cols = [f"{t}_bool" for t in TYPES]
X = items[bool_cols].to_numpy(dtype=np.uint8)

pole = np.where(X[:, 0] == 1, 1, -1)     # +1 if ILE is 1, else -1
X = np.where(X[:, :1] == 1, X, 1 - X)    # flip rows where ILE was 0
items_norm = pd.DataFrame(X, columns=TYPES)
items_norm["pole"] = pole
