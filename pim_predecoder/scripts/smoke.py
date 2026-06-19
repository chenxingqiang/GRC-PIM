"""Smoke test the codec + BP decoder pipeline."""
import numpy as np
from pim_predecoder.core.codec import (
    load_default_HG, encode, syndrome, init_llr_from_bits, bp_decode,
)
from pim_predecoder.core.noise import apply_pim_noise

rng = np.random.default_rng(0)
H, G = load_default_HG(".")
print("H", H.shape, "G", G.shape,
      "values in H:", sorted(set(H.flatten().tolist())))

info = rng.integers(0, 3, size=(4, G.shape[0]))
code = encode(info, G)
print("code shape", code.shape,
      "syndrome sum (should be 0):", int(syndrome(code, H).sum()))

nb = apply_pim_noise(code, p_bit=1e-3, rng=rng)
llr = init_llr_from_bits(nb, p_bit=1e-3)
xh, post, it = bp_decode(llr, H, n_iter=3)
print("decoded shape", xh.shape, "iters used", it)
mismatch = (xh - code) % 3
print("post-ECC mismatches per cw:", (mismatch != 0).sum(axis=-1))
