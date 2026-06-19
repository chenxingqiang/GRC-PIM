"""GF(3) codec and vectorized NB-LDPC BP decoder.

Loads pre-built H/G matrices from deps/NBLDPC_PIM/DAC_TEST and provides:
  - encode():       info symbols (GF(3)) -> codeword (GF(3))
  - syndrome():     codeword -> H @ y mod 3
  - bp_decode():    iterative belief-propagation decode over GF(3)

Code parameters (default quick test):
    - Field GF(3), 256 info symbols, 32 check symbols, n=288, rate=8/9

Paper-scale parameters from Shi et al.:
    - Field GF(3), 1024 info symbols, 64 check symbols, n=1088, rate=16/17
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


GF = 3  # Galois field order


def load_matrix(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load whitespace-separated integer matrix (H or G) from a text file."""
    arr = np.loadtxt(str(path), dtype=np.int64).reshape(shape)
    return arr % GF


def load_default_HG(repo_root: str | Path = ".") -> tuple[np.ndarray, np.ndarray]:
    """Load the default 256/288 GF(3) H, G matrices shipped with deps/NBLDPC_PIM."""
    return load_HG(repo_root=repo_root, info_length=256, rate="8_9")


def load_HG(repo_root: str | Path = ".", info_length: int = 256,
            rate: str = "8_9") -> tuple[np.ndarray, np.ndarray]:
    """Load GF(3) H, G matrices shipped with deps/NBLDPC_PIM.

    Args:
      repo_root: workspace/repository root.
      info_length: number of information GF(3) symbols, e.g. 256 or 1024.
      rate: filename rate tag such as "8_9" or "16_17".
    """
    base = Path(repo_root) / "deps" / "NBLDPC_PIM" / "DAC_TEST"
    num, den = (int(x) for x in rate.split("_"))
    if num <= 0 or den <= num:
        raise ValueError(f"rate must look like '8_9' or '16_17', got {rate!r}")
    n_symbols = info_length * den // num
    if n_symbols * num != info_length * den:
        raise ValueError(f"info_length={info_length} is not divisible by rate={rate}")
    check_symbols = n_symbols - info_length
    H = load_matrix(base / f"H_{info_length}_{rate}.txt", (check_symbols, n_symbols))
    G = load_matrix(base / f"G_{info_length}_{rate}.txt", (info_length, n_symbols))
    return H, G


def encode(info: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Encode info symbols (..., k) over GF(3) into codewords (..., n)."""
    return (info @ G) % GF


def syndrome(y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute syndrome H @ y^T mod 3. y shape (..., n) -> (..., m)."""
    return (y @ H.T) % GF


# ----------------------------------------------------------------------------
# Vectorised belief propagation decoder over GF(3)
# ----------------------------------------------------------------------------
# We use log-domain BP with the message-passing schedule of Davey-MacKay
# specialised to GF(3). For each variable v_i we keep a length-3 LLR vector
# L_i[s] = log P(x_i = s | y). For each non-zero edge (j, i) of H we keep an
# outgoing message m_{j -> i}[s].
#
# Implementation choice: we vectorise over codewords (batch) and edges, but
# loop over the (small) number of BP iterations.

def _gf3_inv(a: int) -> int:
    """Multiplicative inverse in GF(3): inv(1)=1, inv(2)=2."""
    return a  # since 1*1 = 1, 2*2 = 4 = 1 mod 3


def init_llr_from_bits(noisy_bits: np.ndarray, p_bit: float,
                       n_info: int = 256) -> np.ndarray:
    """Initial LLR for each codeword symbol given noisy 2-bit binary readout.

    Each GF(3) symbol s in {0,1,2} is transmitted as 2 bits b1 b0 with
    b1 b0 = 00, 01, 10 respectively. Symbol "11" is illegal -> nearest legal
    symbol gets the highest LLR (Manhattan tie-breaking, matches paper).

    Args:
      noisy_bits: (B, 2*n) array of 0/1 read out from the PIM array.
      p_bit:      assumed per-bit flip probability.
      n_info:     # of info symbols (the rest are check symbols).
    Returns:
      llr: (B, n, 3) float array of log-probabilities.
    """
    B, two_n = noisy_bits.shape
    n = two_n // 2
    # symbol_obs: (B, n) integer in {0,1,2,3}
    bits = noisy_bits.reshape(B, n, 2)
    obs = bits[..., 0] * 2 + bits[..., 1]  # 00->0, 01->1, 10->2, 11->3
    # Hamming distance from each candidate s ∈ {0,1,2} (encoded as 00,01,10)
    cand_bits = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)  # (3, 2)
    # dist[B, n, s] = sum |obs_bits - cand_bits[s]|
    dist = np.abs(bits[..., None, :] - cand_bits[None, None, :, :]).sum(axis=-1)
    # P(obs | x = s) = p_bit^d * (1-p_bit)^(2-d)
    log_p = (dist * np.log(max(p_bit, 1e-12))
             + (2 - dist) * np.log(max(1 - p_bit, 1e-12)))
    # Normalise log-likelihood per symbol
    log_p = log_p - log_p.max(axis=-1, keepdims=True)
    return log_p.astype(np.float32)


def bp_decode(llr: np.ndarray, H: np.ndarray, n_iter: int = 3,
              early_stop: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Sum-product BP decoder over GF(3).

    Args:
      llr:    (B, n, 3) prior log-likelihoods.
      H:      (m, n) parity-check matrix in GF(3).
      n_iter: max BP iterations.
      early_stop: if True, stop once syndrome is satisfied for all codewords.
    Returns:
      x_hat:    (B, n) hard decisions in GF(3).
      post_llr: (B, n, 3) posterior log-likelihoods.
      iters_used: number of iterations actually run.
    """
    B, n, _ = llr.shape
    m = H.shape[0]

    # Edge list: (j, i, h) where h = H[j, i] != 0
    j_idx, i_idx = np.where(H != 0)
    h_val = H[j_idx, i_idx].astype(np.int64)  # weights
    E = j_idx.shape[0]

    # Variable -> Check messages mu[B, E, 3], initially equal to the prior
    mu_v2c = llr[:, i_idx, :].copy()  # (B, E, 3)
    mu_c2v = np.zeros_like(mu_v2c)

    # Pre-compute the GF(3) "value seen by check j from edge e" lookup.
    # For check j: sum_e h_e * x_e = 0 mod 3, i.e. for outgoing edge e,
    # the desired value of x_e is x_e* = -h_e^{-1} * sum_{e' != e} h_{e'} x_{e'}.
    # We compute this by Forward-Backward in log domain over edges of each check.

    # Group edges by check
    order = np.argsort(j_idx)
    j_sorted = j_idx[order]
    edge_per_check: list[np.ndarray] = []
    for j in range(m):
        edge_per_check.append(order[j_sorted == j])

    iters_used = n_iter
    for it in range(n_iter):
        # ---- check-node update -----------------------------------------
        for j in range(m):
            edges = edge_per_check[j]                  # (Dc,)
            hv = h_val[edges]                          # (Dc,)
            # Permute incoming messages so that values index the GF target
            # mu_in[B, Dc, s] -> mu_perm[B, Dc, s'] where s' = h_e * s mod 3
            # Equivalently, mu_perm[..., s'] = mu_in[..., inv(h_e) * s' mod 3].
            mu_in = mu_v2c[:, edges, :]                # (B, Dc, 3)
            inv_h = np.array([_gf3_inv(int(h)) for h in hv])  # (Dc,)
            src_idx = (inv_h[:, None] * np.arange(GF)[None, :]) % GF  # (Dc, 3)
            idx_broad = np.broadcast_to(src_idx[None], mu_in.shape)
            mu_perm = np.take_along_axis(mu_in, idx_broad, axis=-1)
            # Forward-backward sum-product in log domain via "logaddconv" of GF
            # for GF(3) it is convolution mod 3 of probability vectors.
            Dc = edges.shape[0]
            fwd = np.zeros((B, Dc, GF), dtype=np.float32)
            bwd = np.zeros((B, Dc, GF), dtype=np.float32)
            fwd[:, 0, :] = mu_perm[:, 0, :]
            for k in range(1, Dc):
                fwd[:, k, :] = _logconv3(fwd[:, k - 1, :], mu_perm[:, k, :])
            bwd[:, Dc - 1, :] = mu_perm[:, Dc - 1, :]
            for k in range(Dc - 2, -1, -1):
                bwd[:, k, :] = _logconv3(bwd[:, k + 1, :], mu_perm[:, k, :])
            # Outgoing message at edge k = conv(fwd[k-1], bwd[k+1]) excluding k
            out = np.zeros_like(mu_perm)
            out[:, 0, :] = bwd[:, 1, :] if Dc > 1 else 0.0
            out[:, Dc - 1, :] = fwd[:, Dc - 2, :] if Dc > 1 else 0.0
            for k in range(1, Dc - 1):
                out[:, k, :] = _logconv3(fwd[:, k - 1, :], bwd[:, k + 1, :])
            # The check enforces total = 0; the outgoing message at edge k
            # represents log P(x_e = s) where s = -inv(h_e) * (sum_others).
            # We must permute the convolution result back to the x_e domain:
            # out_x[s] = out_perm[(-h_e * s) mod 3].
            for k in range(Dc):
                he = int(hv[k])
                src = (-he * np.arange(GF)) % GF       # (3,)
                tmp = out[:, k, src]                    # (B, 3)
                mu_c2v[:, edges[k], :] = tmp - tmp.max(axis=-1, keepdims=True)

        # ---- variable-node update --------------------------------------
        # Posterior = prior + sum of incoming check messages
        post_llr = llr.copy()
        np.add.at(post_llr, (slice(None), i_idx, slice(None)), mu_c2v)
        # Outgoing v->c message excludes the corresponding incoming c->v
        mu_v2c = post_llr[:, i_idx, :] - mu_c2v
        mu_v2c -= mu_v2c.max(axis=-1, keepdims=True)

        # ---- early stopping --------------------------------------------
        x_hat = np.argmax(post_llr, axis=-1)
        if early_stop:
            syn = (x_hat @ H.T) % GF
            if np.all(syn == 0):
                iters_used = it + 1
                break

    return x_hat, post_llr, iters_used


def _logconv3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum-product convolution over GF(3) in the log domain.

    For probability vectors p and q indexed by GF(3), the convolution is
    (p * q)[k] = sum_i p[i] * q[(k - i) mod 3]. We work in log-domain using
    log-sum-exp.
    """
    # Stack the three shifted products and logsumexp
    out = np.empty_like(a)
    for k in range(GF):
        # (k - i) mod 3 for i = 0,1,2
        terms = np.stack(
            [a[..., i] + b[..., (k - i) % GF] for i in range(GF)], axis=-1
        )
        m = terms.max(axis=-1, keepdims=True)
        out[..., k] = (m + np.log(np.exp(terms - m).sum(axis=-1, keepdims=True)))[..., 0]
    return out
