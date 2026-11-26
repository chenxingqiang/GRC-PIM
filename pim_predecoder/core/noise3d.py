"""Multi-cycle PIM noise model with temporal & spatial correlation.

Adds a temporal axis to the bit-flip channel so a 3D CNN has something
non-trivial to exploit. The same code grid is "read" T times in a row;
the noise model is the sum of:

  - independent per-(row, time, bit) flips at rate p_indep,
  - column-correlated drift: a per-column bias that persists for K cycles,
    then resamples (mimics IR-drop / thermal drift),
  - row-correlated burst: occasionally a whole row flips one bit
    (mimics common-mode read disturbance).

The clean GF(3) symbols are constant across the T cycles (we are reading
the *same* MAC result T times to enable cycle averaging); only the noise
varies.
"""
from __future__ import annotations

import numpy as np

from .codec import GF
from .noise import bits_to_symbols, symbols_to_bits


def apply_pim_noise_3d(
    code_sym: np.ndarray,
    n_cycles: int,
    p_indep: float,
    p_drift: float = 0.0,
    drift_persist: int = 3,
    p_row_burst: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate (B, T, 2*n) noisy bit reads.

    Args:
      code_sym:        (B, n) clean GF(3) codewords.
      n_cycles:        T, number of consecutive MAC cycles to simulate.
      p_indep:         per-bit independent flip probability.
      p_drift:         probability that a column has a persistent biased
                       flip during any cycle.
      drift_persist:   how many consecutive cycles a drift event lasts.
      p_row_burst:     per-row, per-cycle probability of a 1-bit burst.
    """
    if rng is None:
        rng = np.random.default_rng()
    B, n = code_sym.shape
    bits = symbols_to_bits(code_sym).astype(np.int64)         # (B, 2n)
    out = np.broadcast_to(bits[:, None, :], (B, n_cycles, 2 * n)).copy()

    # 1) independent flips
    flip = rng.random(out.shape) < p_indep
    out ^= flip.astype(np.int64)

    # 2) column drift: pick events (col, start_t) and flip that column
    #    in [start_t, start_t + drift_persist) cycles.
    if p_drift > 0:
        n_events = rng.binomial(2 * n * n_cycles, p_drift)
        if n_events:
            cols = rng.integers(0, 2 * n, size=n_events)
            starts = rng.integers(0, n_cycles, size=n_events)
            for c, s in zip(cols, starts):
                end = min(n_cycles, s + drift_persist)
                out[:, int(s):int(end), int(c)] ^= 1

    # 3) row burst: per-row, per-cycle, with prob p_row_burst flip 1 bit
    if p_row_burst > 0:
        burst = rng.random((B, n_cycles)) < p_row_burst
        if burst.any():
            idx = np.where(burst)
            cols = rng.integers(0, 2 * n, size=idx[0].shape[0])
            out[idx[0], idx[1], cols] ^= 1

    return out


def collapse_majority(noisy_bits_3d: np.ndarray) -> np.ndarray:
    """Bit-wise majority vote across cycles -> (B, 2n) bits.

    This is the classical "average T reads" defence and is the right
    baseline to compare a 3D-CNN against.
    """
    return (noisy_bits_3d.mean(axis=1) >= 0.5).astype(np.int64)


def apply_pim_noise_conductance(
    code_sym: np.ndarray,
    n_cycles: int,
    sigma_ln: float = 0.0,
    drift_nu: float = 0.0,
    rtn_traps: int = 0,
    rtn_amp: float = 0.0,
    rtn_switch: float = 0.5,
    read_spacing: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Conductance-domain read channel; errors emerge from ADC thresholds.

    ``apply_pim_noise_3d`` injects errors directly as Bernoulli bit flips.
    This channel instead models the analog quantities a real PIM macro
    actually perturbs, and lets bit errors appear only when a conductance
    crosses an ADC decision threshold:

      1. **Programming / cell-to-cell spread** - a static multiplicative
         log-normal offset per cell (``sigma_ln``), the standard description
         of RRAM LRS/HRS variability.
      2. **Retention drift** - a persistent power-law decay
         ``G(t) = G_0 (t/t_0)^{-nu}`` (``drift_nu``) applied to every read in
         the multi-cycle sequence, so the perturbation is correlated in time
         rather than redrawn per read.
      3. **Random telegraph noise** - ``rtn_traps`` independent two-state
         traps per cell, each following a Markov chain with per-cycle
         switching probability ``rtn_switch`` and fractional amplitude
         ``rtn_amp``. Several traps produce the multi-level and anomalous
         RTN reported for HfO2 RRAM, not a two-level signal.

    The resulting impairment is *structurally* different from the Bernoulli
    channel used for training: bit errors within one symbol are correlated
    (a single level slip can move both bits), and the temporal correlation is
    produced by device physics instead of an injected drift event. It is
    therefore a genuine out-of-distribution test of the reliability gate.

    Args:
      code_sym:      (B, n) clean GF(3) codewords.
      n_cycles:      T, number of consecutive read cycles.
      sigma_ln:      sigma of the static log-normal conductance spread.
      drift_nu:      retention-drift exponent (0 disables drift).
      rtn_traps:     number of two-state RTN traps per cell.
      rtn_amp:       fractional conductance step of one active trap.
      rtn_switch:    per-cycle trap switching probability.
      read_spacing:  multiplicative time step between consecutive reads.
      rng:           numpy random generator.
    Returns:
      (B, T, 2n) binary readout, matching ``apply_pim_noise_3d``.
    """
    if rng is None:
        rng = np.random.default_rng()
    batch_size, n = code_sym.shape

    # GF(3) levels placed at 1/4, 2/4, 3/4 of the ADC range; decision
    # thresholds sit midway between neighbours.
    levels = (np.arange(GF, dtype=np.float64) + 1.0) / (GF + 1.0)
    thresholds = (levels[:-1] + levels[1:]) / 2.0

    g0 = levels[np.asarray(code_sym) % GF]                      # (B, n)
    if sigma_ln > 0:
        g0 = g0 * rng.lognormal(mean=0.0, sigma=sigma_ln, size=g0.shape)

    # Persistent retention decay: read t occurs at normalised time 1 + t*spacing.
    times = 1.0 + read_spacing * np.arange(n_cycles, dtype=np.float64)
    decay = times ** (-float(drift_nu))                         # (T,)
    g = g0[:, None, :] * decay[None, :, None]                   # (B, T, n)

    if rtn_traps > 0 and rtn_amp > 0:
        active = rng.random((batch_size, rtn_traps, n)) < 0.5
        occupancy = np.empty((batch_size, n_cycles, n), dtype=np.float64)
        for t in range(n_cycles):
            if t > 0:
                toggle = rng.random((batch_size, rtn_traps, n)) < rtn_switch
                active = np.logical_xor(active, toggle)
            occupancy[:, t, :] = active.sum(axis=1)
        g = g * (1.0 - rtn_amp * occupancy)

    sym_hat = np.digitize(g, thresholds).astype(np.int64)       # (B, T, n)

    bit1 = (sym_hat >> 1) & 1
    bit0 = sym_hat & 1
    return np.stack([bit1, bit0], axis=-1).reshape(
        batch_size, n_cycles, 2 * n)


# Device window of the CMO/HfOx cells behind aihwkit's ReRamCMONoiseModel,
# i.e. the defaults of its SingleDeviceConductanceConverter.
AIHWKIT_G_MIN_US = 9.0
AIHWKIT_G_MAX_US = 88.199997


def apply_pim_noise_aihwkit(
    code_sym: np.ndarray,
    n_cycles: int,
    t_inference: float = 86400.0,
    level_pitch_us: float = 8.0,
    acceptance_range: float = 2.0,
    prog_noise_scale: float = 1.0,
    read_noise_scale: float = 1.0,
    drift_scale: float = 1.0,
    g_min_us: float = AIHWKIT_G_MIN_US,
    g_max_us: float = AIHWKIT_G_MAX_US,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Conductance-domain read channel driven by a third-party device model.

    ``apply_pim_noise_conductance`` is our own physics-inspired channel. This
    function replaces its noise physics with ``ReRamCMONoiseModel`` from
    aihwkit (IBM Analog Hardware Acceleration Kit, MIT licence), whose
    coefficients are fitted to measured analog filamentary CMO/HfOx ReRAM from
    IBM Research Europe - Zurich. Everything else - the GF(3) level placement,
    the ADC decision rule, and the multi-cycle readout - is identical, so the
    only factor that changes is who authored the noise model. It is therefore
    a cross-validation of the reliability gate against an independent,
    device-calibrated channel.

    The device model is used through its public API only, split into the static
    and the per-read part that a multi-cycle readout requires:

      * **Static state**, drawn once per cell: programming noise from the
        program-and-verify acceptance range, plus the conductance decay (mean
        shift linear in ``log t``) and the relaxation spread. Obtained by
        instantiating the model with ``read_noise_scale=0``.
      * **Per-read 1/f noise**, drawn independently for each of the ``T``
        reads: the model's read-noise term
        ``K log10(G) sqrt(log((t + t_read) / 2 t_read))``. Obtained by
        instantiating the model with ``drift_scale=0``, which cancels both the
        mean shift and the relaxation term and leaves exactly the read noise.

    This split is required because the device model exposes drift and read
    noise through a single call that redraws both; a cell's relaxation state
    must instead persist across the ``T`` reads of one sequence.

    Args:
      code_sym:          (B, n) clean GF(3) codewords.
      n_cycles:          T, number of consecutive read cycles.
      t_inference:       retention time in seconds since programming; drives
                         both the conductance decay and the 1/f read noise.
      level_pitch_us:    conductance spacing between adjacent GF(3) levels, in
                         microsiemens. This is the macro's level density, not a
                         device property: it sets how much of the device window
                         one symbol is allowed to consume and hence the ADC
                         margin the device noise has to cross.
      acceptance_range:  program-and-verify acceptance range in percent; the
                         device model provides fits for 0.2 and 2.
      prog_noise_scale:  scale on the programming noise.
      read_noise_scale:  scale on the 1/f read noise. 1 is the measured device;
                         larger values model a cell with stronger read noise
                         and are the knob that moves the drift score across the
                         gate band at a fixed ADC margin.
      drift_scale:       scale on the conductance decay and relaxation.
      g_min_us, g_max_us: device conductance window in microsiemens.
      rng:               numpy generator; only used to seed torch, which the
                         device model samples from.
    Returns:
      (B, T, 2n) binary readout, matching ``apply_pim_noise_3d``.
    """
    import torch
    from aihwkit.inference.noise.reram import ReRamCMONoiseModel

    if rng is None:
        rng = np.random.default_rng()
    if t_inference <= 0:
        raise ValueError("t_inference must be positive; use read_noise_scale=0 "
                         "for a static-only operating point")
    torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))

    # GF(3) levels centred in the device window, spaced by level_pitch_us;
    # decision thresholds sit midway between neighbours, as in
    # apply_pim_noise_conductance.
    centre = 0.5 * (g_min_us + g_max_us)
    levels = centre + (np.arange(GF, dtype=np.float64) - (GF - 1) / 2.0) * level_pitch_us
    if levels[0] <= g_min_us or levels[-1] >= g_max_us:
        raise ValueError(f"level_pitch_us={level_pitch_us} does not fit the "
                         f"device window [{g_min_us}, {g_max_us}] uS")
    thresholds = (levels[:-1] + levels[1:]) / 2.0

    static = ReRamCMONoiseModel(
        g_max=g_max_us, g_min=g_min_us, acceptance_range=acceptance_range,
        prog_noise_scale=prog_noise_scale, drift_scale=drift_scale,
        read_noise_scale=0.0,
    )
    read = ReRamCMONoiseModel(
        g_max=g_max_us, g_min=g_min_us, acceptance_range=acceptance_range,
        read_noise_scale=read_noise_scale, drift_scale=0.0,
    )

    g_target = torch.as_tensor(
        levels[np.asarray(code_sym) % GF], dtype=torch.float32)
    g_prog = static.apply_programming_noise_to_conductance(g_target)
    g_static = static.apply_drift_noise_to_conductance(g_prog, g_prog, t_inference)

    batch_size, n = g_static.shape
    g_rep = g_static.unsqueeze(1).expand(batch_size, n_cycles, n).contiguous()
    g_reads = read.apply_drift_noise_to_conductance(g_rep, g_rep, t_inference)

    sym_hat = np.digitize(g_reads.numpy(), thresholds).astype(np.int64)

    bit1 = (sym_hat >> 1) & 1
    bit0 = sym_hat & 1
    return np.stack([bit1, bit0], axis=-1).reshape(
        batch_size, n_cycles, 2 * n)
