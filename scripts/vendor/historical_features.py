"""Historical function bodies, with CLI-only setup replaced by initialize()."""
import time, math
from collections import Counter
import numpy as np
import pandas as pd
from scipy.integrate import quad
from .energy_funcs import ah_scaled, njit
def encode_seq(seq: str) -> np.ndarray:
    return np.array([aa_to_idx[a] for a in seq if a in aa_to_idx], dtype=np.int32)

def per_res_props(idx_arr: np.ndarray):
    return LAMB[idx_arr], SIGM[idx_arr], QVAL[idx_arr]

def build_AH_matrix(rc=2.0, eps=0.2 * 4.184):
    A = np.zeros((len(aas), len(aas)), dtype=float)
    for i in range(len(aas)):
        sig1 = SIGM[i]; lamb1 = LAMB[i]
        for j in range(len(aas)):
            sig2 = SIGM[j]; lamb2 = LAMB[j]
            sigma = 0.5 * (sig1 + sig2)
            lamb = 0.5 * (lamb1 + lamb2)
            lo = (2.0 ** (1 / 6)) * sigma
            A[i, j] = quad(lambda r: ah_scaled(r, sigma, eps, lamb, rc), lo, rc)[0]
    return A

@njit(nopython=True)
def get_qs_fast(seq_chars):
    qs = np.zeros(len(seq_chars))
    qs_abs = np.zeros(len(seq_chars))
    for i in range(len(seq_chars)):
        c = seq_chars[i]
        if c == 'R' or c == 'K':
            qs[i] = 1.0; qs_abs[i] = 1.0
        elif c == 'E' or c == 'D':
            qs[i] = -1.0; qs_abs[i] = 1.0
        else:
            qs[i] = 0.0; qs_abs[i] = 0.0
    return qs, qs_abs

@njit(nopython=True)
def calc_SCD(seq_chars, charge_termini=False):
    qs, _ = get_qs_fast(seq_chars)
    if charge_termini:
        if len(qs) > 0: qs[0] += 1.0
        if len(qs) > 1: qs[-1] -= 1.0
    N = len(qs)
    if N == 0: return 0.0
    scd = 0.0
    for i in range(1, N):
        for j in range(0, i):
            scd += qs[i] * qs[j] * (i - j) ** 0.5
    return scd / N

def calc_SHD_fast(idx_arr, beta=-1.0):
    N = len(idx_arr)
    if N <= 1: return 0.0
    l = LAMB[idx_arr]
    shd = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            shd += (l[i] + l[j]) * ((j - i) ** beta)
    return shd / N

def calc_ah_between_sequences_idx(idx1, idx2):
    if len(idx1) == 0 or len(idx2) == 0: return 0.0
    s = AH[np.ix_(idx1, idx2)].sum()
    return s / (len(idx1) * len(idx2))

def aromatic_fraction(seq_chars):
    n = len(seq_chars)
    if n == 0: return 0.0
    cnt = 0
    for c in seq_chars:
        if c in ('F', 'Y', 'W'):
            cnt += 1
    return cnt / n

def rg_fraction(seq_chars):
    n = len(seq_chars)
    if n == 0: return 0.0
    cnt = 0
    for c in seq_chars:
        if c == 'R' or c == 'G':
            cnt += 1
    return cnt / n

def rgg_motif_count(seq_chars):
    s = "".join(seq_chars)
    return s.count("RG")

def sequence_entropy(seq_chars):
    n = len(seq_chars)
    if n == 0: return 0.0
    c = Counter(seq_chars)
    probs = np.array([v / n for v in c.values()], dtype=float)
    nz = probs[probs > 0]
    return float(-(nz * np.log2(nz)).sum())

def calc_kappa(seq_chars):
    if len(seq_chars) < 2: return 0.0
    qs, qs_abs = get_qs_fast(seq_chars)
    N = len(qs)
    q_total = np.sum(qs)
    q_abs_total = np.sum(qs_abs)
    if q_abs_total == 0 or q_total == 0: return 0.0
    S = 0.0
    for i in range(N):
        for j in range(N):
            S += qs[i] * qs[j] * abs(i - j)
    return S / (q_total ** 2 * N)

def opp_charges_total(qi_sum_pos, qi_sum_neg, qj_sum_pos, qj_sum_neg):
    return (qi_sum_pos * qj_sum_neg + qi_sum_neg * qj_sum_pos) - (qi_sum_pos * qj_sum_pos + qi_sum_neg * qj_sum_neg)

def patch_count_matrix(seq_idx, w):
    L = len(seq_idx)
    P = max(L - w + 1, 0)
    if P == 0:
        return np.zeros((0, len(aas)), dtype=np.int32)
    M = np.zeros((P, len(aas)), dtype=np.int32)
    # first window
    window = seq_idx[:w]
    for a in window: M[0, a] += 1
    # roll
    for p in range(1, P):
        out_a = seq_idx[p - 1]; in_a = seq_idx[p + w - 1]
        M[p] = M[p - 1]
        M[p, out_a] -= 1; M[p, in_a] += 1
    return M

def patch_scalar_features(seq_chars, seq_idx, w):
    P = max(len(seq_idx) - w + 1, 0)
    if P == 0:
        zeros = np.zeros(0)
        return dict(scd=zeros, shd=zeros, rg=zeros, arom=zeros, rgg=zeros,
                    kappa=zeros, entropy=zeros, qpos=zeros, qneg=zeros)

    scd = np.zeros(P); shd = np.zeros(P); rg = np.zeros(P); arom = np.zeros(P)
    rgg = np.zeros(P); kap = np.zeros(P); ent = np.zeros(P)
    qpos = np.zeros(P); qneg = np.zeros(P)

    for p in range(P):
        idxs = seq_idx[p:p + w]
        chars = [aas[i] for i in idxs]
        scd[p] = calc_SCD(chars)
        shd[p] = calc_SHD_fast(idxs)
        rg[p] = rg_fraction(chars)
        arom[p] = aromatic_fraction(chars)
        rgg[p] = rgg_motif_count(chars)
        kap[p] = calc_kappa(chars)
        ent[p] = sequence_entropy(chars)
        q = QVAL[idxs]
        qpos[p] = q[q > 0].sum()
        qneg[p] = -q[q < 0].sum()

    return dict(scd=scd, shd=shd, rg=rg, arom=arom, rgg=rgg, kappa=kap, entropy=ent, qpos=qpos, qneg=qneg)

def patch_pair_summaries(seq1_chars, seq2_chars, idx1, idx2, w=5, threshold=-1.0):
    C1 = patch_count_matrix(idx1, w)  # (P1,20)
    C2 = patch_count_matrix(idx2, w)  # (P2,20)
    P1, P2 = C1.shape[0], C2.shape[0]
    if P1 == 0 or P2 == 0:
        return {
            "ah_energy_mean": 0.0, "ah_energy_min": 0.0, "ah_energy_max": 0.0, "ah_energy_std": 0.0,
            "ah_energy_top5_mean": 0.0, "ah_energy_top10_mean": 0.0,
            "opp_charge_mean": 0.0, "opp_charge_min": 0.0, "opp_charge_max": 0.0, "opp_charge_std": 0.0,
            "opp_charge_top5_mean": 0.0, "opp_charge_top10_mean": 0.0,
            "top_patch_gap": 0.0, "count_strong_patches": 0, "corr_oppcharge_ah": 0.0,
            "scd_patch1_mean": 0.0, "scd_patch1_min": 0.0, "scd_patch1_max": 0.0, "scd_patch1_std": 0.0,
            "scd_patch2_mean": 0.0, "scd_patch2_min": 0.0, "scd_patch2_max": 0.0, "scd_patch2_std": 0.0,
            "shd_patch1_mean": 0.0, "shd_patch1_min": 0.0, "shd_patch1_max": 0.0, "shd_patch1_std": 0.0,
            "shd_patch2_mean": 0.0, "shd_patch2_min": 0.0, "shd_patch2_max": 0.0, "shd_patch2_std": 0.0,
            "rg_fraction_patch1_mean": 0.0, "rg_fraction_patch1_min": 0.0, "rg_fraction_patch1_max": 0.0, "rg_fraction_patch1_std": 0.0,
            "rg_fraction_patch2_mean": 0.0, "rg_fraction_patch2_min": 0.0, "rg_fraction_patch2_max": 0.0, "rg_fraction_patch2_std": 0.0,
            "aromatic_fraction_patch1_mean": 0.0, "aromatic_fraction_patch1_min": 0.0, "aromatic_fraction_patch1_max": 0.0, "aromatic_fraction_patch1_std": 0.0,
            "aromatic_fraction_patch2_mean": 0.0, "aromatic_fraction_patch2_min": 0.0, "aromatic_fraction_patch2_max": 0.0, "aromatic_fraction_patch2_std": 0.0,
            "rgg_count_patch1_mean": 0.0, "rgg_count_patch1_min": 0.0, "rgg_count_patch1_max": 0.0, "rgg_count_patch1_std": 0.0,
            "rgg_count_patch2_mean": 0.0, "rgg_count_patch2_min": 0.0, "rgg_count_patch2_max": 0.0, "rgg_count_patch2_std": 0.0,
            "entropy_patch1_mean": 0.0, "entropy_patch1_min": 0.0, "entropy_patch1_max": 0.0, "entropy_patch1_std": 0.0,
            "entropy_patch2_mean": 0.0, "entropy_patch2_min": 0.0, "entropy_patch2_max": 0.0, "entropy_patch2_std": 0.0,
            "kappa_patch1_mean": 0.0, "kappa_patch1_min": 0.0, "kappa_patch1_max": 0.0, "kappa_patch1_std": 0.0,
            "kappa_patch2_mean": 0.0, "kappa_patch2_min": 0.0, "kappa_patch2_max": 0.0, "kappa_patch2_std": 0.0,
        }

    # AH patch–patch energy matrix: E = (C1 @ AH @ C2.T) / (w*w)
    M = C1 @ AH
    E = (M @ C2.T) / (w * w)  # (P1,P2)

    # Opposite-charge broadcasted matrix
    feat1 = patch_scalar_features(seq1_chars, idx1, w)
    feat2 = patch_scalar_features(seq2_chars, idx2, w)
    oc = opp_charges_total(feat1["qpos"][:, None], feat1["qneg"][:, None],
                           feat2["qpos"][None, :], feat2["qneg"][None, :])  # (P1,P2)

    Ef = E.ravel()
    OCf = oc.ravel()

    def topk_mean(arr, k):
        if arr.size == 0: return 0.0
        k = min(k, arr.size)
        idx = np.argpartition(arr, -k)[-k:]
        return float(np.mean(arr[idx]))

    out = {}
    for key, arr in [("ah_energy", Ef), ("opp_charge", OCf)]:
        out[f"{key}_mean"] = float(np.mean(arr))
        out[f"{key}_min"]  = float(np.min(arr))
        out[f"{key}_max"]  = float(np.max(arr))
        out[f"{key}_std"]  = float(np.std(arr))
        out[f"{key}_top5_mean"]  = topk_mean(arr, 5)
        out[f"{key}_top10_mean"] = topk_mean(arr, 10)

    if Ef.size >= 2:
        top2 = np.partition(Ef, -2)[-2:]
        out["top_patch_gap"] = float(top2[-1] - top2[-2])
    else:
        out["top_patch_gap"] = 0.0

    out["count_strong_patches"] = int(np.sum(Ef < -1.0))
    if np.std(OCf) > 0 and np.std(Ef) > 0:
        out["corr_oppcharge_ah"] = float(np.corrcoef(OCf, Ef)[0, 1])
    else:
        out["corr_oppcharge_ah"] = 0.0

    def add_patch_stats(prefix, x):
        out[f"{prefix}_mean"] = float(np.mean(x)) if x.size else 0.0
        out[f"{prefix}_min"]  = float(np.min(x))  if x.size else 0.0
        out[f"{prefix}_max"]  = float(np.max(x))  if x.size else 0.0
        out[f"{prefix}_std"]  = float(np.std(x))  if x.size else 0.0

    for pref, vec in [("scd_patch1", feat1["scd"]), ("scd_patch2", feat2["scd"]),
                      ("shd_patch1", feat1["shd"]), ("shd_patch2", feat2["shd"]),
                      ("rg_fraction_patch1", feat1["rg"]), ("rg_fraction_patch2", feat2["rg"]),
                      ("aromatic_fraction_patch1", feat1["arom"]), ("aromatic_fraction_patch2", feat2["arom"]),
                      ("rgg_count_patch1", feat1["rgg"]), ("rgg_count_patch2", feat2["rgg"]),
                      ("entropy_patch1", feat1["entropy"]), ("entropy_patch2", feat2["entropy"]),
                      ("kappa_patch1", feat1["kappa"]), ("kappa_patch2", feat2["kappa"])]:
        add_patch_stats(pref, vec)

    return out

def analyse_features(names, residue_properties, sequences, patch_size=5, shard=0, nshards=1):
    results = []
    start = time.time()
    done = 0
    last = start
    print(f"[shard {shard}/{nshards}] started", flush=True)

    for i, name in enumerate(names):
        if (i % nshards) != shard:
            continue

        rgg, tad = name.split("__", 1)
        seq_r = sequences[rgg]; seq_t = sequences[tad]
        idx_r = encode_seq(seq_r); idx_t = encode_seq(seq_t)
        Lr, Lt = len(idx_r), len(idx_t)

        lamb_r, sig_r, q_r = per_res_props(idx_r)
        lamb_t, sig_t, q_t = per_res_props(idx_t)

        lambdas_rgg = float(lamb_r.sum()); lambdas_tad = float(lamb_t.sum())
        sigmas_rgg  = float(sig_r.sum());  sigmas_tad  = float(sig_t.sum())
        charges_rgg = float(q_r.sum());    charges_tad = float(q_t.sum())

        avg_lambda = ((lamb_r.mean() if Lr else 0.0) + (lamb_t.mean() if Lt else 0.0)) / 2.0
        avg_sigma  = (sigmas_rgg + sigmas_tad) / 2.0
        avg_charge = (charges_rgg + charges_tad) / 2.0

        ncpr_rgg = (charges_rgg / Lr) if Lr else 0.0
        ncpr_tad = (charges_tad / Lt) if Lt else 0.0
        avg_ncpr = (ncpr_rgg + ncpr_tad) / 2.0

        n_plus_i = float(q_r[q_r > 0].sum()); n_minus_i = float(-q_r[q_r < 0].sum())
        n_plus_j = float(q_t[q_t > 0].sum()); n_minus_j = float(-q_t[q_t < 0].sum())
        charge_comp = (n_plus_i * n_minus_j + n_minus_i * n_plus_j) - (n_plus_i * n_plus_j + n_minus_i * n_minus_j)
        opp_charge  = charge_comp

        average_interaction = calc_ah_between_sequences_idx(idx_r, idx_t)

        scd_rgg = calc_SCD(list(seq_r))
        scd_tad = calc_SCD(list(seq_t))
        avg_scd = (scd_rgg + scd_tad) / 2.0

        shd_rgg = calc_SHD_fast(idx_r)
        shd_tad = calc_SHD_fast(idx_t)
        avg_shd = (shd_rgg + shd_tad) / 2.0

        kappa_rgg = calc_kappa(list(seq_r))
        kappa_tad = calc_kappa(list(seq_t))
        avg_kappa = (kappa_rgg + kappa_tad) / 2.0

        arom_rgg = aromatic_fraction(list(seq_r))
        arom_tad = aromatic_fraction(list(seq_t))
        avg_arom = (arom_rgg + arom_tad) / 2.0

        patch_summary = patch_pair_summaries(list(seq_r), list(seq_t), idx_r, idx_t, w=patch_size, threshold=-1.0)

        feature_entry = {
            "name": name,
            "rgg": rgg, "RGG_length": Lr, "charge_RGG": charges_rgg, "lambda_RGG": lambdas_rgg,
            "NCPR_RGG": ncpr_rgg, "SCD_RGG": scd_rgg, "SHD_RGG": shd_rgg, "kappa_RGG": kappa_rgg,
            "tad": tad, "TAD_length": Lt, "charge_TAD": charges_tad, "lambda_TAD": lambdas_tad,
            "NCPR_TAD": ncpr_tad, "SCD_TAD": scd_tad, "SHD_TAD": shd_tad, "kappa_TAD": kappa_tad,
            "Average_lambda": avg_lambda,
            "Min_lambda": min(lamb_r.mean() if Lr else 0.0, lamb_t.mean() if Lt else 0.0),
            "Average_sigma": avg_sigma,
            "Average_charge": avg_charge,
            "Opposite_Charge_Number": opp_charge,
            "Average_NCPR": avg_ncpr,
            "Charge_Complementarity": charge_comp,
            "Total_Length": (Lr + Lt),
            "Average_Interaction": average_interaction,
            "Average_SCD": avg_scd,
            "Average_Aromatics": avg_arom,
            "Average_SHD": avg_shd,
            "Average_kappa": avg_kappa
        }
        feature_entry.update(patch_summary)
        results.append(feature_entry)

        done += 1
        # lightweight progress
        if (done % 1000 == 0) or (time.time() - last >= 30):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[shard {shard}/{nshards}] {done} pairs  rate={rate:.2f}/s  elapsed={elapsed/60:.1f} min",
                  flush=True)
            last = time.time()

    elapsed = time.time() - start
    print(f"[shard {shard}/{nshards}] DONE {done} pairs in {elapsed/60:.1f} min", flush=True)
    return results
def initialize(residue_csv):
    global residue_properties, aas, aa_to_idx, LAMB, SIGM, QVAL, AH
    residue_properties = pd.read_csv(residue_csv).set_index("one")
    aas = list(residue_properties.index)
    aa_to_idx = {aa: i for i, aa in enumerate(aas)}
    LAMB = residue_properties.loc[aas, "lambdas"].to_numpy(dtype=float)
    SIGM = residue_properties.loc[aas, "sigmas"].to_numpy(dtype=float)
    QVAL = residue_properties.loc[aas, "q"].to_numpy(dtype=float)
    AH = build_AH_matrix()
