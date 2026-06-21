# ═══════════════════════════════════════════════════════════════════════════════
# Richer mRNA-structure features — do they classify (especially LOW)?
# ═══════════════════════════════════════════════════════════════════════════════
#
# Your scalar dG (dG_cds_only, dG_cds_16_55) has ordinal signal but doesn't
# classify, because a single number collapses the whole folding landscape.
# This tests POSITION-RESOLVED structure that the scalar throws away:
#
#   A) Windowed local-dG profile : slide a window along the CDS, fold each
#      window → a VECTOR of local folding energies (where structure sits).
#   B) Start-codon-region dG     : fold just the first N nt (initiation region,
#      the canonical LOW-expression driver).
#   C) Per-position pairing prob : ViennaRNA partition function → probability
#      each base is paired (a "structuredness" profile, richer than MFE).
#
# Each feature set → RF on the SAME 5 folds. Reports overall accuracy AND
# Low-bin recall (the ~48% wall we're trying to break). Scalar dG included as
# the baseline to beat.
#
# CAVEAT: CDS only (no 5' UTR). The strongest initiation structure often lives
# in the UTR, so a null result here may mean "structure is in the UTR we can't
# see", not "structure doesn't drive Low". Interpret accordingly.
#
# Structure features are cached to mrna_struct_cache.npz (folding ~14k seqs
# takes several minutes; cached after the first run).
# ═══════════════════════════════════════════════════════════════════════════════

!pip install ViennaRNA scikit-learn openpyxl --quiet

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')

try:
    import RNA
    HAVE_RNA = True
except Exception:
    HAVE_RNA = False
    print("*** ViennaRNA not importable — install with: pip install ViennaRNA ***")

# ── Config ────────────────────────────────────────────────────────────────────
BIN_NAMES = ['low', 'medium', 'high']
NUM_BINS  = 3
XLSX_PATH = '/content/drive/MyDrive/data/sequence-features-by-bin.xlsx'  # ← set me
CACHE_DIR = '.'
BIN_COL, SEQ_COL = 'bin', 'cds'
N_FOLDS, CV_SEED = 5, 42
CDS_FRAME_OFFSET = 2
BIN_LABEL_MAP = {'low': 0, 'medium': 1, 'high': 2}

# Feature parameters
WINDOW_SIZE   = 40      # nt per local-dG window
WINDOW_STEP   = 10      # slide step → profile length ≈ (L-40)/10
N_WINDOWS     = 30      # fixed profile length (pad/truncate to this)
START_REGION  = 45      # first 45 nt (~15 codons) = initiation region
PAIR_BINS     = 30      # downsample per-position pairing prob to this many bins


# ── Load sequences ────────────────────────────────────────────────────────────
df = pd.read_excel(XLSX_PATH)
seqs, y = [], []
for _, r in df.iterrows():
    dna = str(r[SEQ_COL]).upper().replace('U', 'T').strip()
    dna = dna[CDS_FRAME_OFFSET:]                 # trim to ATG start
    if len(dna) >= START_REGION:
        seqs.append(dna)
        y.append(BIN_LABEL_MAP[str(r[BIN_COL]).lower().strip()])
y = np.array(y, dtype=np.int32)
N = len(seqs)
print(f"Sequences: {N}   bins: {np.bincount(y).tolist()}")


# ── Structure feature computation (cached) ────────────────────────────────────
def windowed_dg(dna):
    """Vector of local MFE over sliding windows; fixed length N_WINDOWS."""
    prof = []
    for i in range(0, len(dna) - WINDOW_SIZE + 1, WINDOW_STEP):
        _, mfe = RNA.fold(dna[i:i+WINDOW_SIZE])
        prof.append(mfe)
    prof = prof[:N_WINDOWS] + [0.0] * max(0, N_WINDOWS - len(prof))
    return prof[:N_WINDOWS]

def start_region_dg(dna):
    """MFE of just the initiation region + a couple of summary stats."""
    sub = dna[:START_REGION]
    _, mfe = RNA.fold(sub)
    # also the single strongest local stem in the region (min window)
    locals_ = []
    for i in range(0, len(sub) - 20 + 1, 5):
        _, m = RNA.fold(sub[i:i+20]); locals_.append(m)
    return [mfe, min(locals_) if locals_ else 0.0, np.mean(locals_) if locals_ else 0.0]

def pairing_profile(dna):
    """Per-position probability of being paired (partition function),
    downsampled to PAIR_BINS values. Vectorized sum over the bpp matrix."""
    fc = RNA.fold_compound(dna)
    fc.pf()
    bpp = np.array(fc.bpp())          # (L+1, L+1), 1-indexed, upper triangular
    # p_paired[i] = sum of row i + col i (each pair counted once in the triangle)
    p_paired = (bpp.sum(axis=0) + bpp.sum(axis=1))[1:]   # drop unused index 0
    p_paired = np.minimum(p_paired, 1.0)
    L = len(p_paired)
    idx = np.linspace(0, L, PAIR_BINS+1).astype(int)
    return [p_paired[idx[k]:idx[k+1]].mean() if idx[k+1] > idx[k] else 0.0
            for k in range(PAIR_BINS)]

COMPUTE_PAIRING = False  # True adds the partition-function pairing profile
                         # (~90 min for 14k seqs). Windowed dG already captures
                         # position-resolved structure; turn on only if you
                         # specifically want pairing probabilities. ~6 min when False.

def compute_features():
    cache = os.path.join(CACHE_DIR, 'mrna_struct_cache.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"Loaded structure cache: windowed {d['F_win'].shape}")
        return d['F_win'], d['F_start'], d['F_pair']
    if not HAVE_RNA:
        raise RuntimeError("ViennaRNA required to compute features (no cache present)")
    import time
    print(f"Folding {N} sequences (pairing={'on' if COMPUTE_PAIRING else 'OFF'})...")
    Fw, Fs, Fp = [], [], []
    t0 = time.time()
    for k, dna in enumerate(seqs):
        if k % 250 == 0 and k > 0:
            rate = k / (time.time() - t0)
            eta = (N - k) / rate / 60
            print(f"  {k}/{N}  ({rate:.0f} seq/s, ~{eta:.1f} min left)   ", end='\r')
        Fw.append(windowed_dg(dna))
        Fs.append(start_region_dg(dna))
        Fp.append(pairing_profile(dna) if COMPUTE_PAIRING else [0.0]*PAIR_BINS)
    print(f"\n  {N}/{N} done in {(time.time()-t0)/60:.1f} min.    ")
    F_win   = np.array(Fw, dtype=np.float32)
    F_start = np.array(Fs, dtype=np.float32)
    F_pair  = np.array(Fp, dtype=np.float32)
    np.savez_compressed(cache, F_win=F_win, F_start=F_start, F_pair=F_pair)
    print(f"Cached → {cache}")
    return F_win, F_start, F_pair


F_win, F_start, F_pair = compute_features()

# Scalar dG baseline from the xlsx (the feature that has ordinal but no class signal)
SCALAR_COLS = [c for c in ['dG_cds_only', 'dG_cds_16_55'] if c in df.columns]
# align scalar rows to the sequences we kept (same filter order)
keep_mask = []
ki = 0
for _, r in df.iterrows():
    dna = str(r[SEQ_COL]).upper().replace('U','T').strip()[CDS_FRAME_OFFSET:]
    keep_mask.append(len(dna) >= START_REGION)
F_scalar = df.loc[keep_mask, SCALAR_COLS].to_numpy(dtype=np.float32)

FEATURES = {
    'Scalar dG (baseline)':  F_scalar,
    'Windowed dG profile':   F_win,
    'Start-region dG':       F_start,
}
if COMPUTE_PAIRING:
    FEATURES['Pairing profile'] = F_pair
    FEATURES['All structure']   = np.hstack([F_win, F_start, F_pair])
else:
    FEATURES['Windowed + start'] = np.hstack([F_win, F_start])
for k, v in FEATURES.items():
    print(f"  {k:22s}: {v.shape}")


# ── 5-fold RF: overall acc + LOW recall ───────────────────────────────────────
def folds_idx(y, n, seed):
    rng = np.random.default_rng(seed); per = {}
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); per[c] = np.array_split(idx, n)
    return [np.sort(np.concatenate([per[c][f] for c in per])) for f in range(n)]
FOLDS = folds_idx(y, N_FOLDS, CV_SEED)

def evaluate(F):
    accs, low_rec, med_rec, high_rec = [], [], [], []
    for f in range(N_FOLDS):
        te = FOLDS[f]; tr = np.setdiff1d(np.arange(N), te)
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=CV_SEED, n_jobs=-1)
        clf.fit(F[tr], y[tr]); p = clf.predict(F[te])
        accs.append((p == y[te]).mean())
        rec = recall_score(y[te], p, labels=[0,1,2], average=None, zero_division=0)
        low_rec.append(rec[0]); med_rec.append(rec[1]); high_rec.append(rec[2])
    return (np.mean(accs), np.mean(low_rec), np.mean(med_rec), np.mean(high_rec))

print("\n" + "="*72)
print("RF on each feature set — overall acc + per-bin recall (5-fold)")
print("="*72)
print(f"  {'Feature set':22s} {'Acc':>7s} {'LOW':>8s} {'Med':>7s} {'High':>7s}")
print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
base_low = None
for name, F in FEATURES.items():
    acc, lo, me, hi = evaluate(F)
    if name.startswith('Scalar'): base_low = lo
    flag = ''
    if base_low is not None and lo > base_low + 0.03 and not name.startswith('Scalar'):
        flag = '  ← LOW improved'
    print(f"  {name:22s} {acc:>7.1%} {lo:>8.1%} {me:>7.1%} {hi:>7.1%}{flag}")

print("\n  Reference: KNN codon identity ≈ 68%, Low recall ≈ 48% (the wall).")
print("  → If any structure feature pushes LOW recall well above ~48%,")
print("    position-resolved mRNA structure carries the missing Low signal.")
print("  → If LOW stays ~48%, the Low signal isn't in CDS structure")
print("    (may be in the 5' UTR, which isn't in this data).")
