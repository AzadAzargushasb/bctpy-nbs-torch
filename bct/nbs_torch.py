from __future__ import division, print_function

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .utils import BCTParamError, get_rng
from .due import due, BibTeX
from .citations import ZALESKY2010


def _resolve_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _auto_perm_chunk(m, k, dtype, device, peak_factor=12, budget_frac=0.5):
    """Estimate the largest K-axis chunk that keeps peak intermediates within budget.

    Peak working set is dominated by O(peak_factor) tensors of shape (m, chunk).
    On CUDA we read free device memory; on CPU we leave the chunk uncapped.
    """
    if device.type != 'cuda':
        return k
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    free, _ = torch.cuda.mem_get_info(device)
    bytes_per_chunk_unit = peak_factor * m * dtype_bytes
    chunk = max(1, int(free * budget_frac / bytes_per_chunk_unit))
    return min(chunk, k)


def _materialize(x, y, dtype, device):
    n = x.shape[0]
    ixes_np = np.where(np.triu(np.ones((n, n), dtype=bool), 1))
    xmat = torch.as_tensor(x[ixes_np[0], ixes_np[1], :], dtype=dtype, device=device)
    ymat = torch.as_tensor(y[ixes_np[0], ixes_np[1], :], dtype=dtype, device=device)
    return xmat, ymat, ixes_np


def _apply_tail(t, tail):
    if tail == 'both':
        return torch.abs(t)
    if tail == 'left':
        return -t
    return t


def _observed_t(xmat, ymat, tail, paired):
    P = xmat.shape[1]
    Q = ymat.shape[1]
    if paired:
        if P != Q:
            raise BCTParamError('Population matrices must be an equal size')
        E = xmat - ymat
        mE = E.mean(dim=1)
        vE = E.var(dim=1, unbiased=True)
        sd = torch.sqrt(vE)
        denom = sd / float(P) ** 0.5
        zero = denom == 0
        t = torch.where(zero, torch.zeros_like(mE), mE / torch.where(zero, torch.ones_like(denom), denom))
    else:
        mx = xmat.mean(dim=1)
        my = ymat.mean(dim=1)
        vx = xmat.var(dim=1, unbiased=True)
        vy = ymat.var(dim=1, unbiased=True)
        sp = torch.sqrt(((P - 1) * vx + (Q - 1) * vy) / (P + Q - 2))
        denom = sp * (1.0 / P + 1.0 / Q) ** 0.5
        zero = denom == 0
        t = torch.where(zero, torch.zeros_like(mx),
                        (mx - my) / torch.where(zero, torch.ones_like(denom), denom))
    return _apply_tail(t, tail)


def _fast_get_components(adj):
    n = adj.shape[0]
    A = (adj != 0).astype(np.int8)
    np.fill_diagonal(A, 1)
    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    return labels + 1, sizes


def _component_edge_sizes_and_label(adj, labels_1based, sizes):
    """Return (sz_links, labelled_adj) following nbs.py lines 183-195.

    Components with size <= 1 (isolated nodes) are dropped from sz_links and
    cleared from the labelled adjacency, matching the reference behaviour.
    """
    ind_sz = np.where(sizes > 1)[0] + 1
    nr = ind_sz.size
    sz_links = np.zeros(nr)
    out_adj = adj.copy()
    for i in range(nr):
        nodes = np.where(ind_sz[i] == labels_1based)[0]
        sub = out_adj[np.ix_(nodes, nodes)]
        sz_links[i] = sub.sum() / 2
        out_adj[np.ix_(nodes, nodes)] *= (i + 2)
    out_adj[np.where(out_adj)] -= 1
    return sz_links, out_adj


def _make_perm_masks(P, Q, K, generator, device, dtype):
    """Unpaired: K random permutation masks of shape (K, P+Q), each row picking P 'group X' columns."""
    noise = torch.rand((K, P + Q), generator=generator, device=device)
    order = noise.argsort(dim=1)
    masks_x = torch.zeros((K, P + Q), dtype=dtype, device=device)
    rows = torch.arange(K, device=device).unsqueeze(1).expand(K, P)
    masks_x[rows, order[:, :P]] = 1.0
    return masks_x


def _make_sign_matrix(P, K, generator, device, dtype):
    """Paired: K random sign vectors of shape (K, P) with entries in {-1, +1}."""
    bits = torch.randint(0, 2, (K, P), generator=generator, device=device, dtype=torch.int8)
    return bits.to(dtype) * 2.0 - 1.0


def _batched_perm_t_unpaired(D, masks_x, P, Q, tail):
    """All K permuted two-sample pooled-variance t-stats via 4 GEMMs.

    D: (m, P+Q), masks_x: (K, P+Q). Returns T of shape (m, K).
    """
    masks_y = 1.0 - masks_x
    D2 = D * D
    Wx = masks_x.transpose(0, 1).contiguous()
    Wy = masks_y.transpose(0, 1).contiguous()
    Sx = D @ Wx
    Sy = D @ Wy
    Qx = D2 @ Wx
    Qy = D2 @ Wy
    mx = Sx / P
    my = Sy / Q
    vx = (Qx - P * mx * mx) / (P - 1)
    vy = (Qy - Q * my * my) / (Q - 1)
    sp = torch.sqrt(((P - 1) * vx + (Q - 1) * vy) / (P + Q - 2))
    denom = sp * (1.0 / P + 1.0 / Q) ** 0.5
    zero = denom == 0
    t = torch.where(zero, torch.zeros_like(mx),
                    (mx - my) / torch.where(zero, torch.ones_like(denom), denom))
    return _apply_tail(t, tail)


def _batched_perm_t_paired(E, S, P, tail):
    """All K paired t-stats. E: (m, P) = xmat - ymat. S: (K, P) sign matrix.

    Per perm u, E^(u) = E * s^(u). Mean = (E s) / P. Var via second moment.
    Returns T of shape (m, K).
    """
    Sm = S.transpose(0, 1).contiguous()
    sum_E = E @ Sm
    mean_E = sum_E / P
    E2 = E * E
    sumsq_E = E2 @ (Sm * Sm)
    var_E = (sumsq_E - P * mean_E * mean_E) / (P - 1)
    sd = torch.sqrt(var_E.clamp(min=0))
    denom = sd / float(P) ** 0.5
    zero = denom == 0
    t = torch.where(zero, torch.zeros_like(mean_E),
                    mean_E / torch.where(zero, torch.ones_like(denom), denom))
    return _apply_tail(t, tail)


def _per_perm_max_component_size(super_mask, ixes_np, n):
    """Iterate over K permutations on host, scatter to NxN adj, run scipy CC,
    record max component size in edges. super_mask: bool array (m, K) on CPU."""
    K = super_mask.shape[1]
    null = np.zeros(K)
    rows, cols = ixes_np
    for u in range(K):
        col = super_mask[:, u]
        sel = np.where(col)[0]
        if sel.size == 0:
            null[u] = 0
            continue
        adj = np.zeros((n, n))
        adj[rows[sel], cols[sel]] = 1
        adj = adj + adj.T
        labels_1, sizes = _fast_get_components(adj)
        ind_sz = np.where(sizes > 1)[0] + 1
        if ind_sz.size == 0:
            null[u] = 0
            continue
        max_e = 0.0
        for i in range(ind_sz.size):
            nodes = np.where(ind_sz[i] == labels_1)[0]
            e = adj[np.ix_(nodes, nodes)].sum() / 2
            if e > max_e:
                max_e = e
        null[u] = max_e
    return null


@due.dcite(BibTeX(ZALESKY2010), description="Network-based statistic (torch)")
def nbs_bct(x, y, thresh, k=1000, tail='both', paired=False, verbose=False,
            seed=None, device=None, dtype=torch.float32, max_perm_chunk=None):
    """Torch-based Network-Based Statistic. Drop-in replacement for bct.nbs.nbs_bct.

    Additional kwargs:
      device: 'cpu', 'cuda', or torch.device. Defaults to cuda if available.
      dtype: torch dtype for arithmetic. Default float32.
      max_perm_chunk: cap on the K dimension per GEMM batch. None = run all at once.
    """
    if tail not in ('both', 'left', 'right'):
        raise BCTParamError('Tail must be both, left, right')

    ix, jx, nx = x.shape
    iy, jy, ny = y.shape
    if not ix == jx == iy == jy:
        raise BCTParamError('Population matrices are of inconsistent size')
    n = ix
    if paired and nx != ny:
        raise BCTParamError('Population matrices must be an equal size')

    device = _resolve_device(device)

    rng_np = get_rng(seed)
    g = torch.Generator(device=device)
    g.manual_seed(int(rng_np.randint(0, 2**31 - 1)))

    with torch.no_grad():
        xmat, ymat, ixes_np = _materialize(x, y, dtype, device)
        m = xmat.shape[0]

        if max_perm_chunk is None:
            max_perm_chunk = _auto_perm_chunk(m, k, dtype, device)

        # Phase 2: observed t
        t_obs = _observed_t(xmat, ymat, tail, paired)

        # Phase 3: observed threshold + components
        super_obs = (t_obs > thresh).cpu().numpy()
        ind_t = np.where(super_obs)[0]
        if ind_t.size == 0:
            raise BCTParamError("Unsuitable threshold")
        adj = np.zeros((n, n))
        adj[ixes_np[0][ind_t], ixes_np[1][ind_t]] = 1
        adj = adj + adj.T
        labels_1, sizes = _fast_get_components(adj)
        sz_links, adj_labelled = _component_edge_sizes_and_label(adj, labels_1, sizes)

        if sz_links.size == 0:
            raise BCTParamError('True matrix is degenerate')
        max_sz = sz_links.max()
        nr_components = sz_links.size
        if verbose:
            print('max component size is %i' % int(max_sz))
            print('estimating null distribution with %i permutations' % k)

        # Phase 4: permutations
        if paired:
            E = xmat - ymat
            full_K = k
            chunk = max_perm_chunk or full_K
            null = np.zeros(full_K)
            done = 0
            while done < full_K:
                this_K = min(chunk, full_K - done)
                S = _make_sign_matrix(nx, this_K, g, device, dtype)
                T = _batched_perm_t_paired(E, S, nx, tail)
                super_mask = (T > thresh).cpu().numpy()
                null[done:done + this_K] = _per_perm_max_component_size(super_mask, ixes_np, n)
                done += this_K
        else:
            D = torch.cat([xmat, ymat], dim=1)
            full_K = k
            chunk = max_perm_chunk or full_K
            null = np.zeros(full_K)
            done = 0
            while done < full_K:
                this_K = min(chunk, full_K - done)
                masks_x = _make_perm_masks(nx, ny, this_K, g, device, dtype)
                T = _batched_perm_t_unpaired(D, masks_x, nx, ny, tail)
                super_mask = (T > thresh).cpu().numpy()
                null[done:done + this_K] = _per_perm_max_component_size(super_mask, ixes_np, n)
                done += this_K

        # Phase 5: p-values
        pvals = np.zeros(nr_components)
        for i in range(nr_components):
            pvals[i] = (null >= sz_links[i]).sum() / k

    return pvals, adj_labelled, null
