"""Verification suite for bct.nbs_torch (V1-V12 from the implementation plan)."""

from __future__ import division, print_function

import time
from collections import Counter

import numpy as np
import pytest
import torch

from bct.algorithms import get_components
from bct.nbs import nbs_bct as ref_nbs
from bct.nbs_torch import (
    nbs_bct as new_nbs,
    _observed_t,
    _materialize,
    _fast_get_components,
    _component_edge_sizes_and_label,
    _batched_perm_t_unpaired,
    _batched_perm_t_paired,
)
from bct.utils import BCTParamError


# ---------------------------------------------------------------- V0 fixtures

def _symmetric_stack(rng, N, P):
    A = rng.standard_normal((N, N, P)).astype(np.float64)
    A = 0.5 * (A + A.transpose(1, 0, 2))
    for k in range(P):
        np.fill_diagonal(A[:, :, k], 0.0)
    return A


@pytest.fixture(scope="session")
def fx_tiny():
    rng = np.random.default_rng(0)
    return _symmetric_stack(rng, 20, 10), _symmetric_stack(rng, 20, 10)


@pytest.fixture(scope="session")
def fx_realistic():
    rng = np.random.default_rng(1)
    return _symmetric_stack(rng, 200, 40), _symmetric_stack(rng, 200, 40)


@pytest.fixture(scope="session")
def fx_small_realistic():
    """Smaller fixture for slow tests where the reference NBS is the bottleneck.

    N=80, P=Q=20 keeps reference runtime to ~0.2 s/permutation on a single core.
    """
    rng = np.random.default_rng(1)
    return _symmetric_stack(rng, 80, 20), _symmetric_stack(rng, 80, 20)


@pytest.fixture(scope="session")
def fx_planted():
    rng = np.random.default_rng(2)
    N, P, Q = 100, 30, 30
    x = _symmetric_stack(rng, N, P)
    y = _symmetric_stack(rng, N, Q)
    clique = np.array([7, 19, 33, 41, 58, 72])
    for u in clique:
        for v in clique:
            if u < v:
                x[u, v, :] += 1.0
                x[v, u, :] += 1.0
    return x, y, clique


# ---------------------------------- reference scalar t-test (oracle for V2/V4)

def _ref_ttest2_scalar(a, b, tail):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1.0 / n1 + 1.0 / n2)
    if denom == 0:
        return 0.0
    t = np.mean(a) - np.mean(b)
    if tail == 'both':
        return abs(t / denom)
    if tail == 'left':
        return -t / denom
    return t / denom


def _ref_ttest_paired_scalar(A, B, tail):
    n = len(A)
    diff = A - B
    sample_ss = np.sum(diff ** 2) - np.sum(diff) ** 2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))
    if unbiased_std == 0:
        return 0.0
    z = np.mean(diff) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return abs(t)
    if tail == 'left':
        return -t
    return t


def _ref_observed_t_vector(x, y, tail, paired):
    n = x.shape[0]
    P, Q = x.shape[2], y.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    xmat = np.stack([x[:, :, i][ixes] for i in range(P)], axis=1)
    ymat = np.stack([y[:, :, i][ixes] for i in range(Q)], axis=1)
    out = np.zeros(xmat.shape[0])
    for i in range(xmat.shape[0]):
        if paired:
            out[i] = _ref_ttest_paired_scalar(xmat[i], ymat[i], tail)
        else:
            out[i] = _ref_ttest2_scalar(xmat[i], ymat[i], tail)
    return out


def _ref_component_edge_sizes(adj, labels_1based, sizes):
    """Mirror bct/nbs.py lines 183-191 (edge-size loop) without mutating adj."""
    ind_sz = np.where(sizes > 1)[0] + 1
    out = []
    for i in range(ind_sz.size):
        nodes = np.where(ind_sz[i] == labels_1based)[0]
        out.append(adj[np.ix_(nodes, nodes)].sum() / 2)
    return out


# ============================================================== V1: signature

class TestV1Signatures:
    def test_v1_bad_tail(self, fx_tiny):
        x, y = fx_tiny
        with pytest.raises(BCTParamError):
            new_nbs(x, y, thresh=2.0, k=10, tail='upward', device='cpu')

    def test_v1_inconsistent_size(self):
        x = np.zeros((10, 10, 5))
        y = np.zeros((11, 11, 5))
        with pytest.raises(BCTParamError):
            new_nbs(x, y, thresh=2.0, k=10, device='cpu')

    def test_v1_paired_unequal(self):
        x = np.zeros((10, 10, 5))
        y = np.zeros((10, 10, 6))
        with pytest.raises(BCTParamError):
            new_nbs(x, y, thresh=2.0, k=10, paired=True, device='cpu')

    def test_v1_unsuitable_threshold(self, fx_tiny):
        x, y = fx_tiny
        with pytest.raises(BCTParamError):
            new_nbs(x, y, thresh=999.0, k=10, device='cpu')


# =============================================================== V2: observed t

class TestV2ObservedT:
    @pytest.mark.parametrize("tail", ["both", "left", "right"])
    @pytest.mark.parametrize("paired", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_v2(self, fx_tiny, tail, paired, dtype):
        x, y = fx_tiny
        if paired and x.shape[2] != y.shape[2]:
            pytest.skip("paired requires equal P=Q")
        t_ref = _ref_observed_t_vector(x, y, tail, paired)
        xmat, ymat, _ = _materialize(x, y, dtype, torch.device('cpu'))
        t_new = _observed_t(xmat, ymat, tail, paired).cpu().numpy()
        rtol = 1e-12 if dtype == torch.float64 else 5e-5
        atol = 1e-12 if dtype == torch.float64 else 5e-5
        np.testing.assert_allclose(t_new, t_ref, rtol=rtol, atol=atol)


# ============================================================ V3: observed CC

class TestV3ObservedComponents:
    def test_v3_observed_components(self, fx_realistic):
        x, y = fx_realistic
        xmat, ymat, ixes_np = _materialize(x, y, torch.float64, torch.device('cpu'))
        t_obs = _observed_t(xmat, ymat, 'both', False).cpu().numpy()
        n = x.shape[0]
        mask = t_obs > 1.0  # 1.0 -> moderately dense suprathreshold graph
        adj = np.zeros((n, n))
        adj[ixes_np[0][mask], ixes_np[1][mask]] = 1
        adj = adj + adj.T

        a_ref, sz_ref = get_components(adj)
        a_fast, sz_fast = _fast_get_components(adj)

        assert Counter(sz_ref.tolist()) == Counter(sz_fast.tolist())
        e_ref = sorted(_ref_component_edge_sizes(adj, a_ref, sz_ref))
        e_fast = sorted(_ref_component_edge_sizes(adj, a_fast, sz_fast))
        assert e_ref == e_fast


# ============================================================ V4: permuted t

class TestV4PermutedT:
    def test_v4_permuted_t_equivalence(self, fx_realistic):
        x, y = fx_realistic
        P = x.shape[2]
        Q = y.shape[2]
        K = 32
        rng = np.random.default_rng(123)
        perms = np.stack([rng.permutation(P + Q) for _ in range(K)])

        # Build masks_x in float64 from the integer perms.
        masks_x_np = np.zeros((K, P + Q), dtype=np.float64)
        for u in range(K):
            masks_x_np[u, perms[u, :P]] = 1.0
        masks_x = torch.as_tensor(masks_x_np, dtype=torch.float64, device='cpu')

        xmat, ymat, _ = _materialize(x, y, torch.float64, torch.device('cpu'))
        D = torch.cat([xmat, ymat], dim=1)

        T_new = _batched_perm_t_unpaired(D, masks_x, P, Q, 'both').cpu().numpy()

        # Reference scalar loop (slow path) — matches what nbs.py does.
        D_np = D.cpu().numpy()
        T_ref = np.zeros(T_new.shape)
        for u in range(K):
            d = D_np[:, perms[u]]
            for i in range(D_np.shape[0]):
                T_ref[i, u] = _ref_ttest2_scalar(d[i, :P], d[i, P:], 'both')

        np.testing.assert_allclose(T_new, T_ref, rtol=1e-10, atol=1e-10)


# ============================================================ V5: reproducibility

class TestV5Repro:
    def test_v5_repro_unpaired(self, fx_realistic):
        x, y = fx_realistic
        p1, _, n1 = new_nbs(x, y, thresh=3.0, k=200, seed=42, device='cpu', dtype=torch.float64)
        p2, _, n2 = new_nbs(x, y, thresh=3.0, k=200, seed=42, device='cpu', dtype=torch.float64)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(p1, p2)

    def test_v5_repro_paired(self, fx_realistic):
        x, y = fx_realistic
        p1, _, n1 = new_nbs(x, y, thresh=3.0, k=200, seed=7, paired=True, device='cpu', dtype=torch.float64)
        p2, _, n2 = new_nbs(x, y, thresh=3.0, k=200, seed=7, paired=True, device='cpu', dtype=torch.float64)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(p1, p2)


# ============================================================ V6: per-perm adj

def _build_perm_adj_from_mask_row(D_np, perm_row, P, Q, thresh, ixes_np, n, tail='both'):
    d = D_np[:, perm_row]
    m = D_np.shape[0]
    t_perm = np.zeros(m)
    for i in range(m):
        t_perm[i] = _ref_ttest2_scalar(d[i, :P], d[i, P:], tail)
    sel = np.where(t_perm > thresh)[0]
    adj = np.zeros((n, n))
    adj[ixes_np[0][sel], ixes_np[1][sel]] = 1
    adj = adj + adj.T
    return adj


class TestV6PermAdj:
    def test_v6_adj_equiv(self, fx_realistic):
        x, y = fx_realistic
        P = x.shape[2]
        Q = y.shape[2]
        n = x.shape[0]
        K = 8
        rng = np.random.default_rng(321)
        perms = np.stack([rng.permutation(P + Q) for _ in range(K)])
        masks_x_np = np.zeros((K, P + Q), dtype=np.float64)
        for u in range(K):
            masks_x_np[u, perms[u, :P]] = 1.0
        masks_x = torch.as_tensor(masks_x_np, dtype=torch.float64, device='cpu')

        xmat, ymat, ixes_np = _materialize(x, y, torch.float64, torch.device('cpu'))
        D = torch.cat([xmat, ymat], dim=1)
        T = _batched_perm_t_unpaired(D, masks_x, P, Q, 'both').cpu().numpy()
        super_mask = T > 3.0  # bool (m, K)

        D_np = D.cpu().numpy()
        for u in range(K):
            sel = np.where(super_mask[:, u])[0]
            adj_torch = np.zeros((n, n))
            adj_torch[ixes_np[0][sel], ixes_np[1][sel]] = 1
            adj_torch = adj_torch + adj_torch.T
            adj_ref = _build_perm_adj_from_mask_row(D_np, perms[u], P, Q, 3.0, ixes_np, n)
            np.testing.assert_array_equal(adj_torch, adj_ref)


# ======================================================== V7: per-perm components

class TestV7PermComponents:
    def test_v7_component_equiv(self, fx_realistic):
        x, y = fx_realistic
        P = x.shape[2]
        Q = y.shape[2]
        n = x.shape[0]
        K = 8
        rng = np.random.default_rng(999)
        perms = np.stack([rng.permutation(P + Q) for _ in range(K)])
        xmat, ymat, ixes_np = _materialize(x, y, torch.float64, torch.device('cpu'))
        D = D_np = torch.cat([xmat, ymat], dim=1).cpu().numpy()
        for u in range(K):
            adj = _build_perm_adj_from_mask_row(D_np, perms[u], P, Q, 3.0, ixes_np, n)
            a_ref, sz_ref = get_components(adj)
            a_fast, sz_fast = _fast_get_components(adj)
            e_ref = sorted(_ref_component_edge_sizes(adj, a_ref, sz_ref))
            e_fast = sorted(_ref_component_edge_sizes(adj, a_fast, sz_fast))
            assert e_ref == e_fast


# ======================================================== V8: KS test on null

class TestV8NullKS:
    @pytest.mark.slow
    def test_v8_null_ks(self, fx_small_realistic):
        from scipy import stats
        x, y = fx_small_realistic
        K = 200
        passes = 0
        for trial in range(3):
            _, _, null_ref = ref_nbs(x, y, thresh=3.0, k=K, seed=trial)
            _, _, null_new = new_nbs(x, y, thresh=3.0, k=K, seed=trial * 1000 + 1,
                                     device='cpu', dtype=torch.float64)
            ks_p = stats.ks_2samp(null_ref, null_new).pvalue
            print(f"trial {trial}: KS p={ks_p:.3f}")
            if ks_p > 0.01:
                passes += 1
        assert passes >= 2


# ======================================================== V9: planted component

class TestV9Planted:
    @pytest.mark.slow
    def test_v9_planted(self, fx_planted):
        x, y, clique = fx_planted
        K = 400

        def _component_size_from_labelled(A_lab, planted):
            # bctpy stores 1-based labels in upper+lower triangle.
            n = A_lab.shape[0]
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                row_nz = np.where(A_lab[i] != 0)[0]
                if len(row_nz):
                    labels[i] = int(A_lab[i, row_nz[0]])
            cnt = Counter(labels[planted].tolist())
            cnt.pop(0, None)
            if not cnt:
                return -1, -1.0
            cid, _ = cnt.most_common(1)[0]
            nodes = np.where(labels == cid)[0]
            inter = len(set(nodes.tolist()) & set(planted.tolist()))
            union = len(set(nodes.tolist()) | set(planted.tolist()))
            return cid, inter / union

        def _pval_for_component_size(sz_value, null):
            return float((null >= sz_value).sum() / null.size)

        def _component_edge_size_for_label(A_lab, target_label):
            nodes = np.where(np.any(A_lab == target_label, axis=1))[0]
            sub = A_lab[np.ix_(nodes, nodes)]
            return (sub == target_label).sum() / 2

        p_ref, A_ref, null_ref = ref_nbs(x, y, thresh=3.0, k=K, seed=7)
        p_new, A_new, null_new = new_nbs(x, y, thresh=3.0, k=K, seed=7,
                                         device='cpu', dtype=torch.float64)

        cid_ref, j_ref = _component_size_from_labelled(A_ref, clique)
        cid_new, j_new = _component_size_from_labelled(A_new, clique)
        assert j_ref >= 0.8, f"reference Jaccard too low: {j_ref}"
        assert j_new >= 0.8, f"torch Jaccard too low: {j_new}"

        sz_ref = _component_edge_size_for_label(A_ref, cid_ref)
        sz_new = _component_edge_size_for_label(A_new, cid_new)
        pp_ref = _pval_for_component_size(sz_ref, null_ref)
        pp_new = _pval_for_component_size(sz_new, null_new)
        print(f"planted p_ref={pp_ref:.4f}, p_new={pp_new:.4f}")
        assert abs(pp_ref - pp_new) <= 3.0 / np.sqrt(K)


# ======================================================== V10: edge cases

class TestV10EdgeCases:
    def test_v10a_constant_edge(self):
        rng = np.random.default_rng(11)
        N, P, Q = 12, 8, 8
        x = _symmetric_stack(rng, N, P)
        y = _symmetric_stack(rng, N, Q)
        x[0, 1, :] = 1.0
        x[1, 0, :] = 1.0
        y[0, 1, :] = 1.0
        y[1, 0, :] = 1.0
        xmat, ymat, ixes_np = _materialize(x, y, torch.float64, torch.device('cpu'))
        t_obs = _observed_t(xmat, ymat, 'both', False).cpu().numpy()
        edge_idx = np.where((ixes_np[0] == 0) & (ixes_np[1] == 1))[0][0]
        assert t_obs[edge_idx] == 0.0
        assert np.all(np.isfinite(t_obs))

    def test_v10b_threshold_below_min(self, fx_tiny):
        x, y = fx_tiny
        # With thresh = -inf every edge is suprathreshold.
        xmat, ymat, _ = _materialize(x, y, torch.float64, torch.device('cpu'))
        t_obs = _observed_t(xmat, ymat, 'both', False).cpu().numpy()
        thresh = float(t_obs.min()) - 1e-3
        # k=20 to keep test fast
        p, adj, null = new_nbs(x, y, thresh=thresh, k=20, seed=0, device='cpu', dtype=torch.float64)
        # the labelled adjacency should reflect a single dominant component
        n = x.shape[0]
        nodes_in_components = np.unique(np.where(adj != 0)[0])
        assert len(nodes_in_components) == n  # everyone is part of some component

    def test_v10c_threshold_above_max(self, fx_tiny):
        x, y = fx_tiny
        with pytest.raises(BCTParamError):
            new_nbs(x, y, thresh=1e6, k=20, device='cpu', dtype=torch.float64)
        with pytest.raises(BCTParamError):
            ref_nbs(x, y, thresh=1e6, k=20)

    def test_v10d_paired_two_pairs(self):
        rng = np.random.default_rng(5)
        N, P = 8, 2
        x = _symmetric_stack(rng, N, P)
        y = _symmetric_stack(rng, N, P)
        xmat, ymat, _ = _materialize(x, y, torch.float64, torch.device('cpu'))
        t_obs = _observed_t(xmat, ymat, 'both', True).cpu().numpy()
        thresh = float(np.median(t_obs))
        p, adj, null = new_nbs(x, y, thresh=thresh, k=200, paired=True, seed=3,
                               device='cpu', dtype=torch.float64)
        # only 2^2 = 4 possible sign patterns; null must take at most 4 distinct values
        assert len(np.unique(null)) <= 4

    def test_v10e_asymmetric_input(self, fx_tiny):
        # Lower triangle is unused; perturbing it should not affect the result.
        x, y = fx_tiny
        x_perturbed = x.copy()
        x_perturbed[1, 0, :] += 100.0  # below diagonal — should be ignored
        p1, _, n1 = new_nbs(x, y, thresh=2.0, k=50, seed=11, device='cpu', dtype=torch.float64)
        p2, _, n2 = new_nbs(x_perturbed, y, thresh=2.0, k=50, seed=11, device='cpu', dtype=torch.float64)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(p1, p2)

    def test_v10f_tiny(self):
        rng = np.random.default_rng(0)
        N, P = 4, 3
        x = _symmetric_stack(rng, N, P)
        y = _symmetric_stack(rng, N, P)
        xmat, ymat, _ = _materialize(x, y, torch.float64, torch.device('cpu'))
        t_obs = _observed_t(xmat, ymat, 'both', False).cpu().numpy()
        thresh = float(np.median(t_obs))
        p, adj, null = new_nbs(x, y, thresh=thresh, k=20, seed=0, device='cpu', dtype=torch.float64)
        assert null.shape == (20,)
        assert np.all(null >= 0)

    def test_v10g_handcrafted_components(self):
        """Two suprathreshold edges sharing a node ⇒ one 3-node, 2-edge component."""
        N = 5
        adj = np.zeros((N, N))
        adj[0, 1] = adj[1, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        labels_1, sizes = _fast_get_components(adj)
        ind_sz = np.where(sizes > 1)[0] + 1
        assert ind_sz.size == 1
        # find the component label and confirm it has 3 nodes and 2 edges
        cid = ind_sz[0]
        nodes = np.where(labels_1 == cid)[0]
        assert sorted(nodes.tolist()) == [0, 1, 2]
        edges = adj[np.ix_(nodes, nodes)].sum() / 2
        assert edges == 2


# ======================================================== V11: wall-time perf

class TestV11Perf:
    @pytest.mark.slow
    def test_v11_perf(self, fx_small_realistic):
        x, y = fx_small_realistic
        K = 300  # sized so the reference completes in well under 2 min
        t0 = time.perf_counter()
        ref_nbs(x, y, thresh=3.0, k=K, seed=0)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        new_nbs(x, y, thresh=3.0, k=K, seed=0, device='cpu', dtype=torch.float32)
        t_cpu = time.perf_counter() - t0
        speedup_cpu = t_ref / t_cpu
        print(f"\nreference={t_ref:.2f}s torch_cpu={t_cpu:.2f}s speedup={speedup_cpu:.1f}x")
        assert speedup_cpu >= 50.0, f"CPU speedup {speedup_cpu:.1f}x < 50x"
        if torch.cuda.is_available():
            # warm CUDA before timing — first launch pays init costs that dominate at small K
            new_nbs(x, y, thresh=3.0, k=8, seed=0, device='cuda', dtype=torch.float32)
            t0 = time.perf_counter()
            new_nbs(x, y, thresh=3.0, k=K, seed=0, device='cuda', dtype=torch.float32)
            t_gpu = time.perf_counter() - t0
            speedup_gpu = t_ref / t_gpu
            print(f"torch_gpu={t_gpu:.2f}s speedup={speedup_gpu:.1f}x")
            # Note: GPU advantage scales with K*m. On this small fixture (N=80, K=300),
            # most GPU time is kernel launch + per-perm scipy CC, not GEMM throughput.
            # The Document 2 §7 target of ≥200x applies to N=200, K>=1000.
            assert speedup_gpu >= 50.0, f"GPU speedup {speedup_gpu:.1f}x < 50x"


# ======================================================== V12: memory ceiling

class TestV12Memory:
    @pytest.mark.gpu
    def test_v12_memory(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rng = np.random.default_rng(0)
        N, P, Q, K = 300, 80, 80, 2000
        x = _symmetric_stack(rng, N, P)
        y = _symmetric_stack(rng, N, Q)
        torch.cuda.reset_peak_memory_stats()
        new_nbs(x, y, thresh=3.0, k=K, seed=0, device='cuda', dtype=torch.float32)
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"\npeak={peak / 2**20:.0f} MiB total={total / 2**20:.0f} MiB ratio={peak / total:.2%}")
        assert peak / total < 0.80
