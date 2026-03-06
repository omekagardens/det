"""
DET-C1: Integrated Conscious Regime & Bond-Aware Communication (v6.3 branch)
=============================================================================

Research extension module that sits on top of the DET v6.3 3D collider.

This module is intentionally non-canonical:
- It does not modify core collider state update laws.
- It computes local readouts over user-declared regimes.
- It enforces locality for inter-regime communication via bonded neighbor paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from det_v6_3_3d_collider import DETCollider3D


Coord3D = Tuple[int, int, int]


@dataclass
class ConsciousnessParamsC1:
    """Parameters for DET-C1 readouts."""

    alpha_U: float = 1.0
    beta_U: float = 1.0
    V0: float = 1.0
    path_presence_min: float = 0.0
    path_coherence_min: float = 0.0
    periodic_paths: bool = False
    epsilon: float = 1e-12


@dataclass
class RegimeStateC1:
    """Regime-level readouts for DET-C1."""

    name: str
    U: float
    bar_P: float
    bar_C: float
    H_raw: float
    X: float
    K: float
    H_eff: float
    P_eff: float
    W: float
    R: float
    node_count: int


@dataclass
class PathStateC1:
    """Path-level communication readout between two regimes."""

    regime_a: str
    regime_b: str
    path_exists: bool
    hop_count: int
    path_presence: float
    path_coherence: float
    Gamma: float
    V: float
    nonverbal_accuracy: float
    path_nodes: List[Coord3D]


def _iter_all_coords(shape: Tuple[int, int, int]) -> Iterable[Coord3D]:
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                yield z, y, x


class DETConsciousnessC1:
    """
    Local readout module for consciousness-style regime integration and communication.

    Core sketch:
      K_G = alpha_U * U_G * bar_C_G
      X_G = beta_U  * U_G * (1 - bar_C_G)
      P_eff_G ~ bar_P_G * (1 + K_G) / (1 + X_G)

      Gamma_AB = P_AB * C_AB * sqrt(U_A * U_B)
      V_AB = V0 / (1 + Gamma_AB)
    """

    def __init__(self, collider: DETCollider3D, params: Optional[ConsciousnessParamsC1] = None):
        self.sim = collider
        self.params = params or ConsciousnessParamsC1()

    def _compute_presence_snapshot(self) -> np.ndarray:
        """
        Recompute instantaneous presence readout without stepping dynamics.

        This mirrors the collider's local presence expression.
        """
        p = self.sim.p
        if p.coherence_weighted_H:
            H = self.sim._compute_coherence_weighted_H()  # pylint: disable=protected-access
        else:
            H = self.sim.sigma

        if p.debt_temporal_enabled:
            debt_temporal_factor = 1.0 + p.zeta_temporal * self.sim.q
        else:
            debt_temporal_factor = 1.0

        return self.sim.a * self.sim.sigma / (1.0 + self.sim.F) / (1.0 + H) / debt_temporal_factor

    def _compute_node_coherence(self) -> np.ndarray:
        """Per-node mean coherence over the six local bonds."""
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)

        return (
            self.sim.C_X
            + np.roll(self.sim.C_X, 1, axis=2)
            + self.sim.C_Y
            + np.roll(self.sim.C_Y, 1, axis=1)
            + self.sim.C_Z
            + np.roll(self.sim.C_Z, 1, axis=0)
        ) / 6.0

    def _compute_load_snapshot(self) -> np.ndarray:
        p = self.sim.p
        if p.coherence_weighted_H:
            return self.sim._compute_coherence_weighted_H()  # pylint: disable=protected-access
        return self.sim.sigma

    @staticmethod
    def _validate_mask(mask: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} does not match collider shape {shape}.")
        return mask.astype(bool)

    def compute_regime_state(self, name: str, mask: np.ndarray, U: float) -> RegimeStateC1:
        """
        Compute DET-C1 regime metrics for one regime mask.
        """
        eps = self.params.epsilon
        mask = self._validate_mask(mask, self.sim.F.shape)
        U_clip = float(np.clip(U, 0.0, 1.0))

        if not np.any(mask):
            raise ValueError(f"Regime '{name}' has an empty mask.")

        P_field = self._compute_presence_snapshot()
        C_field = self._compute_node_coherence()
        H_field = self._compute_load_snapshot()

        bar_P = float(np.mean(P_field[mask]))
        bar_C = float(np.mean(C_field[mask]))
        H_raw = float(np.mean(H_field[mask]))

        K = self.params.alpha_U * U_clip * bar_C
        X = self.params.beta_U * U_clip * (1.0 - bar_C)
        H_eff = max(0.0, H_raw + X - K)
        P_eff = bar_P * (1.0 + K) / (1.0 + X + eps)

        # Higher integration/coherence lowers symbolic compensation burden.
        R = U_clip * bar_C
        W = self.params.V0 / (1.0 + R)

        return RegimeStateC1(
            name=name,
            U=U_clip,
            bar_P=bar_P,
            bar_C=bar_C,
            H_raw=H_raw,
            X=X,
            K=K,
            H_eff=H_eff,
            P_eff=P_eff,
            W=W,
            R=R,
            node_count=int(np.sum(mask)),
        )

    def compute_regime_states(
        self, regime_masks: Dict[str, np.ndarray], regime_U: Dict[str, float]
    ) -> Dict[str, RegimeStateC1]:
        """
        Compute states for multiple regimes.
        """
        states: Dict[str, RegimeStateC1] = {}
        for name, mask in regime_masks.items():
            U = regime_U.get(name, 0.0)
            states[name] = self.compute_regime_state(name=name, mask=mask, U=U)
        return states

    def _neighbors(self, node: Coord3D, shape: Tuple[int, int, int]) -> Iterable[Coord3D]:
        z, y, x = node
        nz, ny, nx = shape
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            zz, yy, xx = z + dz, y + dy, x + dx
            if self.params.periodic_paths:
                yield zz % nz, yy % ny, xx % nx
            else:
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    yield zz, yy, xx

    def _bond_coherence(self, u: Coord3D, v: Coord3D) -> float:
        """
        Get the local bond coherence connecting neighboring nodes u and v.
        """
        uz, uy, ux = u
        vz, vy, vx = v
        dz, dy, dx = vz - uz, vy - uy, vx - ux

        # Handle periodic wrapped neighbors by choosing shortest signed delta.
        if self.params.periodic_paths:
            shape = self.sim.F.shape
            if dz > shape[0] // 2:
                dz -= shape[0]
            if dz < -shape[0] // 2:
                dz += shape[0]
            if dy > shape[1] // 2:
                dy -= shape[1]
            if dy < -shape[1] // 2:
                dy += shape[1]
            if dx > shape[2] // 2:
                dx -= shape[2]
            if dx < -shape[2] // 2:
                dx += shape[2]

        if dx == 1 and dy == 0 and dz == 0:
            return float(self.sim.C_X[uz, uy, ux])
        if dx == -1 and dy == 0 and dz == 0:
            return float(self.sim.C_X[vz, vy, vx])
        if dy == 1 and dx == 0 and dz == 0:
            return float(self.sim.C_Y[uz, uy, ux])
        if dy == -1 and dx == 0 and dz == 0:
            return float(self.sim.C_Y[vz, vy, vx])
        if dz == 1 and dx == 0 and dy == 0:
            return float(self.sim.C_Z[uz, uy, ux])
        if dz == -1 and dx == 0 and dy == 0:
            return float(self.sim.C_Z[vz, vy, vx])

        return 0.0

    def _bond_presence(self, u: Coord3D, v: Coord3D, P_field: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, P_field[u]) * max(0.0, P_field[v])))

    def _mask_coords(self, mask: np.ndarray) -> List[Coord3D]:
        return [coord for coord in _iter_all_coords(mask.shape) if mask[coord]]

    def _find_local_path(
        self,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        P_field: np.ndarray,
    ) -> List[Coord3D]:
        """
        BFS over local neighbors with local bond threshold gating.
        """
        coords_a = self._mask_coords(mask_a)
        coords_b_set = set(self._mask_coords(mask_b))
        if not coords_a or not coords_b_set:
            return []

        # If overlap exists, path is immediate.
        for c in coords_a:
            if c in coords_b_set:
                return [c]

        q = deque(coords_a)
        visited = set(coords_a)
        parent: Dict[Coord3D, Coord3D] = {}

        while q:
            u = q.popleft()
            for v in self._neighbors(u, self.sim.F.shape):
                if v in visited:
                    continue
                coh = self._bond_coherence(u, v)
                pres = self._bond_presence(u, v, P_field)

                if coh < self.params.path_coherence_min:
                    continue
                if pres < self.params.path_presence_min:
                    continue

                visited.add(v)
                parent[v] = u
                if v in coords_b_set:
                    path = [v]
                    while path[-1] in parent:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(v)

        return []

    def _path_bond_metrics(self, path_nodes: List[Coord3D], P_field: np.ndarray) -> Tuple[float, float]:
        if len(path_nodes) <= 1:
            return 1.0, 1.0

        coherences: List[float] = []
        presences: List[float] = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            coherences.append(self._bond_coherence(u, v))
            presences.append(self._bond_presence(u, v, P_field))

        path_coherence = float(np.mean(coherences))
        # Geometric mean is a robust path-throughput proxy.
        path_presence = float(np.exp(np.mean(np.log(np.asarray(presences) + self.params.epsilon))))
        return path_presence, path_coherence

    @staticmethod
    def nonverbal_accuracy_from_gamma(gamma: float) -> float:
        """Bounded [0,1) communication accuracy proxy."""
        gamma_clip = max(0.0, float(gamma))
        return gamma_clip / (1.0 + gamma_clip)

    def compute_path_state(
        self,
        regime_a: str,
        regime_b: str,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        U_a: float,
        U_b: float,
    ) -> PathStateC1:
        """
        Compute DET-C1 path communication metrics between two regimes.
        """
        eps = self.params.epsilon
        mask_a = self._validate_mask(mask_a, self.sim.F.shape)
        mask_b = self._validate_mask(mask_b, self.sim.F.shape)
        U_a = float(np.clip(U_a, 0.0, 1.0))
        U_b = float(np.clip(U_b, 0.0, 1.0))

        P_field = self._compute_presence_snapshot()
        path_nodes = self._find_local_path(mask_a=mask_a, mask_b=mask_b, P_field=P_field)

        if not path_nodes:
            return PathStateC1(
                regime_a=regime_a,
                regime_b=regime_b,
                path_exists=False,
                hop_count=0,
                path_presence=0.0,
                path_coherence=0.0,
                Gamma=0.0,
                V=self.params.V0,
                nonverbal_accuracy=0.0,
                path_nodes=[],
            )

        path_presence, path_coherence = self._path_bond_metrics(path_nodes, P_field)
        Gamma = path_presence * path_coherence * float(np.sqrt(U_a * U_b))
        V = self.params.V0 / (1.0 + Gamma + eps)
        acc = self.nonverbal_accuracy_from_gamma(Gamma)

        return PathStateC1(
            regime_a=regime_a,
            regime_b=regime_b,
            path_exists=True,
            hop_count=max(0, len(path_nodes) - 1),
            path_presence=path_presence,
            path_coherence=path_coherence,
            Gamma=Gamma,
            V=V,
            nonverbal_accuracy=acc,
            path_nodes=path_nodes,
        )
