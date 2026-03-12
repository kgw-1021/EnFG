from __future__ import annotations
from typing import List, Optional, Tuple
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.graph.graph import Node, Edge, Graph


# ─────────────────────────────────────────────────────────────────────────────
# VNode  –  Variable Node
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Variable node in a factor graph.

    Maintains an ensemble (set of particles) that represents the belief
    over this variable.  Receives residual messages from connected factor
    nodes and performs an EKI (Ensemble Kalman Inversion) update that
    simultaneously satisfies *all* incoming residuals, followed by a
    small inflation step to prevent ensemble collapse.

    Parameters
    ----------
    name       : unique identifier string
    dims       : list describing the shape/semantics of the variable
    n_particles: number of ensemble members
    noise_std  : standard deviation of the additive inflation noise
    """

    def __init__(
        self,
        name: str,
        dims: list,
        n_particles: int = 100,
        noise_std: float = 1e-4,
    ) -> None:
        super().__init__(name, dims)
        self.n_particles = n_particles
        self.noise_std   = noise_std

        d = int(np.prod(dims)) if dims else 1
        # Ensemble: shape (d, n_particles)
        self.ensemble: np.ndarray = np.random.randn(d, n_particles)

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def mean(self) -> np.ndarray:
        """Ensemble mean, shape (d,)."""
        return self.ensemble.mean(axis=1)

    @property
    def dim(self) -> int:
        return self.ensemble.shape[0]

    # ── EKI update ───────────────────────────────────────────────────────────

    def collect_messages(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Ask every connected factor edge for the (residual, H) pair it has
        computed for this node.

        Returns a list of (r_j, H_j) tuples where
            r_j  : residual matrix  shape (obs_dim_j, n_particles)
            H_j  : sensitivity matrix shape (obs_dim_j, state_dim)
        Only edges that have already stored a message for this node are
        included.
        """
        pairs = []
        for edge in self.edges:
            msg = edge._messages.get(self.name)
            if msg is not None:
                pairs.append(msg)          # (r_j, H_j)
        return pairs

    def eki_update(self) -> None:
        """
        Joint EKI update from all incoming residual messages.

        For each factor j we have
            r_j^(i)  = y_j  -  f_j( x^(i) )          (obs_dim_j, N)
            H_j      = linearised sensitivity           (obs_dim_j, d)

        Stacking all factors:
            R  = [ r_1; r_2; … ]                       (total_obs, N)
            H  = [ H_1; H_2; … ]                       (total_obs, d)

        EKI gain:
            C_xr = (1/(N-1)) * X' * R'^T               (d, total_obs)
            C_rr = (1/(N-1)) * R' * R'^T + eps*I        (total_obs, total_obs)
            K    = C_xr @ inv(C_rr)                     (d, total_obs)

        Update:
            X  ←  X  +  K @ R                          (d, N)

        Inflation:
            X  ←  X  +  noise_std * randn(d, N)
        """
        pairs = self.collect_messages()
        if not pairs:
            return

        X = self.ensemble
        N = self.n_particles
        Xc = X - X.mean(axis=1, keepdims=True)

        # 1. 수신된 메시지: 에러 E = h(x) - y 와 팩터 공분산 Gamma
        E_list = [E for E, _ in pairs]
        Gamma_list = [Gamma for _, Gamma in pairs]

        E_joint = np.concatenate(E_list, axis=0)        # (total_obs, N)
        Gamma_joint = block_diag(*Gamma_list)           # (total_obs, total_obs)

        # 2. 경험적 공분산 계산
        Ec = E_joint - E_joint.mean(axis=1, keepdims=True)
        C_xE = (Xc @ Ec.T) / (N - 1)                    # Cross-covariance
        C_EE = (Ec @ Ec.T) / (N - 1) + Gamma_joint      # Auto-covariance

        # 3. 칼만 이득
        try:
            K = C_xE @ np.linalg.inv(C_EE)
        except np.linalg.LinAlgError:
            K = C_xE @ np.linalg.pinv(C_EE)

        # 4. EKI 업데이트 (가상 관측치 0 에 노이즈 주입)
        # Target = 0 이므로, Perturbed Target = noise
        noise = np.random.multivariate_normal(np.zeros(E_joint.shape[0]), Gamma_joint, size=N).T
        
        # 잔차 에러 E를 줄이는 방향으로 업데이트
        self.ensemble = X + K @ (noise - E_joint)

        # 5. 붕괴 방지용 인플레이션
        self.ensemble += self.noise_std * np.random.randn(*self.ensemble.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Factor nodes  –  base
# ─────────────────────────────────────────────────────────────────────────────

class _FNodeBase(Node, ABC):
    """
    Internal base shared by UnaryFNode and BinaryFNode.

    Subclasses that implement *concrete* measurement / constraint models
    should override ``_compute_residual_*`` (see each subclass for the
    exact signature).
    """

    def __init__(self, name: str, dims: list) -> None:
        super().__init__(name, dims)

    # ── abstract interface for subclasses ────────────────────────────────────

    @abstractmethod
    def compute_and_send(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Compute residuals and store them on the connected edges."""

    # ── utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_vnode(edge: Edge, self_node: Node) -> VNode:
        """Return the VNode on the other side of *edge*."""
        other = edge.get_other(self_node)
        if not isinstance(other, VNode):
            raise TypeError(
                f"Expected VNode, got {type(other).__name__} on edge of {self_node.name}"
            )
        return other


# ─────────────────────────────────────────────────────────────────────────────
# UnaryFNode  –  factor connected to ONE variable node
# ─────────────────────────────────────────────────────────────────────────────

class UnaryFNode(_FNodeBase):
    """
    Unary factor node: connected to exactly **one** variable node.

    Receives the variable's ensemble, computes a residual
        r^(i) = y  -  f( x^(i) )
    and stores it on the shared edge so the variable node can retrieve it.

    Subclasses
    ----------
    Override ``_measurement_function`` to implement a concrete observation
    model  f: R^d → R^m.  Optionally override ``_sensitivity`` to supply an
    analytic Jacobian (defaults to finite-differences).

    Example
    -------
    class MyObsFactor(UnaryFNode):
        def __init__(self, name, dims, y_obs):
            super().__init__(name, dims)
            self.y_obs = y_obs           # (m,)

        def _measurement_function(self, x_ensemble):  # (d, N) → (m, N)
            return self.y_obs[:, None] - some_sensor_model(x_ensemble)
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma

    # ── abstract measurement model ───────────────────────────────────────────

    @abstractmethod
    def _error_function(self, x_ensemble: np.ndarray) -> np.ndarray:
        """ E = h(x) - y """
        pass

    # ── main interface ────────────────────────────────────────────────────────

    def _find_edge_and_vnode(self) -> Tuple[Edge, VNode]:
        if len(self.edges) != 1:
            raise RuntimeError(
                f"UnaryFNode '{self.name}' must have exactly 1 edge, "
                f"got {len(self.edges)}."
            )
        edge  = self.edges[0]
        vnode = self._get_vnode(edge, self)
        return edge, vnode

    def _do_compute(self) -> None:
        edge, vnode = self._find_edge_and_vnode()
        E = self._error_function(vnode.ensemble)
        edge._messages[vnode.name] = (E, self.gamma)

    def compute_and_send(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Compute residual and push to the edge (optionally async)."""
        if executor is not None:
            executor.submit(self._do_compute)
        else:
            self._do_compute()


# ─────────────────────────────────────────────────────────────────────────────
# BinaryFNode  –  factor connected to TWO variable nodes
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFNode(_FNodeBase):
    """
    Pairwise (binary) factor node: connected to exactly **two** variable nodes.

    Receives ensembles from both variable nodes, computes residuals for
    *each* variable, and stores them on the respective edges.

    Subclasses
    ----------
    Override ``_pairwise_residuals`` to implement a concrete pairwise
    constraint / measurement model.

    Example
    -------
    class RelativePoseFactor(BinaryFNode):
        def __init__(self, name, dims, z_rel):
            super().__init__(name, dims)
            self.z_rel = z_rel               # (m,)  relative observation

        def _pairwise_residuals(self, x0, x1):  # (d0,N),(d1,N) → (m,N),(m,N)
            pred = x1 - x0                   # simple relative model
            r    = self.z_rel[:, None] - pred
            return r, -r                     # residual seen from node0, node1
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma

    # ── abstract pairwise model ───────────────────────────────────────────────

    @abstractmethod
    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """ E = h(x0, x1) - y """
        pass

    # ── main interface ────────────────────────────────────────────────────────

    def _find_edges_and_vnodes(self) -> Tuple[Edge, VNode, Edge, VNode]:
        if len(self.edges) != 2:
            raise RuntimeError(
                f"BinaryFNode '{self.name}' must have exactly 2 edges, "
                f"got {len(self.edges)}."
            )
        edge0, edge1  = self.edges
        vnode0 = self._get_vnode(edge0, self)
        vnode1 = self._get_vnode(edge1, self)
        return edge0, vnode0, edge1, vnode1

    def _do_compute(self) -> None:
        edge0, vnode0, edge1, vnode1 = self._find_edges_and_vnodes()
        E = self._error_function(vnode0.ensemble, vnode1.ensemble)
        edge0._messages[vnode0.name] = (E, self.gamma)
        edge1._messages[vnode1.name] = (E, self.gamma)

    def compute_and_send(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Compute residuals for both nodes and push to edges (optionally async)."""
        if executor is not None:
            executor.submit(self._do_compute)
        else:
            self._do_compute()


class FactorGraph(Graph):
    """
    Factor graph that runs EKI-based belief propagation.

    All factor ``compute_and_send`` calls are dispatched to a thread pool,
    then all variable ``eki_update`` calls are dispatched to another pool,
    ensuring maximum parallelism within each half-iteration.
    """

    def __init__(self, max_workers: int = 4) -> None:
        super().__init__()
        self.max_workers = max_workers

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[_FNodeBase]:
        return [n for n in self.nodes if isinstance(n, _FNodeBase)]

    def iterate(self, n_iter: int = 1) -> None:
        """
        Run *n_iter* full message-passing iterations sequentially.

        Each iteration:
          1. All factor nodes compute & push residuals  (sequential)
          2. All variable nodes perform EKI update      (sequential)
        """
        for _ in range(n_iter):
            # ── Step 1: factor → variable (sequential) ─────────────────────
            # ThreadPoolExecutor 없이 순차적으로 팩터 노드 연산 수행
            for fn in self.fnodes:
                fn.compute_and_send() 

            # ── Step 2: variable EKI update (sequential) ────────────────────
            # 순차적으로 변수 노드의 EKI 업데이트 수행
            for vn in self.vnodes:
                vn.eki_update()