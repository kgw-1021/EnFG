from __future__ import annotations
from typing import List, Optional, Tuple
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.graph.graph import Node, Edge, Graph

# ─────────────────────────────────────────────────────────────────────────────
# VNode  –  Variable Node (Upgraded to EKS + ADMM)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Variable node in a factor graph.

    Maintains an ensemble (set of particles) that represents the belief
    over this variable. Receives residual messages and z_target proposals 
    from connected factor nodes, performs an ADMM-regularized EKI update, 
    and applies a small inflation step.

    ADMM Penalty Methods:
      - 'covariance' : Uses ensemble dispersion to dynamically adjust rho (Robust).
      - 'residual'   : Traditional ADMM residual balancing (Boyd et al.).
      - 'fixed'      : Keeps rho constant.
    """

    def __init__(
        self,
        name: str,
        dims: list,
        n_particles: int = 100,
        noise_std: float = 1e-4,
        # --- ADMM Parameters ---
        rho_init: float = 1.0,
        rho_update_method: str = 'covariance',  # 'covariance', 'residual', 'fixed'
        rho_max: float = 100.0,
        alpha_cov: float = 1.0,     # For 'covariance' method
        mu_res: float = 5.0,       # For 'residual' method
        tau_res: float = 1.5,        # For 'residual' method
    ) -> None:
        super().__init__(name, dims)
        self.n_particles = n_particles
        self.noise_std = noise_std

        d = int(np.prod(dims)) if dims else 1
        
        # Initialize ensemble (prior)
        self.ensemble = np.random.randn(d, self.n_particles)
        
        # --- ADMM State Variables ---
        self.rho = rho_init
        self.rho_method = rho_update_method.lower()
        self.rho_max = rho_max
        
        self.lambda_dual = np.zeros((self.dim, 1))  # 누적 패널티 (Dual variable)
        self.z_target: Optional[np.ndarray] = None  # 현재 스텝의 합의점
        self.z_target_prev: Optional[np.ndarray] = None # 이전 스텝의 합의점 (residual balancing용)
        
        # Method-specific hyperparameters
        self.alpha_cov = alpha_cov
        self.mu_res = mu_res
        self.tau_res = tau_res

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
        """ 1. 팩터로부터 메시지를 취합하고 ADMM 제약을 포함하여 파티클을 업데이트 """
        messages = self.collect_messages()
        if not messages:
            return

        E_list, Gamma_list, z_targets = [], [], []
        
        for msg in messages:
            if len(msg) >= 2:
                E_list.append(msg[0])  # 물리 잔차: pred - obs
                Gamma_list.append(msg[1])
            if len(msg) == 3:  
                z_targets.append(msg[2])

        if z_targets:
            self.z_target = np.mean(z_targets, axis=0).reshape(self.dim, 1)

        if not E_list: return

        # 스택킹
        E_stacked = np.vstack(E_list)              # Shape: (d_obs, N)
        Gamma_stacked = block_diag(*Gamma_list)    # Shape: (d_obs, d_obs)

        X = self.ensemble
        N = self.n_particles
        X_mean = np.mean(X, axis=1, keepdims=True)
        E_mean = np.mean(E_stacked, axis=1, keepdims=True)

        # 공분산 계산
        C_xy = ((X - X_mean) @ (E_stacked - E_mean).T) / (N - 1)
        C_yy = ((E_stacked - E_mean) @ (E_stacked - E_mean).T) / (N - 1)

        # 칼만 이득 계산 (수치적 안정을 위해 작은 jitter 추가)
        K = C_xy @ np.linalg.inv(C_yy + Gamma_stacked + 1e-8 * np.eye(Gamma_stacked.shape[0]))

        # ── [오류 1 수정] 올바른 Perturbed Observation 생성 및 부호 수정 ──
        # np.random.multivariate_normal은 (N, d_obs) 형태로 반환하므로 .T 로 (d_obs, N)을 맞춤
        zero_mean = np.zeros(Gamma_stacked.shape[0])
        noise = np.random.multivariate_normal(zero_mean, Gamma_stacked, N).T

        # 수정된 표준 EKI 공식: X_new = X + K * ( -E_stacked + noise )
        self.ensemble = X + K @ (-E_stacked + noise)

        # ── Step 2: ADMM 이동 — 평균에만 적용, 분산 구조 유지 ─────────────────
        if self.rho > 0 and self.z_target is not None:
            x_mean_new = self.ensemble.mean(axis=1, keepdims=True)   # 업데이트 후 평균

            y_virt = self.z_target - (self.lambda_dual / self.rho)

            # C_xx 재계산 (X_new 기준)
            Xc    = self.ensemble - x_mean_new
            C_xx  = (Xc @ Xc.T) / (N - 1)

            # ADMM 평균 이동량: (ρC_xx)(ρC_xx + I)^{-1} (y_virt - x_mean)
            A     = self.rho * C_xx
            shift = A @ np.linalg.inv(A + np.eye(self.dim)) @ (y_virt - x_mean_new)

            # 모든 파티클에 동일한 shift 적용 → 분산 구조 보존
            self.ensemble = self.ensemble + shift

        # (선택) EKI 고유의 앙상블 붕괴를 막기 위한 아주 작은 기본 인플레이션 노이즈
        self.ensemble += np.random.randn(self.dim, N) * self.noise_std

    def update_admm_dual(self) -> None:
        """ 2. 파티클 업데이트 후, 합의점(z_target)과의 오차를 lambda에 누적 """
        if self.rho > 0 and self.z_target is not None:
            x_mean = self.mean.reshape(-1, 1)
            self.lambda_dual += self.rho * (x_mean - self.z_target)

    def update_penalty(self) -> None:
        """ 3. 설정된 방식에 따라 패널티 강도(rho)를 자율적으로 조절 """
        if self.z_target is None:
            return

        if self.rho_method == 'covariance':
            self._update_penalty_covariance()
        elif self.rho_method == 'residual':
            self._update_penalty_residual_balancing()
        
        # 상태 저장
        self.z_target_prev = self.z_target.copy()

    def _update_penalty_covariance(self) -> None:
        """ 제안 기법: 파티클의 퍼짐 정도(Trace of Covariance)에 반비례하게 rho 설정 """
        X = self.ensemble
        Xc = X - np.mean(X, axis=1, keepdims=True)
        C_xx = (Xc @ Xc.T) / (self.n_particles - 1)
        
        trace_c = np.trace(C_xx)
        # 퍼져있을 때 (탐색 중) -> rho 작음 / 뭉쳐있을 때 (수렴 중) -> rho 큼
        new_rho = self.rho_max / (1.0 + self.alpha_cov * trace_c)
        self.rho = min(new_rho, self.rho_max)

    def _update_penalty_residual_balancing(self) -> None:
        """ 기존 기법: 원시 잔차(r)와 쌍대 잔차(s)를 비교하여 rho를 조절 """
        if self.z_target_prev is None:
            return

        x_mean = self.mean.reshape(-1, 1)
        r = np.linalg.norm(x_mean - self.z_target)
        s = np.linalg.norm(self.rho * (self.z_target - self.z_target_prev))

        if r > self.mu_res * s:
            self.rho = min(self.rho * self.tau_res, self.rho_max)
        elif s > self.mu_res * r:
            self.rho = max(self.rho / self.tau_res, 1e-4) # 0으로 가는 것 방지

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
# BinaryFNode  –  factor connected to TWO variable nodes (Updated with ADMM)
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFNode(_FNodeBase):
    """
    Pairwise (binary) factor node: connected to exactly **two** variable nodes.
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma

    # ── abstract pairwise model ───────────────────────────────────────────────

    @abstractmethod
    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """ E = h(x0, x1) - y """
        pass

    @abstractmethod
    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ADMM 합의를 위한 가상의 목표점(z_target) 계산
        
        Returns:
            z_target0 (np.ndarray): vnode0이 가야 할 목표 좌표 (d0,)
            z_target1 (np.ndarray): vnode1이 가야 할 목표 좌표 (d1,)
        """
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
        
        # 1. 앙상블을 통한 기존 잔차 E 계산 (EKI 드리프트 용도)
        E = self._error_function(vnode0.ensemble, vnode1.ensemble)
        
        # 2. 각 변수 노드의 평균을 이용해 ADMM z_target 계산
        z_target0, z_target1 = self._compute_z_targets(vnode0.mean, vnode1.mean)
        
        # 3. 메시지에 (E, Gamma, z_target) 3가지를 담아서 전송
        # VNode에서 z_target을 (dim, 1) 형태로 기대하므로 reshape를 보장해줍니다.
        edge0._messages[vnode0.name] = (E, self.gamma, z_target0.reshape(-1, 1))
        edge1._messages[vnode1.name] = (E, self.gamma, z_target1.reshape(-1, 1))

    def compute_and_send(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Compute residuals and z_targets for both nodes and push to edges."""
        if executor is not None:
            executor.submit(self._do_compute)
        else:
            self._do_compute()

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
# FactorGraph (EKS + ADMM Iterator)
# ─────────────────────────────────────────────────────────────────────────────

class FactorGraph(Graph):
    """
    Factor graph that runs EKI-ADMM based belief propagation.
    """

    def __init__(self, max_workers: int = 4) -> None:
        super().__init__()
        self.max_workers = max_workers

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[Node]: # _FNodeBase 타입 힌팅 대체
        return [n for n in self.nodes if not isinstance(n, VNode)]

    def iterate(self, n_iter: int = 1) -> None:
        """
        Run *n_iter* full message-passing iterations sequentially.
        """
        for _ in range(n_iter):
            # ── Step 1: Factor 노드 연산 및 메시지 생성 (z_target 포함) ──
            for fn in self.fnodes:
                fn.compute_and_send()  # 내부에서 vnode._messages에 저장

            # ── Step 2: Variable 노드 EKI 업데이트 ──
            for vn in self.vnodes:
                vn.eki_update()

            # ── Step 3: ADMM 상태 (Dual, Penalty) 업데이트 ──
            for vn in self.vnodes:
                vn.update_admm_dual()
                vn.update_penalty()

