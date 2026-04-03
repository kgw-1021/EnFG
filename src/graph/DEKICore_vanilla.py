from __future__ import annotations
from typing import List, Optional, Tuple
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.graph.graph import Node, Edge, Graph

# ─────────────────────────────────────────────────────────────────────────────
# VNode  –  Variable Node (Modified: Penalty as Virtual Observation in EKI)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Variable node in a factor graph.

    [Ablation Version] 
    ADMM 패널티를 별도의 Shift 연산이 아닌, EKI 내부의 가상 관측값(Virtual Observation)
    으로 스택킹하여 한 번의 칼만 이득(Kalman Gain)으로 업데이트합니다.
    """

    def __init__(
        self,
        name: str,
        dims: list,
        n_particles: int = 100,
        noise_std: float = 1e-4,
        init_std: float = 1.0,
        # --- ADMM Parameters ---
        rho_init: float = 1.0,
        rho_update_method: str = 'residual', 
        rho_max: float = 100.0,
        alpha_cov: float = 1.0,     
        mu_res: float = 5.0,       
        tau_res: float = 1.5,       
        debug_mode: bool = False, 
    ) -> None:
        super().__init__(name, dims)
        self.n_particles = n_particles
        self.base_noise_std = noise_std
        self.current_noise_std = noise_std 

        d = int(np.prod(dims)) if dims else 1
        self.ensemble = np.random.randn(d, self.n_particles) * init_std
        
        self.rho = rho_init
        self.rho_method = rho_update_method.lower()
        self.rho_max = rho_max
        self.lambda_dual = np.zeros((self.dim, 1))
        self.z_target: Optional[np.ndarray] = None
        self.z_target_prev: Optional[np.ndarray] = None 
        self.C_xx: np.ndarray = np.eye(self.dim) * init_std ** 2
        
        # Method-specific hyperparameters
        self.alpha_cov = alpha_cov
        self.mu_res = mu_res
        self.tau_res = tau_res

        self.debug_mode = debug_mode
        self.debug_history = {
            'rho': [], 'trace_C_xx': [], 'primal_residual': [],
            'dual_residual': [], 'lambda_norm': [], 'mean': [],
            'z_target': [], 'factor_residuals': [], 
            'current_noise': [] 
        }

    def _compute_residuals(self) -> Tuple[float, float]:
        if self.z_target is None or self.z_target_prev is None:
            return 0.0, 0.0
        x_mean = self.mean.reshape(-1, 1)
        r = float(np.linalg.norm(x_mean - self.z_target))
        s = float(np.linalg.norm(self.rho * (self.z_target - self.z_target_prev)))
        return r, s

    def _record_debug_info(self) -> None:
        r, s = self._compute_residuals()
        self.debug_history['rho'].append(self.rho)
        self.debug_history['trace_C_xx'].append(float(np.trace(self.C_xx)))
        self.debug_history['primal_residual'].append(r)
        self.debug_history['dual_residual'].append(s)
        self.debug_history['lambda_norm'].append(float(np.linalg.norm(self.lambda_dual)))
        self.debug_history['mean'].append(self.mean.copy())
        self.debug_history['z_target'].append(self.z_target.copy() if self.z_target is not None else None)
        self.debug_history['current_noise'].append(self.current_noise_std)
        
        factor_res_dict = {}
        for edge in self.edges:
            msg = edge._messages.get(self.name)
            if msg is not None:
                fnode = edge.get_other(self)
                factor_res_dict[fnode.name] = msg[0].copy() 
        self.debug_history['factor_residuals'].append(factor_res_dict)

    def _update_dynamic_scale(self) -> None:
        r, s = self._compute_residuals()
        max_res = max(r, s)
        
        # 잔차가 클 때는 1.0 (최대치) 유지, 잔차가 줄어들면 스케일도 함께 감소
        # 비례 상수 2.0은 잔차가 줄기 시작할 때 조금 더 여유있게 탐색을 허용하기 위한 튜닝값입니다.
        scale_factor = min(1.0, max_res * 2.0)
        
        # 완전한 0이 되면 EKI 칼만 이득 역행렬 계산 시 특이성 문제가 생길 수 있으므로
        # 극소수의 하한선(1e-8)을 두어 수치적 안정성을 확보합니다.
        self.current_noise_std = self.base_noise_std * max(scale_factor, 1e-8)

    @property
    def mean(self) -> np.ndarray:
        return self.ensemble.mean(axis=1)

    @property
    def dim(self) -> int:
        return self.ensemble.shape[0]

    def collect_messages(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for edge in self.edges:
            msg = edge._messages.get(self.name)
            if msg is not None:
                pairs.append(msg)
        return pairs

    def eki_update(self) -> None:
        """ 1. 팩터 메시지와 ADMM 제약(가상 관측)을 모두 통합하여 EKI 업데이트 """
        messages = self.collect_messages()

        E_list, Gamma_list, z_targets = [], [], []
        
        for msg in messages:
            if len(msg) >= 2:
                E_list.append(msg[0])  # 물리/관측 잔차
                Gamma_list.append(msg[1])
            if len(msg) == 3:  
                z_targets.append(msg[2])

        if z_targets:
            self.z_target = np.mean(z_targets, axis=0).reshape(self.dim, 1)

        # ── [핵심 변경] ADMM 제약을 가상 관측값으로 변환하여 리스트에 추가 ──
        if self.rho > 0 and self.z_target is not None:
            # y_virt = z_target - (lambda / rho)
            y_virt = self.z_target - (self.lambda_dual / self.rho)
            
            # 가상 잔차 E_virt = x_pred - y_virt
            E_virt = self.ensemble - y_virt  # Shape: (dim, N)
            
            # 가상 노이즈 공분산 Gamma_virt = (1 / rho) * I
            Gamma_virt = np.eye(self.dim) / self.rho
            
            E_list.append(E_virt)
            Gamma_list.append(Gamma_virt)
        # ─────────────────────────────────────────────────────────────────

        if not E_list: return

        # 스택킹 (물리 잔차 + 가상 제약 잔차를 하나로 합침)
        E_stacked = np.vstack(E_list)              
        Gamma_stacked = block_diag(*Gamma_list)    

        X = self.ensemble
        N = self.n_particles
        X_mean = np.mean(X, axis=1, keepdims=True)
        E_mean = np.mean(E_stacked, axis=1, keepdims=True)

        # 공분산 계산
        C_xy = ((X - X_mean) @ (E_stacked - E_mean).T) / (N - 1)
        C_yy = ((E_stacked - E_mean) @ (E_stacked - E_mean).T) / (N - 1)

        # 한 번의 통합된 칼만 이득 계산
        K = C_xy @ np.linalg.inv(C_yy + Gamma_stacked + 1e-8 * np.eye(Gamma_stacked.shape[0]))

        zero_mean = np.zeros(Gamma_stacked.shape[0])
        noise = np.random.multivariate_normal(zero_mean, Gamma_stacked, N).T

        # EKI 단일 업데이트 수행
        self.ensemble = X + K @ (-E_stacked + noise)
        x_mean_new = self.ensemble.mean(axis=1, keepdims=True)
        Xc    = self.ensemble - x_mean_new
        self.C_xx = (Xc @ Xc.T) / (N - 1)
        
        # 1. 노이즈 스케일 동적 업데이트
        self._update_dynamic_scale()
        
        # 2. 고유값 분해 및 정규화 (방향성 추출)
        evals, evecs = np.linalg.eigh(self.C_xx)
        evals = np.maximum(evals, 0.0)
        
        max_eval = np.max(evals)
        if max_eval > 1e-8:
            evals_normalized = evals / max_eval
        else:
            evals_normalized = evals
            
        sqrt_C_xx_norm = evecs @ np.diag(np.sqrt(evals_normalized))
        
        # 3. 방향성(Shape) 매트릭스에 줄어드는 스케일(current_noise_std)을 곱해 적용
        cov_aware_noise = sqrt_C_xx_norm @ np.random.randn(self.dim, N)
        self.ensemble += cov_aware_noise * self.current_noise_std
        

    def update_admm_dual(self) -> None:
        if self.rho > 0 and self.z_target is not None:
            x_mean = self.mean.reshape(-1, 1)
            self.lambda_dual += self.rho * (x_mean - self.z_target)

    def update_penalty(self) -> None:
            if self.z_target is None:
                return
            if self.rho_method == 'covariance':
                self._update_penalty_covariance()
            elif self.rho_method == 'residual':
                self._update_penalty_residual_balancing()
            
            if self.debug_mode:
                self._record_debug_info()
                
            self.z_target_prev = self.z_target.copy()

    def _update_penalty_covariance(self) -> None:
        trace_c = np.trace(self.C_xx)
        new_rho = self.rho_max / (1.0 + self.alpha_cov * trace_c)
        self.rho = min(new_rho, self.rho_max)

    def _update_penalty_residual_balancing(self) -> None:
        if self.z_target_prev is None: return
        r, s = self._compute_residuals()
        if r > self.mu_res * s:
            self.rho = min(self.rho * self.tau_res, self.rho_max)
        elif s > self.mu_res * r:
            self.rho = max(self.rho / self.tau_res, 1e-4)

# ─────────────────────────────────────────────────────────────────────────────
# Factor nodes & Graph (동일하게 유지)
# ─────────────────────────────────────────────────────────────────────────────
class _FNodeBase(Node, ABC):
    """
    Internal base shared by UnaryFNode and BinaryFNode.
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
    def __init__(self) -> None:
        super().__init__()

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[Node]: 
        return [n for n in self.nodes if not isinstance(n, VNode)]

    def iterate(self, n_iter: int = 1) -> None:
        """
        Run *n_iter* full message-passing iterations sequentially.
        """
        for _ in range(n_iter):
            # ── Step 1: Factor 노드 연산 및 메시지 생성 (z_target 포함) ──
            for fn in self.fnodes:
                fn.compute_and_send()  

            # ── Step 2: Variable 노드 EKI 업데이트 ──
            for vn in self.vnodes:
                vn.eki_update()

            # ── Step 3: ADMM 상태 (Dual, Penalty) 업데이트 ──
            for vn in self.vnodes:
                vn.update_admm_dual()
                vn.update_penalty()