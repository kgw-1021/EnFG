from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# 사용자의 graph.py 구조 임포트
from src.graph.graph import Node, Edge, Graph

# ─────────────────────────────────────────────────────────────────────────────
# 1. VNode – Variable Node (상태만 저장, 업데이트는 전역 솔버가 수행)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    전역 최적화를 위한 Variable Node.
    자체적으로 최적화를 수행하지 않으며, 상태(mu)만 저장합니다.
    FactorGraph(중앙 솔버)가 전역 인덱스를 할당하고 업데이트를 관리합니다.
    """
    def __init__(self, name: str, dims: list, initial_mu: np.ndarray = None) -> None:
        super().__init__(name, dims)
        self.d = int(np.prod(dims)) if dims else 1
        self.mu: np.ndarray = initial_mu.copy() if initial_mu is not None else np.zeros(self.d)
        
        # 전역 상태 벡터(Global State Vector)에서의 인덱스 범위
        self.global_start_idx: int = -1
        self.global_end_idx: int = -1

# ─────────────────────────────────────────────────────────────────────────────
# 2. Factor Nodes (에러 및 자코비안 계산 전용)
# ─────────────────────────────────────────────────────────────────────────────

class _FNodeBase(Node, ABC):
    def __init__(self, name: str, gamma: np.ndarray) -> None:
        super().__init__(name, dims=[])
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma) # Information Matrix (W)
        
    def _get_vnode(self, edge: Edge) -> VNode:
        """
        [수정됨] Edge 객체의 get_other 메서드를 사용하여
        팩터 노드(self)와 연결된 반대편 변수 노드(VNode)를 안전하게 가져옵니다.
        """
        return edge.get_other(self)


class UnaryFNode(_FNodeBase, ABC):
    @abstractmethod
    def error(self, x: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray: pass


class BinaryFNode(_FNodeBase, ABC):
    @abstractmethod
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. FactorGraph - Centralized LM Solver (핵심)
# ─────────────────────────────────────────────────────────────────────────────

class FactorGraph(Graph):
    """
    거대한 희소 행렬(Sparse Matrix)을 조립하여 전역 최적화를 수행하는 중앙집중형 솔버.
    """
    def __init__(self, lm_lambda_init: float = 1e-3) -> None:
        super().__init__()
        self.lm_lambda = lm_lambda_init
        self.total_dims = 0

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[_FNodeBase]:
        return [n for n in self.nodes if isinstance(n, _FNodeBase)]

    def _assign_global_indices(self) -> None:
        """ 모든 변수 노드에 대해 전역 벡터(Global Vector)의 인덱스를 연속적으로 할당합니다. """
        idx = 0
        for vn in self.vnodes:
            vn.global_start_idx = idx
            idx += vn.d
            vn.global_end_idx = idx
        self.total_dims = idx

    def _get_global_state(self) -> np.ndarray:
        """ 모든 VNode의 상태를 모아 하나의 거대한 1차원 벡터 X를 만듭니다. """
        X = np.zeros(self.total_dims)
        for vn in self.vnodes:
            X[vn.global_start_idx : vn.global_end_idx] = vn.mu
        return X

    def _set_global_state(self, X: np.ndarray) -> None:
        """ 업데이트된 전역 벡터 X를 다시 개별 VNode의 mu로 분배합니다. """
        for vn in self.vnodes:
            vn.mu = X[vn.global_start_idx : vn.global_end_idx]

    def _compute_total_cost(self) -> float:
        """ 현재 상태에서의 전체 그래프의 잔차 제곱합(Cost)을 계산합니다. """
        cost = 0.0
        for fn in self.fnodes:
            if isinstance(fn, UnaryFNode):
                v0 = fn._get_vnode(fn.edges[0])
                e = fn.error(v0.mu)
                cost += float(e.T @ fn.gamma_inv @ e)
            elif isinstance(fn, BinaryFNode):
                v0 = fn._get_vnode(fn.edges[0])
                v1 = fn._get_vnode(fn.edges[1])
                e = fn.error(v0.mu, v1.mu)
                cost += float(e.T @ fn.gamma_inv @ e)
        return cost / 2.0

    def _build_linear_system(self) -> Tuple[sp.csc_matrix, np.ndarray]:
        """ 
        모든 팩터의 자코비안을 모아 전역 H(Hessian) 행렬과 g(Gradient) 벡터를 조립(Assembly)합니다.
        [비대각 성분(Cross-correlation)이 완벽하게 반영되는 부분]
        """
        rows, cols, data = [], [], []
        g_global = np.zeros(self.total_dims)

        def add_block(r_start, r_end, c_start, c_end, block):
            for i in range(r_end - r_start):
                for j in range(c_end - c_start):
                    rows.append(r_start + i)
                    cols.append(c_start + j)
                    data.append(block[i, j])

        for fn in self.fnodes:
            if isinstance(fn, UnaryFNode):
                v0 = fn._get_vnode(fn.edges[0])
                e = fn.error(v0.mu)
                J0 = fn.jacobian(v0.mu)

                H00 = J0.T @ fn.gamma_inv @ J0
                g0 = J0.T @ fn.gamma_inv @ e

                idx0_s, idx0_e = v0.global_start_idx, v0.global_end_idx
                add_block(idx0_s, idx0_e, idx0_s, idx0_e, H00)
                g_global[idx0_s:idx0_e] += g0

            elif isinstance(fn, BinaryFNode):
                v0 = fn._get_vnode(fn.edges[0])
                v1 = fn._get_vnode(fn.edges[1])
                
                e = fn.error(v0.mu, v1.mu)
                J0, J1 = fn.jacobians(v0.mu, v1.mu)

                # Information matrix blocks
                H00 = J0.T @ fn.gamma_inv @ J0
                H11 = J1.T @ fn.gamma_inv @ J1
                H01 = J0.T @ fn.gamma_inv @ J1
                H10 = J1.T @ fn.gamma_inv @ J0

                g0 = J0.T @ fn.gamma_inv @ e
                g1 = J1.T @ fn.gamma_inv @ e

                idx0_s, idx0_e = v0.global_start_idx, v0.global_end_idx
                idx1_s, idx1_e = v1.global_start_idx, v1.global_end_idx

                # Block assembly (Diagonal & Off-diagonal)
                add_block(idx0_s, idx0_e, idx0_s, idx0_e, H00)
                add_block(idx1_s, idx1_e, idx1_s, idx1_e, H11)
                add_block(idx0_s, idx0_e, idx1_s, idx1_e, H01)  # Cross-correlation 1
                add_block(idx1_s, idx1_e, idx0_s, idx0_e, H10)  # Cross-correlation 2

                g_global[idx0_s:idx0_e] += g0
                g_global[idx1_s:idx1_e] += g1

        H_global = sp.coo_matrix((data, (rows, cols)), shape=(self.total_dims, self.total_dims)).tocsc()
        return H_global, g_global

    def iterate(self, n_iter: int = 1) -> None:
        """
        중앙집중형 Levenberg-Marquardt 최적화를 수행합니다.
        """
        # 1. 인덱싱 (초기 1회만 수행해도 무방)
        if self.total_dims == 0:
            self._assign_global_indices()

        for step in range(n_iter):
            # 2. 전역 선형 시스템 조립
            H_global, g_global = self._build_linear_system()
            current_cost = self._compute_total_cost()
            X_current = self._get_global_state()

            # 3. LM Damping 루프 (성공적인 스텝을 찾을 때까지 람다 조정)
            max_lm_attempts = 10
            step_accepted = False

            for attempt in range(max_lm_attempts):
                H_diag = H_global.diagonal()
                H_diag = np.maximum(H_diag, 1e-6) # 0 방지
                D_matrix = sp.diags(H_diag)
                
                H_lm = H_global + self.lm_lambda * D_matrix
                
                # 4. 희소 선형 시스템 풀이 (Sparse Solve: H * dX = -g)
                try:
                    dX = splinalg.spsolve(H_lm, -g_global)
                except RuntimeError:
                    # 행렬이 특이(Singular)한 경우 람다를 키워 억제
                    self.lm_lambda *= 10.0
                    continue

                # 5. 상태 업데이트 및 평가
                self._set_global_state(X_current + dX)
                new_cost = self._compute_total_cost()

                # 6. Step Accept / Reject 판정
                if new_cost < current_cost:
                    # 성공: 람다를 줄이고 다음 이터레이션으로 넘어감
                    self.lm_lambda = max(1e-7, self.lm_lambda / 10.0)
                    step_accepted = True
                    break
                else:
                    # 실패: 상태를 롤백하고 람다를 키워서 다시 시도
                    self._set_global_state(X_current)
                    self.lm_lambda *= 10.0

            if not step_accepted:
                # 더 이상 비용을 줄일 수 없음 (수렴 완료)
                break