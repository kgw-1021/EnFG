from __future__ import annotations
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve

from src.graph.graph import Node, Edge, Graph


# ─────────────────────────────────────────────────────────────────────────────
# VNode  –  Variable Node (Gauss-Newton / LM Version)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Variable node for Gauss-Newton / Levenberg-Marquardt optimization.

    단일 평균 벡터(mu)만 유지합니다.
    팩터로부터 (J^T W J, J^T W e) 형태의 Hessian/gradient 기여를 수집하고,
    전역 정규 방정식을 풀어 상태를 업데이트합니다.

    Parameters
    ----------
    name       : unique identifier string
    dims       : list describing the shape/semantics of the variable
    prior_std  : 약한 prior (수치 안정성용, LM의 lambda_init과 별개)
    """

    def __init__(
        self,
        name: str,
        dims: list,
        prior_std: float = 100.0,   # 약한 prior (거의 영향 없음)
    ) -> None:
        super().__init__(name, dims)
        self.prior_std = prior_std
        d = int(np.prod(dims)) if dims else 1

        self.mu: np.ndarray = np.zeros(d, dtype=float)
        self.Sigma: np.ndarray = np.eye(d) * (prior_std ** 2)

    @property
    def dim(self) -> int:
        return self.mu.shape[0]

    def collect_messages(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (H_j, g_j) where
            H_j : J_j^T Gamma_j^{-1} J_j   (d, d)  Hessian 기여
            g_j : J_j^T Gamma_j^{-1} e_j   (d,)    gradient 기여
        """
        pairs = []
        for edge in self.edges:
            msg = edge._messages.get(self.name)
            if msg is not None:
                pairs.append(msg)
        return pairs

    def gn_update(self, lm_lambda: float = 0.0) -> None:
        """
        Gauss-Newton (lm_lambda=0) 또는 LM (lm_lambda>0) 업데이트.

        정규 방정식:
            (H_total + lambda * diag(H_total)) * delta = -g_total
            mu <- mu + delta
        """
        pairs = self.collect_messages()
        if not pairs:
            return

        d = self.dim
        # 약한 prior: 수치 안정성용
        H_total = np.eye(d) / (self.prior_std ** 2)
        g_total = np.zeros(d)

        for H_j, g_j in pairs:
            H_total += H_j
            g_total += g_j

        # LM 댐핑: H의 대각 성분에 비례 (Fletcher variant)
        if lm_lambda > 0:
            H_total += lm_lambda * np.diag(np.diag(H_total))

        try:
            delta = solve(H_total, -g_total)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H_total) @ (-g_total)

        self.mu = self.mu + delta

        try:
            self.Sigma = np.linalg.inv(H_total)
        except np.linalg.LinAlgError:
            self.Sigma = np.linalg.pinv(H_total)


# ─────────────────────────────────────────────────────────────────────────────
# Factor nodes  –  base
# ─────────────────────────────────────────────────────────────────────────────

class _FNodeBase(Node, ABC):
    def __init__(self, name: str, dims: list) -> None:
        super().__init__(name, dims)

    @abstractmethod
    def compute_and_send(self) -> None:
        pass

    @staticmethod
    def _get_vnode(edge: Edge, self_node: Node) -> VNode:
        other = edge.get_other(self_node)
        if not isinstance(other, VNode):
            raise TypeError(f"Expected VNode, got {type(other).__name__}")
        return other


# ─────────────────────────────────────────────────────────────────────────────
# UnaryFNode
# ─────────────────────────────────────────────────────────────────────────────

class UnaryFNode(_FNodeBase):
    """
    Unary factor: connected to ONE variable node.

    서브클래스에서 _error_function(x: (d,) -> (m,)) 구현.
    Jacobian은 중심 차분법으로 자동 계산.
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma)

    @abstractmethod
    def _error_function(self, x: np.ndarray) -> np.ndarray:
        """E = h(x) - y,  x: (d,) -> E: (m,)"""
        pass

    def _jacobian(self, x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """중심 차분 Jacobian. Returns (J, e) where J: (m, d), e: (m,)"""
        e0 = self._error_function(x)
        m, d = len(e0), len(x)
        J = np.zeros((m, d))
        for i in range(d):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            J[:, i] = (self._error_function(xp) - self._error_function(xm)) / (2 * eps)
        return J, e0

    def compute_and_send(self) -> None:
        edge = self.edges[0]
        vnode = self._get_vnode(edge, self)
        J, e = self._jacobian(vnode.mu)
        H = J.T @ self.gamma_inv @ J          # (d, d)
        g = J.T @ self.gamma_inv @ e          # (d,)  gradient = J^T W e
        edge._messages[vnode.name] = (H, g)


# ─────────────────────────────────────────────────────────────────────────────
# BinaryFNode
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFNode(_FNodeBase):
    """
    Binary factor: connected to TWO variable nodes.

    서브클래스에서 _error_function(x0:(d0,), x1:(d1,)) -> (m,) 구현.
    각 변수에 대한 편미분 Jacobian을 수치적으로 계산.
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma)

    @abstractmethod
    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """E = h(x0, x1) - y,  x0:(d0,), x1:(d1,) -> E:(m,)"""
        pass

    def _jacobian(
        self, x0: np.ndarray, x1: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (J0, J1, e)
            J0 : (m, d0)  ∂E/∂x0
            J1 : (m, d1)  ∂E/∂x1
            e  : (m,)     E(x0, x1)
        """
        e0 = self._error_function(x0, x1)
        m, d0, d1 = len(e0), len(x0), len(x1)
        J0 = np.zeros((m, d0))
        J1 = np.zeros((m, d1))

        for i in range(d0):
            xp = x0.copy(); xp[i] += eps
            xm = x0.copy(); xm[i] -= eps
            J0[:, i] = (self._error_function(xp, x1) - self._error_function(xm, x1)) / (2 * eps)

        for i in range(d1):
            xp = x1.copy(); xp[i] += eps
            xm = x1.copy(); xm[i] -= eps
            J1[:, i] = (self._error_function(x0, xp) - self._error_function(x0, xm)) / (2 * eps)

        return J0, J1, e0

    def compute_and_send(self) -> None:
        edge0, edge1 = self.edges
        vnode0 = self._get_vnode(edge0, self)
        vnode1 = self._get_vnode(edge1, self)

        J0, J1, e = self._jacobian(vnode0.mu, vnode1.mu)

        # vnode0으로 보내는 Hessian/gradient 기여
        H0 = J0.T @ self.gamma_inv @ J0
        g0 = J0.T @ self.gamma_inv @ e
        edge0._messages[vnode0.name] = (H0, g0)

        # vnode1으로 보내는 Hessian/gradient 기여
        H1 = J1.T @ self.gamma_inv @ J1
        g1 = J1.T @ self.gamma_inv @ e
        edge1._messages[vnode1.name] = (H1, g1)


# ─────────────────────────────────────────────────────────────────────────────
# FactorGraph  –  Gauss-Newton / LM runner
# ─────────────────────────────────────────────────────────────────────────────

class FactorGraph(Graph):
    """
    Factor graph runner for Gauss-Newton and Levenberg-Marquardt.

    Parameters
    ----------
    lm_lambda_init : float
        초기 LM 댐핑 계수. 0이면 순수 Gauss-Newton.
    lm_adapt : bool
        True이면 잔차 변화에 따라 lambda를 자동 조절 (LM).
        False이면 lambda 고정 (damped GN).
    lm_nu : float
        LM lambda 증가/감소 배율.
    """

    def __init__(
        self,
        lm_lambda_init: float = 0.0,
        lm_adapt: bool = False,
        lm_nu: float = 2.0,
    ) -> None:
        super().__init__()
        self.lm_lambda = lm_lambda_init
        self.lm_adapt = lm_adapt
        self.lm_nu = lm_nu
        self._prev_cost: Optional[float] = None

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[_FNodeBase]:
        return [n for n in self.nodes if isinstance(n, _FNodeBase)]

    def _total_cost(self) -> float:
        """현재 상태에서 전체 weighted residual cost 계산."""
        cost = 0.0
        for fn in self.fnodes:
            if isinstance(fn, UnaryFNode):
                edge = fn.edges[0]
                vnode = fn._get_vnode(edge, fn)
                e = fn._error_function(vnode.mu)
                cost += float(e @ fn.gamma_inv @ e)
            elif isinstance(fn, BinaryFNode):
                edge0, edge1 = fn.edges
                v0 = fn._get_vnode(edge0, fn)
                v1 = fn._get_vnode(edge1, fn)
                e = fn._error_function(v0.mu, v1.mu)
                cost += float(e @ fn.gamma_inv @ e)
        return cost

    def iterate(self, n_iter: int = 1) -> None:
        """
        n_iter 번 GN/LM 업데이트 수행.
        LM 모드에서는 잔차가 줄면 lambda 감소, 늘면 lambda 증가.
        """
        for _ in range(n_iter):
            # ── Step 1: 각 팩터가 (H, g) 메시지 계산 후 전송 ──
            for fn in self.fnodes:
                fn.compute_and_send()

            # ── Step 2: 현재 비용 계산 (LM 적응용) ──
            if self.lm_adapt:
                cost_before = self._total_cost()

            # ── Step 3: 각 변수 노드 GN/LM 업데이트 ──
            # mu를 임시 저장 (LM reject용)
            mu_backup = {vn.name: vn.mu.copy() for vn in self.vnodes}

            for vn in self.vnodes:
                vn.gn_update(lm_lambda=self.lm_lambda)

            # ── Step 4: LM 적응 (lambda 조절) ──
            if self.lm_adapt:
                cost_after = self._total_cost()
                if cost_after < cost_before:
                    # 잔차 감소: 스텝 수용, lambda 줄여서 GN에 가깝게
                    self.lm_lambda = max(self.lm_lambda / self.lm_nu, 1e-7)
                    self._prev_cost = cost_after
                else:
                    # 잔차 증가: 스텝 거부, lambda 키워서 gradient descent에 가깝게
                    for vn in self.vnodes:
                        vn.mu = mu_backup[vn.name]
                    self.lm_lambda = min(self.lm_lambda * self.lm_nu, 1e4)