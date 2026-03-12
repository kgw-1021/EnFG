from __future__ import annotations
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve

# user의 graph.py 구조를 그대로 사용한다고 가정
from src.graph.graph import Node, Edge, Graph


# ─────────────────────────────────────────────────────────────────────────────
# VNode  –  Variable Node (GaBP Version)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Variable node in a factor graph for Gaussian Belief Propagation.

    EKI 버전이 '앙상블(Ensemble)'을 유지했다면, GaBP 버전은 
    가우시안 분포의 파라미터인 '평균(mu)'과 '공분산(Sigma)'을 유지합니다.
    팩터로부터 전달된 정보 행렬(Information Matrix, Lambda)과 
    정보 벡터(Information Vector, eta)를 합산하여 상태를 업데이트합니다.

    Parameters
    ----------
    name       : unique identifier string
    dims       : list describing the shape/semantics of the variable
    prior_std  : 초기 공분산 및 발산 방지용 Prior의 표준편차
    """

    def __init__(
        self,
        name: str,
        dims: list,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__(name, dims)
        self.prior_std = prior_std
        d = int(np.prod(dims)) if dims else 1
        
        # State: Mean and Covariance instead of Ensemble
        self.mu: np.ndarray = np.zeros(d)  # 초기 평균값 (필요시 랜덤 초기화 가능)
        self.Sigma: np.ndarray = np.eye(d) * (prior_std ** 2)

    @property
    def dim(self) -> int:
        return self.mu.shape[0]

    # ── GaBP update (Gauss-Newton step) ──────────────────────────────────────

    def collect_messages(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        연결된 모든 팩터 에지로부터 메시지를 수집합니다.
        Returns a list of (Lambda_j, eta_j)
            Lambda_j : Information Matrix contribution (d, d)
            eta_j    : Information Vector contribution (d,)
        """
        pairs = []
        for edge in self.edges:
            msg = edge._messages.get(self.name)
            if msg is not None:
                pairs.append(msg)
        return pairs

    def gabp_update(self) -> None:
        """
        수집된 정보(Information)를 바탕으로 MAP(Maximum A Posteriori) 업데이트를 수행합니다.
        수식: 
            Lambda_total = Lambda_prior + sum(Lambda_j)
            eta_total = sum(eta_j)
            Delta_mu = Lambda_total^-1 * eta_total
            mu_new = mu_old + Delta_mu
        """
        pairs = self.collect_messages()
        if not pairs:
            return

        # 약간의 Prior를 더해주어 자코비안이 부족할 때 행렬이 Singular가 되는 것을 방지 (Levenberg-Marquardt 역할)
        Lambda_total = np.eye(self.dim) * (1.0 / (self.prior_std ** 2))
        eta_total = np.zeros(self.dim)

        # 1. 수신된 메시지(정보 행렬과 정보 벡터) 합산
        for Lambda_j, eta_j in pairs:
            Lambda_total += Lambda_j
            eta_total += eta_j

        # 2. 업데이트량(Delta mu) 계산 (역행렬 직접 계산 대신 solve 사용)
        try:
            d_mu = solve(Lambda_total, eta_total)
        except np.linalg.LinAlgError:
            # 특이 행렬(Singular matrix) 발생 시 Pseudo-inverse로 Fallback
            d_mu = np.linalg.pinv(Lambda_total) @ eta_total

        # 3. 상태(Mean) 및 공분산(Covariance) 업데이트
        self.mu += d_mu
        try:
            self.Sigma = np.linalg.inv(Lambda_total)
        except np.linalg.LinAlgError:
            self.Sigma = np.linalg.pinv(Lambda_total)


# ─────────────────────────────────────────────────────────────────────────────
# Factor nodes  –  base
# ─────────────────────────────────────────────────────────────────────────────

class _FNodeBase(Node, ABC):
    """
    Internal base shared by UnaryFNode and BinaryFNode.
    GaBP에서는 비선형 모델의 Jacobian 계산이 필수적이므로, 
    사용자가 직접 제공하지 않은 경우 수치적 미분(Finite Differences)을 수행합니다.
    """

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
# UnaryFNode  –  factor connected to ONE variable node
# ─────────────────────────────────────────────────────────────────────────────

class UnaryFNode(_FNodeBase):
    """
    Unary factor node.
    현재 변수 노드의 평균(mu)을 바탕으로 잔차(Error)와 야코비안(Jacobian)을 계산한 뒤,
    정보 행렬(Lambda)과 정보 벡터(eta)를 생성하여 에지에 저장합니다.
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma)

    @abstractmethod
    def _error_function(self, x: np.ndarray) -> np.ndarray:
        """ E = h(x) - y. 입력 x는 1D array (d,) 형태입니다. """
        pass

    def _numerical_jacobian(self, x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """ 중심 차분법(Central Difference)을 이용한 수치적 야코비안 계산 """
        e0 = self._error_function(x)
        m, d = len(e0), len(x)
        J = np.zeros((m, d))
        
        for i in range(d):
            x_p = x.copy(); x_p[i] += eps
            x_m = x.copy(); x_m[i] -= eps
            J[:, i] = (self._error_function(x_p) - self._error_function(x_m)) / (2 * eps)
        return J, e0

    def compute_and_send(self) -> None:
        edge = self.edges[0]
        vnode = self._get_vnode(edge, self)

        # 1. 야코비안(J)과 잔차(e) 계산
        J, e = self._numerical_jacobian(vnode.mu)

        # 2. Information Matrix (Lambda) & Vector (eta) 계산
        # 정규 방정식: (J^T * Gamma^-1 * J) * delta_x = -J^T * Gamma^-1 * e
        Lambda = J.T @ self.gamma_inv @ J
        eta = -J.T @ self.gamma_inv @ e

        # 3. 메시지 전송
        edge._messages[vnode.name] = (Lambda, eta)


# ─────────────────────────────────────────────────────────────────────────────
# BinaryFNode  –  factor connected to TWO variable nodes
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFNode(_FNodeBase):
    """
    Pairwise (binary) factor node.
    두 변수 노드의 상태(mu_0, mu_1)에 대해 각각 편미분(Jacobian)을 수행하여,
    각각의 노드가 받아야 할 독립적인 정보(Lambda, eta)를 분배합니다.
    """

    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma)

    @abstractmethod
    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """ E = h(x0, x1) - y """
        pass

    def _numerical_jacobian(self, x0: np.ndarray, x1: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        e0 = self._error_function(x0, x1)
        m = len(e0)
        d0, d1 = len(x0), len(x1)
        J0 = np.zeros((m, d0))
        J1 = np.zeros((m, d1))
        
        # x0에 대한 야코비안
        for i in range(d0):
            x_p = x0.copy(); x_p[i] += eps
            x_m = x0.copy(); x_m[i] -= eps
            J0[:, i] = (self._error_function(x_p, x1) - self._error_function(x_m, x1)) / (2 * eps)
            
        # x1에 대한 야코비안
        for i in range(d1):
            x_p = x1.copy(); x_p[i] += eps
            x_m = x1.copy(); x_m[i] -= eps
            J1[:, i] = (self._error_function(x0, x_p) - self._error_function(x0, x_m)) / (2 * eps)
            
        return J0, J1, e0

    def compute_and_send(self) -> None:
        edge0, edge1 = self.edges
        vnode0 = self._get_vnode(edge0, self)
        vnode1 = self._get_vnode(edge1, self)

        # 1. 각 변수에 대한 야코비안과 공동 잔차 계산
        J0, J1, e = self._numerical_jacobian(vnode0.mu, vnode1.mu)

        # 2. 각 노드로 보낼 Information 메시지 계산
        Lambda_0 = J0.T @ self.gamma_inv @ J0
        eta_0 = -J0.T @ self.gamma_inv @ e
        
        Lambda_1 = J1.T @ self.gamma_inv @ J1
        eta_1 = -J1.T @ self.gamma_inv @ e

        # 3. 메시지 전송
        edge0._messages[vnode0.name] = (Lambda_0, eta_0)
        edge1._messages[vnode1.name] = (Lambda_1, eta_1)


# ─────────────────────────────────────────────────────────────────────────────
# FactorGraph  –  Sequential Runner
# ─────────────────────────────────────────────────────────────────────────────

class FactorGraph(Graph):
    """
    Factor graph that runs GaBP (Gauss-Newton optimization).
    이전 대화에 따라 스레딩을 제거하고 순차적(Sequential)으로 실행되도록 구성했습니다.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def vnodes(self) -> List[VNode]:
        return [n for n in self.nodes if isinstance(n, VNode)]

    @property
    def fnodes(self) -> List[_FNodeBase]:
        return [n for n in self.nodes if isinstance(n, _FNodeBase)]

    def iterate(self, n_iter: int = 1) -> None:
        """
        Run *n_iter* full message-passing iterations sequentially.
        """
        for _ in range(n_iter):
            # Step 1: Factor -> Variable (Compute Jacobians & Information messages)
            for fn in self.fnodes:
                fn.compute_and_send()

            # Step 2: Variable update (Solve Normal Equations)
            for vn in self.vnodes:
                vn.gabp_update()