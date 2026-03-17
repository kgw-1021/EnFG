from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np

from src.graph.graph import Node, Edge, Graph

# ─────────────────────────────────────────────────────────────────────────────
# 1. VNode – Variable Node (Cavity Message 적용)
# ─────────────────────────────────────────────────────────────────────────────

class VNode(Node):
    """
    Gaussian Belief Propagation을 위한 Variable Node.
    
    [개선 사항]
    - 수신된 메시지를 팩터별로 독립적으로 저장합니다 (`self.messages`).
    - 특정 팩터에게 메시지를 보낼 때, 그 팩터가 준 정보를 제외하는 Cavity Message를 생성합니다.
    """

    def __init__(
        self,
        name: str,
        dims: list,
        prior_std: float = 1.0,
        prior_mu: Optional[np.ndarray] = None
    ) -> None:
        super().__init__(name, dims)
        d = int(np.prod(dims)) if dims else 1

        # 1. 초기 Prior 설정 (정보 행렬 Lambda, 정보 벡터 eta)
        self.prior_Lambda: np.ndarray = np.eye(d) / (prior_std ** 2)
        self.prior_mu: np.ndarray = prior_mu if prior_mu is not None else np.zeros(d)
        self.prior_eta: np.ndarray = self.prior_Lambda @ self.prior_mu

        # 2. 현재 상태
        self.mu: np.ndarray = self.prior_mu.copy()
        self.Sigma: np.ndarray = np.eye(d) * (prior_std ** 2)

        # 3. 팩터로부터 받은 메시지를 저장하는 딕셔너리
        # 형태: { factor_name: (Lambda_msg, eta_msg) }
        self.messages: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_cavity_message(self, target_factor_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        대상 팩터(target_factor)를 제외한 나머지 모든 정보의 합(Cavity)을 반환합니다.
        이는 정보의 중복(Double-counting)을 막는 GaBP의 핵심입니다.
        """
        L_cav = self.prior_Lambda.copy()
        eta_cav = self.prior_eta.copy()

        for fname, (L_msg, eta_msg) in self.messages.items():
            if fname != target_factor_name:
                L_cav += L_msg
                eta_cav += eta_msg

        return L_cav, eta_cav

    def update_belief(self) -> None:
        """
        모든 팩터로부터의 메시지와 Prior를 합산하여 최종 상태(mu, Sigma)를 업데이트합니다.
        """
        L_marg = self.prior_Lambda.copy()
        eta_marg = self.prior_eta.copy()

        for L_msg, eta_msg in self.messages.values():
            L_marg += L_msg
            eta_marg += eta_msg

        # 정보 행렬(Lambda)을 공분산(Sigma)으로 변환하고 평균(mu) 업데이트
        try:
            self.Sigma = np.linalg.inv(L_marg)
        except np.linalg.LinAlgError:
            self.Sigma = np.linalg.pinv(L_marg)
            
        self.mu = self.Sigma @ eta_marg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Factor Nodes (Marginalization 적용)
# ─────────────────────────────────────────────────────────────────────────────

class _FNodeBase(Node, ABC):
    def __init__(self, name: str, gamma: np.ndarray) -> None:
        super().__init__(name, dims=[])
        self.gamma = gamma
        self.gamma_inv = np.linalg.inv(gamma)
        
    def _get_vnode(self, edge: Edge) -> VNode:
        v0, v1 = edge._node0, edge._node1
        return v0 if v0.name != self.name else v1

    @abstractmethod
    def compute_and_send(self) -> None:
        pass


class UnaryFNode(_FNodeBase, ABC):
    """ 단일 변수에 연결된 팩터 (예: GPS Anchor) """
    @abstractmethod
    def error(self, x: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray: pass

    def compute_and_send(self) -> None:
        v0 = self._get_vnode(self.edges[0])
        
        # 1. 현재 mu에서 Jacobian 및 Error 계산
        J0 = self.jacobian(v0.mu)
        e = self.error(v0.mu)

        # 2. H (Hessian) 및 g (Gradient) 계산
        H00 = J0.T @ self.gamma_inv @ J0
        g0 = J0.T @ self.gamma_inv @ e

        # 3. Absolute Information Vector 계산 (상대적 증분이 아닌 절대 좌표 기준)
        eta0_abs = H00 @ v0.mu - g0

        # 4. 변수 노드에 메시지 전송 (단일 노드이므로 주변화 필요 없음)
        v0.messages[self.name] = (H00, eta0_abs)


class BinaryFNode(_FNodeBase, ABC):
    """ 두 변수를 연결하는 팩터 (예: 거리 측정, Odometry) """
    @abstractmethod
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: pass

    def compute_and_send(self) -> None:
        v0 = self._get_vnode(self.edges[0])
        v1 = self._get_vnode(self.edges[1])

        # 1. 변수 노드들로부터 Cavity Message 수신 (내가 예전에 준 정보는 뺀 상태)
        L0_cav, eta0_cav = v0.get_cavity_message(self.name)
        L1_cav, eta1_cav = v1.get_cavity_message(self.name)

        # 2. 현재 상태에서 에러 및 자코비안 평가
        J0, J1 = self.jacobians(v0.mu, v1.mu)
        e = self.error(v0.mu, v1.mu)

        # 3. 정보 행렬 블록 (H) 및 그래디언트 (g) 계산
        H00 = J0.T @ self.gamma_inv @ J0
        H11 = J1.T @ self.gamma_inv @ J1
        H01 = J0.T @ self.gamma_inv @ J1
        H10 = J1.T @ self.gamma_inv @ J0

        g0 = J0.T @ self.gamma_inv @ e
        g1 = J1.T @ self.gamma_inv @ e

        # 4. 팩터 자체의 절대 정보 벡터 (Absolute Information Vector)
        eta_F0 = H00 @ v0.mu + H01 @ v1.mu - g0
        eta_F1 = H10 @ v0.mu + H11 @ v1.mu - g1

        # ─────────────────────────────────────────────────────────
        # 5. Schur Complement를 이용한 Marginalization (핵심 부분)
        # ─────────────────────────────────────────────────────────
        
        # [ 메시지 F -> v0 ] : v1을 주변화(Marginalize out)하여 v0에게 전달
        try:
            M1 = np.linalg.inv(H11 + L1_cav)
        except np.linalg.LinAlgError:
            M1 = np.linalg.pinv(H11 + L1_cav)

        L_F_to_0 = H00 - H01 @ M1 @ H10
        eta_F_to_0 = eta_F0 - H01 @ M1 @ (eta_F1 + eta1_cav)

        # [ 메시지 F -> v1 ] : v0를 주변화(Marginalize out)하여 v1에게 전달
        try:
            M0 = np.linalg.inv(H00 + L0_cav)
        except np.linalg.LinAlgError:
            M0 = np.linalg.pinv(H00 + L0_cav)

        L_F_to_1 = H11 - H10 @ M0 @ H01
        eta_F_to_1 = eta_F1 - H10 @ M0 @ (eta_F0 + eta0_cav)

        # 6. 계산된 메시지를 각각의 변수 노드에 저장
        v0.messages[self.name] = (L_F_to_0, eta_F_to_0)
        v1.messages[self.name] = (L_F_to_1, eta_F_to_1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FactorGraph Runner
# ─────────────────────────────────────────────────────────────────────────────

class FactorGraph(Graph):
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
        GaBP 메시지 패싱 루프
        """
        for _ in range(n_iter):
            # Step 1: 팩터 노드들이 변수들의 Cavity Message를 받아 계산 후 메시지 전송
            for fn in self.fnodes:
                fn.compute_and_send()

            # Step 2: 변수 노드들이 취합된 메시지를 바탕으로 Belief(mu, Sigma) 업데이트
            for vn in self.vnodes:
                vn.update_belief()