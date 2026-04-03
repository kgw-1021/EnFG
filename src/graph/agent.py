import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.graph.DEKICore_vanilla import FactorGraph, VNode
from src.graph.factors import (
    GoalFactor, DynamicsFactor, CollisionFactor, GridObsFactor,
    VelocityConstraintFNode, ControlSmoothnessFNode, StartFactor,
)


# ------------------------------------------------------------------
# 공유 메모리에 저장되는 에이전트 정보
# ------------------------------------------------------------------
@dataclass
class AgentBelief:
    """
    에이전트가 공유 메모리에 게시하는 정보.

    [변경 1] 기존에는 mean (shape: horizon×4) 만 저장했으나,
    이제 sigma (스칼라 per timestep) 도 함께 저장합니다.

    sigma는 각 시간 스텝 VNode의 앙상블 표준편차의 평균:
        sigma_t = sqrt(Tr(C_xx_t) / d)
    이것만 추가하면 통신량은 기존 대비 horizon개의 스칼라만 늘어납니다.
    (mean: horizon×4, sigma: horizon×1 → 총 5*horizon 스칼라)

    CollisionFactorBinaryApprox는 이 sigma를 받아
    가상 앙상블 xi_j^n ≈ mu_j + N(0, sigma^2 I) 를 생성합니다.
    """
    mean: np.ndarray   # shape: (horizon, 4)
    sigma: np.ndarray  # shape: (horizon,)  — 시간 스텝별 앙상블 std


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        agent_id: int,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        n_particles: int = 100,
        horizon: int = 10,
        dt: float = 0.1,
        env_map=None,
        safe_dist: float = 0.5,
        collision_weight: float = 1e-4,
        dyn_weight: float = 1e-2,
        smooth_weight: float = 10.0,
        vel_weight: float = 10.0,
        obs_weight: float = 1e-1,
        start_goal_weight: float = 1e-4,
    ):
        self.id = agent_id
        self.horizon = horizon
        self.dt = dt
        self.goal_pos = goal_pos
        self.start_pos = start_pos
        self.env_map = env_map
        self.n_particles = n_particles
        self.safe_dist = safe_dist

        self.dyn_weight = dyn_weight
        self.smooth_weight = smooth_weight
        self.vel_weight = vel_weight
        self.obs_weight = obs_weight
        self.collision_weight = collision_weight
        self.start_goal_weight = start_goal_weight

        self.graph = FactorGraph()
        self.vnodes: List[VNode] = []

        # other_id → 해당 에이전트와의 CollisionFactor 리스트
        # [변경 3] detach 시 이 딕셔너리에서 제거하여 메모리 누수 방지
        self.collision_factors: Dict[int, List[CollisionFactor]] = {}

        self._build_graph()

    # ==============================================================
    # 그래프 초기 구성
    # ==============================================================
    def _make_vnode(self, t: int) -> VNode:
        """시간 스텝 t 에 해당하는 VNode를 생성하고 앙상블을 초기화합니다."""
        vnode = VNode(
            name=f"A{self.id}_t{t}",
            dims=[4],
            n_particles=self.n_particles,
            init_std=20.0,
            noise_std=1.0,
            rho_init=1.0,
            rho_update_method="residual",
        )
        alpha = t / max(1, self.horizon - 1)
        init_px = self.start_pos[0] * (1 - alpha) + self.goal_pos[0] * alpha
        init_py = self.start_pos[1] * (1 - alpha) + self.goal_pos[1] * alpha
        init_theta = (
            self.start_pos[2]
            if len(self.start_pos) > 2
            else np.arctan2(
                self.goal_pos[1] - self.start_pos[1],
                self.goal_pos[0] - self.start_pos[0],
            )
        )
        N = self.n_particles
        vnode.ensemble[0, :] = init_px    + np.random.randn(N) 
        vnode.ensemble[1, :] = init_py    + np.random.randn(N) 
        vnode.ensemble[2, :] = init_theta + np.random.randn(N) * 0.1
        vnode.ensemble[3, :] = 0.5        + np.random.randn(N) * 0.1
        return vnode

    def _build_graph(self):
        """VNode와 내부 팩터를 생성하여 팩터 그래프를 구성합니다."""
        # 1. VNode
        for t in range(self.horizon):
            vnode = self._make_vnode(t)
            self.graph.nodes.append(vnode)
            self.vnodes.append(vnode)

        # 2. 동역학 / 평활화 (Binary)
        for t in range(self.horizon - 1):
            dyn = DynamicsFactor(
                f"Dyn_A{self.id}_t{t}", dt=self.dt, weight=self.dyn_weight
            )
            self.graph.nodes.append(dyn)
            self.graph.connect(dyn, self.vnodes[t])
            self.graph.connect(dyn, self.vnodes[t + 1])

            smooth = ControlSmoothnessFNode(
                f"Smooth_A{self.id}_t{t}",
                dt=self.dt,
                w_smooth=self.smooth_weight,
                w_limit=self.smooth_weight,
            )
            self.graph.nodes.append(smooth)
            self.graph.connect(smooth, self.vnodes[t])
            self.graph.connect(smooth, self.vnodes[t + 1])

        # 3. 속도 제한 (Unary, 전 스텝)
        for t in range(self.horizon):
            vel = VelocityConstraintFNode(
                f"Vel_A{self.id}_t{t}",
                v_max=0.01,
                v_min=-0.005,
                weight=self.vel_weight,
            )
            self.graph.nodes.append(vel)
            self.graph.connect(vel, self.vnodes[t])

        # 4. 장애물 (Unary, 전 스텝)
        if self.env_map is not None:
            for t in range(self.horizon):
                obs = GridObsFactor(
                    name=f"BB_Obstacle_A{self.id}_t{t}",
                    occupancy_map_func=self.env_map.get_penalty,
                    weight=self.obs_weight,
                )
                self.graph.nodes.append(obs)
                self.graph.connect(obs, self.vnodes[t])

        # 5. 시작 / 목표 (Unary)
        start_f = StartFactor(
            f"Start_A{self.id}",
            start_pos=self.start_pos,
            weight=self.start_goal_weight,
        )
        self.graph.nodes.append(start_f)
        self.graph.connect(start_f, self.vnodes[0])

        goal_f = GoalFactor(
            f"Goal_A{self.id}",
            goal_pos=self.goal_pos,
            weight=self.start_goal_weight,
        )
        self.graph.nodes.append(goal_f)
        self.graph.connect(goal_f, self.vnodes[-1])

    def slide_horizon(self, new_start_pos: Optional[np.ndarray] = None):
        """
        [변경 2] RHC(Receding Horizon Control)처럼 horizon 윈도우를 한 스텝 앞으로 밉니다.

        동작 순서:
          1. vnodes[0] (실행 완료된 스텝)을 그래프에서 완전히 제거합니다.
             → 이 VNode에 연결된 내부 팩터 엣지도 함께 정리됩니다.
          2. vnodes 리스트를 한 칸 앞으로 당깁니다 (vnodes[1] → vnodes[0]).
          3. 새로운 마지막 VNode를 생성하여 추가합니다.
          4. 새 VNode와 기존 마지막 VNode 사이에 Dynamics / Smoothness 팩터를 연결합니다.
          5. StartFactor를 새로운 vnodes[0]으로 이전합니다.
          6. GoalFactor를 새로운 vnodes[-1]으로 이전합니다.
          7. CollisionFactor도 슬라이딩 후 VNode에 맞게 재연결합니다.

        Parameters
        ----------
        new_start_pos : np.ndarray, optional
            슬라이딩 후 vnodes[0]의 실제 로봇 위치.
            None이면 기존 vnodes[1]의 mean을 사용합니다.
        """
        # --- 슬라이딩 전 준비 ---
        old_first = self.vnodes[0]
        old_last  = self.vnodes[-1]

        # StartFactor, GoalFactor를 그래프에서 임시 제거
        self._remove_boundary_factors()

        # CollisionFactor를 그래프에서 임시 분리 (재연결 위해)
        detached_col = self._detach_all_collision_factors()

        # --- 1. vnodes[0] 제거 ---
        # old_first에 연결된 모든 엣지(팩터와의 연결)를 끊고 그래프에서 제거
        self.graph.remove_node(old_first)

        # --- 2. 리스트 슬라이딩 ---
        self.vnodes = self.vnodes[1:]

        # vnodes의 이름을 t 인덱스에 맞게 재정렬
        for t, vnode in enumerate(self.vnodes):
            vnode._name = f"A{self.id}_t{t}"

        # --- 3. 새 마지막 VNode 추가 ---
        new_t = self.horizon - 1
        new_vnode = self._make_vnode(new_t)
        self.graph.nodes.append(new_vnode)
        self.vnodes.append(new_vnode)

        # --- 4. 새 VNode와 기존 마지막 VNode 연결 ---
        prev_last = self.vnodes[-2]  # 슬라이딩 후 마지막에서 두 번째

        dyn = DynamicsFactor(
            f"Dyn_A{self.id}_t{new_t - 1}", dt=self.dt, weight=self.dyn_weight
        )
        self.graph.nodes.append(dyn)
        self.graph.connect(dyn, prev_last)
        self.graph.connect(dyn, new_vnode)

        smooth = ControlSmoothnessFNode(
            f"Smooth_A{self.id}_t{new_t - 1}",
            dt=self.dt,
            w_smooth=self.smooth_weight,
            w_limit=self.smooth_weight,
        )
        self.graph.nodes.append(smooth)
        self.graph.connect(smooth, prev_last)
        self.graph.connect(smooth, new_vnode)

        # 장애물 팩터도 새 VNode에 추가
        if self.env_map is not None:
            obs = GridObsFactor(
                name=f"BB_Obstacle_A{self.id}_t{new_t}",
                occupancy_map_func=self.env_map.get_penalty,
                weight=self.obs_weight,
            )
            self.graph.nodes.append(obs)
            self.graph.connect(obs, new_vnode)

        vel = VelocityConstraintFNode(
            f"Vel_A{self.id}_t{new_t}",
            v_max=0.01,
            v_min=-0.005,
            weight=self.vel_weight,
        )
        self.graph.nodes.append(vel)
        self.graph.connect(vel, new_vnode)

        # --- 5. StartFactor 재부착 (vnodes[0]) ---
        actual_start = (
            new_start_pos if new_start_pos is not None
            else self.vnodes[0].mean.flatten()
        )
        start_f = StartFactor(
            f"Start_A{self.id}",
            start_pos=actual_start,
            weight=self.start_goal_weight,
        )
        self.graph.nodes.append(start_f)
        self.graph.connect(start_f, self.vnodes[0])

        # --- 6. GoalFactor 재부착 (vnodes[-1]) ---
        goal_f = GoalFactor(
            f"Goal_A{self.id}",
            goal_pos=self.goal_pos,
            weight=self.start_goal_weight,
        )
        self.graph.nodes.append(goal_f)
        self.graph.connect(goal_f, self.vnodes[-1])

        # --- 7. CollisionFactor 재연결 ---
        # 슬라이딩으로 vnodes가 한 칸 밀렸으므로, 각 팩터를 새 VNode에 연결
        self._reattach_collision_factors(detached_col)

    def _remove_boundary_factors(self):
        """
        StartFactor와 GoalFactor를 그래프에서 찾아 제거합니다.
        slide_horizon 내부에서만 사용합니다.
        """
        to_remove = [
            n for n in self.graph.nodes
            if isinstance(n, (StartFactor, GoalFactor))
            and (f"A{self.id}" in n.name)
        ]
        for node in to_remove:
            self.graph.remove_node(node)

    def _detach_all_collision_factors(self) -> Dict[int, List[CollisionFactor]]:
        """
        현재 등록된 모든 CollisionFactor를 그래프에서 분리하고
        {other_id: [factors]} 딕셔너리를 반환합니다.
        실제 팩터 객체는 살려두고 엣지만 끊습니다.
        """
        snapshot = {}
        for other_id, factors in self.collision_factors.items():
            for factor in factors:
                # 엣지만 제거 (팩터 객체는 유지)
                for edge in list(factor.edges):
                    self.graph.remove_edge(edge)
                if factor in self.graph.nodes:
                    self.graph.nodes.remove(factor)
            snapshot[other_id] = factors
        self.collision_factors = {}
        return snapshot

    def _reattach_collision_factors(
        self, detached: Dict[int, List[CollisionFactor]]
    ):
        """
        slide_horizon 이후 슬라이딩된 VNode에 CollisionFactor를 재연결합니다.

        슬라이딩으로 vnodes가 한 칸 앞으로 밀렸으므로:
          - 기존 t=1..H-1 팩터 → vnodes[0..H-2] 에 재연결
          - 기존 t=0 팩터(이미 실행 완료)는 폐기
          - 새 t=H-1 자리에는 새 팩터를 생성하여 추가
        """
        for other_id, old_factors in detached.items():
            new_factors: List[CollisionFactor] = []

            # t=1..H-1 팩터를 vnodes[0..H-2]에 재연결
            for t, factor in enumerate(old_factors[1:], start=0):
                factor._name = f"Col_A{self.id}_A{other_id}_t{t}"
                self.graph.nodes.append(factor)
                self.graph.connect(factor, self.vnodes[t])
                new_factors.append(factor)

            # 새로운 마지막 스텝 팩터 생성
            new_t = self.horizon - 1
            new_factor = CollisionFactor(
                f"Col_A{self.id}_A{other_id}_t{new_t}",
                safe_dist=self.safe_dist,
                weight=self.collision_weight,
            )
            self.graph.nodes.append(new_factor)
            self.graph.connect(new_factor, self.vnodes[new_t])
            new_factors.append(new_factor)

            self.collision_factors[other_id] = new_factors

    # ==============================================================
    # 3. CollisionFactor 동적 부착 / 제거
    # ==============================================================
    def attach_collision_factor(self, other_id: int):
        """
        [변경 3] 특정 에이전트와의 CollisionFactor를 생성하고 그래프에 부착합니다.

        이미 부착되어 있으면 중복 생성하지 않습니다.
        """
        if other_id in self.collision_factors:
            return  # 이미 부착됨

        factors: List[CollisionFactor] = []
        for t in range(self.horizon):
            factor = CollisionFactor(
                f"Col_A{self.id}_A{other_id}_t{t}",
                safe_dist=self.safe_dist,
                weight=self.collision_weight,
            )
            self.graph.nodes.append(factor)
            self.graph.connect(factor, self.vnodes[t])
            factors.append(factor)

        self.collision_factors[other_id] = factors

    def detach_collision_factor(self, other_id: int):
        """
        [변경 3] 특정 에이전트와의 CollisionFactor를 그래프에서 완전히 제거합니다.

        graph.remove_node()를 사용하여:
          1. 팩터에 연결된 모든 엣지(VNode와의 연결)를 끊습니다.
          2. 팩터 노드를 graph.nodes 리스트에서 제거합니다.
          3. self.collision_factors 딕셔너리에서도 제거합니다.
             → 이후 어디서도 이 팩터를 참조하지 않으면 GC가 수거합니다.

        메모리 누수 방지 포인트:
          - graph.remove_node 내부에서 edge 양쪽 노드의 edges 리스트에서도 제거
          - collision_factors에서 del 하여 Agent가 가진 레퍼런스도 해제
        """
        if other_id not in self.collision_factors:
            return  # 이미 없음

        for factor in self.collision_factors[other_id]:
            self.graph.remove_node(factor)

        del self.collision_factors[other_id]

    # ==============================================================
    # 기존 메서드 (변경 없음 or 소폭 정리)
    # ==============================================================
    def update_external_beliefs(self, shared_trajectories: dict):
        """ 
        공유 메모리(Centralized Info)에서 상대방 궤적을 읽어와 내 Collision Factor에 주입 
        shared_trajectories 형태: { agent_id: np.ndarray shape (horizon, 4) }
        """
        for other_id, factors in self.collision_factors.items():
            if other_id in shared_trajectories:
                other_traj = shared_trajectories[other_id] 
                for t, factor in enumerate(factors):
                    # 충돌 회피 팩터에 상대방의 평균 위치를 업데이트
                    factor.other_pos_mean = other_traj[t]

    def get_mean_trajectory(self) -> np.ndarray:
        """현재 에이전트의 전체 궤적(평균값) 반환. Shape: (horizon, 4)"""
        return np.array([v.mean.flatten() for v in self.vnodes])

    def get_ensemble_data(self) -> np.ndarray:
        """
        전체 Time Horizon에 대한 앙상블 데이터 반환.
        Shape: (horizon, 4, n_particles)
        """
        return np.array([v.ensemble.copy() for v in self.vnodes])

    def inflate_ensemble(self, std: float = 1.0):
        """
        각 VNode의 앙상블을 현재 mean 주변으로 재팽창합니다.
        (기존 initialize_Vnodes → inflate_ensemble 로 이름 변경: 의미를 더 명확히)

        RHC 사이클마다 호출하여 수축된 앙상블의 탐색 능력을 복원합니다.
        theta에는 std * 0.1을 적용하여 운동학적 일관성을 유지합니다.
        """
        for vnode in self.vnodes:
            mean = vnode.mean.flatten()
            N = vnode.ensemble.shape[1]
            vnode.ensemble[0, :] = mean[0] + np.random.randn(N) * std
            vnode.ensemble[1, :] = mean[1] + np.random.randn(N) * std
            vnode.ensemble[2, :] = mean[2] + np.random.randn(N) * std * 0.1
            vnode.ensemble[3, :] = mean[3] + np.random.randn(N) * std * 0.5

    def step(self, iterations: int = 1):
        """EKI-ADMM Message Passing n스텝 수행"""
        self.graph.iterate(n_iter=iterations)