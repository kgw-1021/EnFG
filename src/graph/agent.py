import numpy as np

# DEKICore 모듈로 변경 (경로는 실제 프로젝트 구조에 맞게 수정하세요)
from src.graph.DEKICore import FactorGraph, VNode

# 작성해주신 새로운 팩터들 임포트
from src.graph.factors import (
    GoalFactor, DynamicsFactor, CollisionFactor, GridObsFactor,
    VelocityConstraintFNode, ControlSmoothnessFNode, StartFactor
    )

class Agent:
    def __init__(self, agent_id: int, start_pos: np.ndarray, goal_pos: np.ndarray, n_particles: int = 100, horizon: int = 10, dt: float = 0.1, env_map=None,
                 safe_dist: float = 0.5, collision_weight: float = 1e-4, dyn_weight: float = 1e-4, smooth_weight: float = 1e-1, vel_weight: float = 1e-3, obs_weight: float = 1e-4, start_goal_weight: float = 1e-3):
    
        self.id = agent_id
        self.horizon = horizon
        self.dt = dt
        self.goal_pos = goal_pos
        self.start_pos = start_pos
        self.env_map = env_map
        self.n_particles = n_particles
        # D-EKI를 위한 Factor Graph 초기화
        self.graph = FactorGraph(max_workers=2) 
        self.safe_dist = safe_dist

        self.dyn_weight = dyn_weight
        self.smooth_weight = smooth_weight
        self.vel_weight = vel_weight
        self.obs_weight = obs_weight
        self.collision_weight = collision_weight
        self.start_goal_weight = start_goal_weight
        
        self.vnodes = []
        self.collision_factors = {} 
        
        # ==========================================================
        # 1. 상태 변수 노드(VNode) 생성: [px, py, theta, v]
        # ==========================================================
        for t in range(horizon):
            # [수정] ADMM 합의를 위한 rho_init 및 rho_update_method(공분산 기반) 추가
            vnode = VNode(
                name=f"A{self.id}_t{t}", 
                dims=[4], 
                n_particles=n_particles,
                init_std=10.0,
                noise_std=1e-2,
                rho_init=1.0, 
                rho_update_method='residual'
            )
            
            # 파티클 초기화 (Start -> Goal 로의 선형 보간을 초기 추정치로 주면 수렴이 훨씬 빠름)
            alpha = t / max(1, horizon - 1)
            init_px = start_pos[0] * (1 - alpha) + goal_pos[0] * alpha
            init_py = start_pos[1] * (1 - alpha) + goal_pos[1] * alpha
            init_theta = start_pos[2] if len(start_pos) > 2 else np.arctan2(goal_pos[1]-start_pos[1], goal_pos[0]-start_pos[0])
            
            vnode.ensemble[0, :] = init_px + np.random.randn(n_particles) * 0.5    # px
            vnode.ensemble[1, :] = init_py + np.random.randn(n_particles) * 0.5    # py
            vnode.ensemble[2, :] = init_theta + np.random.randn(n_particles) * 0.1 # theta
            vnode.ensemble[3, :] = 0.5 + np.random.randn(n_particles) * 0.1        # v (초기 속도 약간 부여)
            
            self.graph.nodes.append(vnode)
            self.vnodes.append(vnode)

        # ==========================================================
        # 2. 내부 팩터(Internal Factors) 부착
        # ==========================================================
        
        # 2-1. Dynamics & Smoothness Factors (시간 t 와 t+1 을 연결)
        for t in range(horizon - 1):
            # 운동학 모델 팩터
            dyn_factor = DynamicsFactor(f"Dyn_A{self.id}_t{t}", dt=self.dt, weight=1e-4)
            self.graph.nodes.append(dyn_factor)
            self.graph.connect(dyn_factor, self.vnodes[t])
            self.graph.connect(dyn_factor, self.vnodes[t+1])
            
            # 제어 평활화(가속도/각속도 제한) 팩터
            smooth_factor = ControlSmoothnessFNode(f"Smooth_A{self.id}_t{t}", dt=self.dt, w_smooth=smooth_weight, w_limit=smooth_weight)
            self.graph.nodes.append(smooth_factor)
            self.graph.connect(smooth_factor, self.vnodes[t])
            self.graph.connect(smooth_factor, self.vnodes[t+1])

        # 2-2. 속도 제한 팩터 (모든 시간 스텝의 단일 노드에 적용)
        for t in range(horizon):
            vel_factor = VelocityConstraintFNode(f"Vel_A{self.id}_t{t}", v_max=0.01, v_min=-0.005, weight=vel_weight)
            self.graph.nodes.append(vel_factor)
            self.graph.connect(vel_factor, self.vnodes[t])

        if env_map is not None:
            for t in range(horizon):
                # 앞서 병렬화 최적화가 적용된 BlackBoxGridFactor 사용
                bb_factor = GridObsFactor(
                    name=f"BB_Obstacle_A{self.id}_t{t}", 
                    occupancy_map_func=env_map.get_penalty, 
                    weight=obs_weight
                )
                self.graph.nodes.append(bb_factor)
                self.graph.connect(bb_factor, self.vnodes[t])

        start_factor = StartFactor(f"Start_A{self.id}", start_pos=self.start_pos, weight=start_goal_weight)
        self.graph.nodes.append(start_factor)
        self.graph.connect(start_factor, self.vnodes[0])

        # 2-3. 목표 지점 팩터 (마지막 시간 스텝 노드에만 적용)
        goal_factor = GoalFactor(f"Goal_A{self.id}", goal_pos=self.goal_pos, weight=start_goal_weight)
        self.graph.nodes.append(goal_factor)
        self.graph.connect(goal_factor, self.vnodes[-1])

    # ==========================================================
    # 3. 외부 팩터(External Factors - 멀티 로봇 상호작용) 부착
    # ==========================================================
    def attach_collision_factor(self, other_id: int):
        """ 타 에이전트와의 충돌 회피 팩터 생성 (각 Time step별로 부착) """
        factors = []
        for t in range(self.horizon):
            factor = CollisionFactor(f"Col_A{self.id}_A{other_id}_t{t}", safe_dist=self.safe_dist, weight=self.collision_weight)
            self.graph.nodes.append(factor)
            self.graph.connect(factor, self.vnodes[t])
            factors.append(factor)
            
        self.collision_factors[other_id] = factors

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

    def extract_trajectory(self) -> np.ndarray:
        """ 현재 에이전트의 전체 궤적(평균값) 반환 -> Shape: (horizon, 4) """
        return np.array([v.mean.flatten() for v in self.vnodes])
    
    def get_ensemble_data(self) -> np.ndarray:
        """
        현재 에이전트의 전체 Time Horizon에 대한 앙상블(파티클) 데이터를 추출하여 반환합니다.
        외부에서 히스토리를 저장하거나 시각화할 때 사용합니다.
        
        Returns:
            np.ndarray: Shape (horizon, 4, n_particles) -> 4는 [x, y, theta, v]
        """
        return np.array([v.ensemble.copy() for v in self.vnodes])

    def initialize_Vnodes(self, std: float = 1.0):
        """
        외부 ADMM 루프마다 각 VNode의 앙상블을 현재 mean 주변으로 재초기화합니다.
        내부 EKI 반복으로 수축한 공분산을 강제로 확장해,
        새로운 외부 정보(타 에이전트 궤적)에 반응할 여지를 만듭니다.

        앙상블 = mean + N(0, std²)
        위치(px, py)와 속도(v)에만 노이즈를 주고,
        각도(theta)는 작은 std로 따로 처리해 운동학적 일관성을 유지합니다.
        """
        for vnode in self.vnodes:
            mean = vnode.mean.flatten()          # (4,)
            N    = vnode.ensemble.shape[1]       # 파티클 수

            vnode.ensemble[0, :] = mean[0] + np.random.randn(N) * std        # px
            vnode.ensemble[1, :] = mean[1] + np.random.randn(N) * std        # py
            vnode.ensemble[2, :] = mean[2] + np.random.randn(N) * std * 0.1  # theta (작게)
            vnode.ensemble[3, :] = mean[3] + np.random.randn(N) * std * 0.5  # v

    def step(self, iterations: int = 1):
        """ EKI-ADMM Message Passing 1스텝(또는 n스텝) 수행 """
        self.graph.iterate(n_iter=iterations)