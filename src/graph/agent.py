import numpy as np
from graph.EKIcore import FactorGraph, VNode
from src.graph.factors import GoalFactor, DynamicsFactor, CollisionFactor, VelocityConstraintFNode, ControlSmoothnessFNode

class Agent:
    def __init__(self, agent_id: int, start_pos: np.ndarray, goal_pos: np.ndarray, horizon: int = 10, dt: float = 0.1):
        self.id = agent_id
        self.horizon = horizon
        self.dt = dt
        self.graph = FactorGraph(max_workers=2) 
        
        self.vnodes = []
        self.collision_factors = {} 
        
        # ==========================================================
        # 1. 상태 변수 노드(VNode) 생성: [px, py, theta, v]
        # ==========================================================
        for t in range(horizon):
            vnode = VNode(name=f"A{self.id}_t{t}", dims=[4], n_particles=100)
            
            # 파티클 초기화
            vnode.ensemble[0, :] = start_pos[0] + np.random.randn(100) * 0.1 # px
            vnode.ensemble[1, :] = start_pos[1] + np.random.randn(100) * 0.1 # py
            vnode.ensemble[2, :] = start_pos[2] if len(start_pos) > 2 else 0.0 # theta
            vnode.ensemble[3, :] = 0.0 # 초기 속도 v = 0
            
            self.vnodes.append(vnode)
            self.graph.nodes.append(vnode)

            # ------------------------------------------------------
            # [추가 1] 속도 제약 팩터 부착 (Unary Factor: 각 VNode마다 1개씩)
            # ------------------------------------------------------
            vel_factor = VelocityConstraintFNode(
                name=f"Vel_A{self.id}_t{t}", 
                v_max=2.0, 
                v_min=-0.5, 
                weight=1e-3  # 속도 제한은 꽤 엄격하게 지켜야 하므로 작게 설정
            )
            # Unary는 해당 변수 노드와 팩터를 1:1로 연결합니다.
            self.graph.connect(vnode, vel_factor)
            self.graph.nodes.append(vel_factor)
            
        # ==========================================================
        # 2. 시간(t)의 흐름에 따른 기구학 및 제어 부드러움 팩터 부착
        # ==========================================================
        for t in range(horizon - 1):
            v_curr = self.vnodes[t]
            v_next = self.vnodes[t+1]
            
            # ------------------------------------------------------
            # [수정] 기구학 팩터 (Binary Factor) 올바른 연결 방법
            # ------------------------------------------------------
            dyn_factor = DynamicsFactor(f"Dyn_A{self.id}_t{t}", dt=self.dt, weight=1e-4)
            # Binary 팩터는 자신을 중심으로 t와 t+1 노드 양쪽에 엣지를 생성해야 합니다!
            self.graph.connect(v_curr, dyn_factor)
            self.graph.connect(v_next, dyn_factor)
            self.graph.nodes.append(dyn_factor)
            
            # ------------------------------------------------------
            # [추가 2] 제어 부드러움 팩터 (Binary Factor: t와 t+1 사이)
            # ------------------------------------------------------
            smooth_factor = ControlSmoothnessFNode(
                name=f"Smooth_A{self.id}_t{t}", 
                dt=self.dt,
                w_smooth=1e-3,   # 부드러움은 약간 느슨하게
                w_limit=1e-4    # 한계 돌파 방지는 엄격하게
            )
            # 역시 Binary이므로 양쪽 노드를 팩터에 연결합니다.
            self.graph.connect(v_curr, smooth_factor)
            self.graph.connect(v_next, smooth_factor)
            self.graph.nodes.append(smooth_factor)

        # ==========================================================
        # 3. 목표 지점 도달 팩터 부착
        # ==========================================================
        # Horizon의 마지막(예측의 끝부분)에만 부착하여 그쪽으로 끌어당기게 함
        goal_factor = GoalFactor(f"Goal_A{self.id}", goal_pos, weight=5e-3)
        self.graph.connect(self.vnodes[-1], goal_factor)
        self.graph.nodes.append(goal_factor)

    def attach_collision_factor(self, other_id: int):
        """ 타 에이전트와의 충돌 회피 팩터 생성 (각 Time step별로 부착) """
        factors = []
        for t in range(self.horizon):
            factor = CollisionFactor(f"Col_A{self.id}_A{other_id}_t{t}", safe_dist=1.0, weight=0.01)
            self.graph.connect(self.vnodes[t], factor)
            self.graph.nodes.append(factor)
            factors.append(factor)
            
        self.collision_factors[other_id] = factors

    def update_external_beliefs(self, shared_trajectories: dict):
        """ 공유 메모리에서 상대방 궤적을 읽어와 내 Collision Factor에 주입 """
        for other_id, factors in self.collision_factors.items():
            if other_id in shared_trajectories:
                other_traj = shared_trajectories[other_id] # shape: (horizon, state_dim)
                for t, factor in enumerate(factors):
                    factor.other_pos_mean = other_traj[t]

    def extract_trajectory(self) -> np.ndarray:
        return np.array([v.mean for v in self.vnodes])

    def step(self, iterations: int = 3):
        """ EKI Message Passing 수행 """
        self.graph.iterate(n_iter=iterations)
        
        # MPC Shift (시간이 1스텝 지났으므로 궤적을 앞으로 당김)
        for t in range(self.horizon - 1):
            self.vnodes[t].ensemble = self.vnodes[t+1].ensemble.copy()
            
        # 마지막 노드는 복사본에 약간의 노이즈를 주어 탐색을 유도
        self.vnodes[-1].ensemble += np.random.randn(*self.vnodes[-1].ensemble.shape) * 0.1