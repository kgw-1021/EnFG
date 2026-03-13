import numpy as np
from src.graph.DEKICore import UnaryFNode, BinaryFNode

class StartFactor(UnaryFNode):
    """ 출발 지점(Initial State) 완전 고정 제약 """
    def __init__(self, name: str, start_pos: np.ndarray, weight: float = 1e-4):
        # start_pos가 [x, y, theta, v] 4차원이라고 가정
        dim = len(start_pos)
        gamma = np.eye(dim) * weight
        super().__init__(name, dims=[dim], gamma=gamma)
        self.start_pos = start_pos.reshape(dim, 1)

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        dim = self.start_pos.shape[0]
        error = x[:dim, :] - self.start_pos
        
        # 각도(theta)가 포함된 경우 랩핑(Wrapping) 처리 (-pi ~ pi)
        if dim >= 3:
            error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
            
        return error

class GoalFactor(UnaryFNode):
    """ 목표 지점 도달 제약 """
    def __init__(self, name: str, goal_pos: np.ndarray, weight: float = 1e-4):
        # goal_pos가 [x, y, theta, v] 4차원이라고 가정
        dim = len(goal_pos)
        gamma = np.eye(dim) * weight
        super().__init__(name, dims=[dim], gamma=gamma)
        self.goal_pos = goal_pos.reshape(dim, 1)

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        dim = self.goal_pos.shape[0]
        error = x[:dim, :] - self.goal_pos
        
        # 각도(theta)가 포함된 경우 랩핑(Wrapping) 처리 (-pi ~ pi)
        if dim >= 3:
            error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
            
        return error


class DynamicsFactor(BinaryFNode):
    """ 등속도/등조향 운동학 모델에 기반한 궤적 예측 제약 """
    def __init__(self, name: str, dt: float, weight: float = 1e-4):
        gamma = np.eye(4) * weight
        super().__init__(name, dims=[4], gamma=gamma)
        self.dt = dt

    def _error_function(self, x_t: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        px, py, theta, v = x_t
        
        pred_px = px + v * np.cos(theta) * self.dt
        pred_py = py + v * np.sin(theta) * self.dt
        pred_theta = theta 
        pred_v = v         
        
        pred_next = np.stack([pred_px, pred_py, pred_theta, pred_v], axis=0) 
        error = x_next - pred_next
        error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
        
        return error

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray):
        """
        [핵심] ADMM 가상 관측치를 위한 운동학적 타겟 투영
        mean0(x_t), mean1(x_{next}) -> Shape: (4,) [px, py, theta, v]
        """
        dt = self.dt
        px0, py0, theta0, v0 = mean0
        px1, py1, theta1, v1 = mean1

        # 1. z_target0: "미래 상태(mean1)를 역추적했을 때, 현재(mean0) 내가 있어야 할 위치" (Backward Projection)
        px0_targ = px1 - v1 * np.cos(theta1) * dt
        py0_targ = py1 - v1 * np.sin(theta1) * dt
        # 속도와 각도는 변하지 않는다고 가정(운동학 모델)하므로 상대방 값을 이상적 타겟으로 사용
        z_target0 = np.array([px0_targ, py0_targ, theta1, v1])

        # 2. z_target1: "현재 상태(mean0)를 밀어봤을 때, 미래(mean1) 네가 있어야 할 위치" (Forward Projection)
        px1_targ = px0 + v0 * np.cos(theta0) * dt
        py1_targ = py0 + v0 * np.sin(theta0) * dt
        z_target1 = np.array([px1_targ, py1_targ, theta0, v0])

        return z_target0, z_target1


class CollisionFactor(UnaryFNode):
    """ 분산형 충돌 회피 팩터 (공유 메모리 활용) """
    def __init__(self, name: str, safe_dist: float, weight: float = 1e-2):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.safe_dist = safe_dist
        self.other_pos_mean = None 

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        if self.other_pos_mean is None:
            return np.zeros((1, x.shape[1]))
            
        my_pos = x[:2, :]
        target_pos = self.other_pos_mean[:2].reshape(2, 1)
        
        dists = np.linalg.norm(my_pos - target_pos, axis=0)
        error = np.maximum(0.0, self.safe_dist - dists)
        return error.reshape(1, -1)


class VelocityConstraintFNode(UnaryFNode):
    """ 속도 한계 제약 """
    def __init__(self, name: str, v_max: float = 2.0, v_min: float = -0.5, weight: float = 1e-3):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.v_max = v_max
        self.v_min = v_min

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        v = x[3, :] 
        err_upper = np.maximum(0.0, v - self.v_max)
        err_lower = np.minimum(0.0, v - self.v_min)
        error = err_upper + np.abs(err_lower) 
        return error.reshape(1, -1)


class ControlSmoothnessFNode(BinaryFNode):
    """ 가속도/각속도 제한 및 궤적 부드러움 제약 """
    def __init__(self, name: str, dt: float = 0.1, 
                 w_smooth: float = 1e-1, w_limit: float = 1e-4):
        gamma = np.diag([w_smooth, w_smooth, w_limit, w_limit])
        super().__init__(name, dims=[4], gamma=gamma)
        
        self.dt = dt
        self.accel_max = 1.0  
        self.omega_max = 1.0  

    def _error_function(self, x_prev: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        dt = self.dt
        v_prev, v_next = x_prev[3, :], x_next[3, :]
        accel = (v_next - v_prev) / dt
        
        theta_prev, theta_next = x_prev[2, :], x_next[2, :]
        d_theta = (theta_next - theta_prev + np.pi) % (2 * np.pi) - np.pi
        omega = d_theta / dt
        
        err_accel = accel
        err_omega = omega
        err_accel_lim = np.maximum(0.0, np.abs(accel) - self.accel_max)
        err_omega_lim = np.maximum(0.0, np.abs(omega) - self.omega_max)
        
        error = np.stack([err_accel, err_omega, err_accel_lim, err_omega_lim], axis=0) 
        return error

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray):
        """
        [핵심] 궤적 평활화(Smoothness)를 위한 ADMM 타겟
        가속도와 각속도를 최소화(=0)하기 위해 이웃 노드 간의 속도와 각도를 일치시키도록 타겟을 제공합니다.
        """
        # 위치(px, py)는 평활화 제약에 해당하지 않으므로 자기 자신의 현재 위치를 타겟으로 반환 (ADMM 간섭 배제)
        z_target0 = mean0.copy()
        z_target1 = mean1.copy()
        
        # 가속도가 0에 수렴하는 가장 이상적인 상태 -> 두 노드의 속도가 완전히 같아지는 것 (평균점 합의)
        avg_v = (mean0[3] + mean1[3]) / 2.0
        
        # 각속도가 0에 수렴하는 가장 이상적인 상태 -> 두 노드의 각도가 같아지는 것
        # 랩핑(Wrap) 문제를 방지하기 위해 삼각함수를 통한 원형 평균(Circular Mean) 계산
        sin_mean = (np.sin(mean0[2]) + np.sin(mean1[2])) / 2.0
        cos_mean = (np.cos(mean0[2]) + np.cos(mean1[2])) / 2.0
        avg_theta = np.arctan2(sin_mean, cos_mean)
        
        # 합의된 속도와 각도를 타겟으로 주입
        z_target0[2], z_target0[3] = avg_theta, avg_v
        z_target1[2], z_target1[3] = avg_theta, avg_v
        
        return z_target0, z_target1
    

class GridObsFactor(UnaryFNode):
    """ 
    미분이 불가능한(Non-differentiable) 이진 점유 맵 또는 블랙박스 장애물 팩터.
    기울기(Gradient)나 ADMM 기하학적 투영 없이 오직 EKI의 공분산(Covariance)만으로 회피합니다.
    """
    def __init__(self, name: str, occupancy_map_func, weight: float = 1e-3):
        # 방향성(x,y) 없이 오직 '위험도'라는 1차원 스칼라 값만 에러로 뱉습니다.
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.occupancy_map_func = occupancy_map_func

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[1]
        error = np.zeros((1, N))
        
        for i in range(N):
            pos_x, pos_y = x[0, i], x[1, i]
            
            # 기울기를 구하지 않고, 단순히 현재 위치의 페널티(위험도)만 읽어옵니다.
            # 목표는 이 페널티 값이 0이 되는 것입니다.
            penalty = self.occupancy_map_func(pos_x, pos_y)
            error[0, i] = penalty
            
        return error