import numpy as np
from graph.EKIcore import UnaryFNode, BinaryFNode

class GoalFactor(UnaryFNode):
    """ 목표 지점 도달 제약 """
    def __init__(self, name: str, goal_pos: np.ndarray, weight: float = 1e-2):
        # 2D 좌표 에러이므로 gamma는 2x2
        gamma = np.eye(2) * weight
        super().__init__(name, dims=[2], gamma=gamma)
        self.goal_pos = goal_pos.reshape(2, 1)

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        # h(x) - y = 현재 위치 - 목표 위치
        return x[:2, :] - self.goal_pos

class DynamicsFactor(BinaryFNode):
    def __init__(self, name: str, dt: float, weight: float = 1e-4):
        gamma = np.eye(4) * weight
        super().__init__(name, dims=[4], gamma=gamma)
        self.dt = dt

    def _error_function(self, x_t: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        # x_t shape: (4, 100) -> [px, py, theta, v]
        px, py, theta, v = x_t
        
        # 비선형 운동학 예측 (등속도, 등조향 가정)
        pred_px = px + v * np.cos(theta) * self.dt
        pred_py = py + v * np.sin(theta) * self.dt
        pred_theta = theta # 다음 노드와의 차이(omega)는 Smoothness 팩터가 담당
        pred_v = v         # 다음 노드와의 차이(accel)는 Smoothness 팩터가 담당
        
        pred_next = np.stack([pred_px, pred_py, pred_theta, pred_v], axis=0) # (4, 100)
        
        # 에러 = 실제 x_next - 예측된 pred_next
        error = x_next - pred_next
        
        # 각도 에러(theta 차이)는 -pi ~ pi 사이로 랩핑(Wrap) 처리
        error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
        
        return error

class CollisionFactor(UnaryFNode):
    """ 
    [핵심] 공유 메모리에서 상대방 궤적을 읽어와 회피하는 분산형 충돌 팩터.
    같은 그래프 내부의 노드를 연결하는 것이 아니므로 Unary로 취급됩니다.
    """
    def __init__(self, name: str, safe_dist: float, weight: float = 1e-2):
        # 거리에 대한 에러 1D
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.safe_dist = safe_dist
        self.other_pos_mean = None # 매 스텝마다 Agent가 주입해줌

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        if self.other_pos_mean is None:
            return np.zeros((1, x.shape[1]))
            
        my_pos = x[:2, :]
        target_pos = self.other_pos_mean[:2].reshape(2, 1)
        
        # 유클리드 거리
        dists = np.linalg.norm(my_pos - target_pos, axis=0)
        
        # 에러 정의: 거리가 safe_dist보다 작으면 양수(에러 발생), 크면 0 (ReLU 형태)
        # h(x) = max(0, safe_dist - dist)
        error = np.maximum(0.0, self.safe_dist - dists)
        return error.reshape(1, -1)
    
class VelocityConstraintFNode(UnaryFNode):
    def __init__(self, name: str, v_max: float = 2.0, v_min: float = -0.5, weight: float = 1e-3):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.v_max = v_max
        self.v_min = v_min

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        # x shape: (4, 100) -> [px, py, theta, v]
        v = x[3, :] # 4번째 행(인덱스 3)이 전체 파티클의 속도
        
        err_upper = np.maximum(0.0, v - self.v_max)
        err_lower = np.minimum(0.0, v - self.v_min)
        
        error = err_upper + np.abs(err_lower) # Shape: (100,)
        
        # (obs_dim, N) 형태로 맞추기 위해 (1, 100)으로 변환
        return error.reshape(1, -1)
    
class ControlSmoothnessFNode(BinaryFNode):
    def __init__(self, name: str, dt: float = 0.1, 
                 w_smooth: float = 1e-1, w_limit: float = 1e-4):
        gamma = np.diag([w_smooth, w_smooth, w_limit, w_limit])
        super().__init__(name, dims=[4], gamma=gamma)
        
        self.dt = dt
        self.accel_max = 1.0  
        self.omega_max = 1.0  

    def _error_function(self, x_prev: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        # x_prev, x_next shape: (4, 100)
        dt = self.dt
        
        # 1. 선가속도 계산
        v_prev, v_next = x_prev[3, :], x_next[3, :]
        accel = (v_next - v_prev) / dt
        
        # 2. 각속도 계산
        theta_prev, theta_next = x_prev[2, :], x_next[2, :]
        d_theta = (theta_next - theta_prev + np.pi) % (2 * np.pi) - np.pi
        omega = d_theta / dt
        
        # 잔차 정의
        err_accel = accel
        err_omega = omega
        err_accel_lim = np.maximum(0.0, np.abs(accel) - self.accel_max)
        err_omega_lim = np.maximum(0.0, np.abs(omega) - self.omega_max)
        
        # axis=0 으로 쌓아야 (4, 100) 모양이 됨
        error = np.stack([err_accel, err_omega, err_accel_lim, err_omega_lim], axis=0) 
        
        return error
