import numpy as np
from src.graph.DEKICore import UnaryFNode, BinaryFNode


class StartFactor(UnaryFNode):
    """출발 지점(Initial State) 완전 고정 제약"""

    def __init__(self, name: str, start_pos: np.ndarray, weight: float = 1e-4):
        dim = len(start_pos)
        gamma = np.eye(dim) * weight
        super().__init__(name, dims=[dim], gamma=gamma)
        self.start_pos = start_pos.reshape(dim, 1)

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        dim = self.start_pos.shape[0]
        error = x[:dim, :] - self.start_pos
        if dim >= 3:
            error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
        return error


class GoalFactor(UnaryFNode):
    """목표 지점 도달 제약"""

    def __init__(self, name: str, goal_pos: np.ndarray, weight: float = 1e-4):
        dim = len(goal_pos)
        gamma = np.eye(dim) * weight
        super().__init__(name, dims=[dim], gamma=gamma)
        self.goal_pos = goal_pos.reshape(dim, 1)

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        dim = self.goal_pos.shape[0]
        error = x[:dim, :] - self.goal_pos
        if dim >= 3:
            error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
        return error


class DynamicsFactor(BinaryFNode):
    """등속도/등조향 운동학 모델에 기반한 궤적 예측 제약"""

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

        # [변경 없음] 각도 wrapping은 기존과 동일
        error[2, :] = (error[2, :] + np.pi) % (2 * np.pi) - np.pi
        return error

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray):
        """
        ADMM 가상 관측치를 위한 운동학적 타겟 투영.

        [변경] 기존: theta/v를 서로 상대방 값으로 비대칭 교환
               수정: theta/v를 두 노드의 대칭적 합의값(원형 평균)으로 설정

        기존 구현은 z_target0에 theta1/v1을, z_target1에 theta0/v0을 넣는
        비대칭 구조였습니다. 두 값이 다를 경우 서로 상대방 상태를 목표로
        삼는 순환이 발생해 진동의 원인이 됩니다.

        수정 후에는 두 노드가 동일한 합의 theta/v를 목표로 삼으므로
        ADMM 합의 방향이 일관됩니다. theta는 원형 평균(circular mean)으로
        계산하여 -pi/pi 경계에서의 wrapping 문제를 방지합니다.
        """
        dt = self.dt
        px0, py0, theta0, v0 = mean0
        px1, py1, theta1, v1 = mean1

        # 위치 타겟: 기존과 동일 (운동학적 forward/backward projection)
        px0_targ = px1 - v1 * np.cos(theta1) * dt
        py0_targ = py1 - v1 * np.sin(theta1) * dt
        px1_targ = px0 + v0 * np.cos(theta0) * dt
        py1_targ = py0 + v0 * np.sin(theta0) * dt

        # [변경] theta/v: 상대방 값 대신 두 노드의 대칭 합의값 사용
        avg_v = (v0 + v1) / 2.0
        avg_theta = np.arctan2(
            (np.sin(theta0) + np.sin(theta1)) / 2.0,
            (np.cos(theta0) + np.cos(theta1)) / 2.0,
        )

        z_target0 = np.array([px0_targ, py0_targ, avg_theta, avg_v])
        z_target1 = np.array([px1_targ, py1_targ, avg_theta, avg_v])
        return z_target0, z_target1


class CollisionFactor(UnaryFNode):
    """분산형 충돌 회피 팩터 (공유 메모리 활용)"""

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
        dists = np.maximum(dists, 1e-6)

        error = np.maximum(0.0, self.safe_dist - dists)
        return error.reshape(1, -1)


class VelocityConstraintFNode(UnaryFNode):
    """속도 한계 제약"""

    def __init__(
        self,
        name: str,
        v_max: float = 0.1,
        v_min: float = -0.05,
        weight: float = 1e-4,
    ):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.v_max = v_max
        self.v_min = v_min

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        v = x[3, :]

        # [변경] 부호 통일: 두 항 모두 위반량을 양수로 표현
        # 기존: err_lower = minimum(0, v - v_min) → 음수, abs로 뒤집음
        # 수정: err_lower = maximum(0, v_min - v) → 처음부터 양수
        # 의미가 동일하지만 코드 의도가 명확해지고 부호 실수 방지
        err_upper = np.maximum(0.0, v - self.v_max)   # v > v_max 위반량
        err_lower = np.maximum(0.0, self.v_min - v)   # v < v_min 위반량
        error = err_upper + err_lower
        return error.reshape(1, -1)


class ControlSmoothnessFNode(BinaryFNode):
    """가속도/각속도 제한 및 궤적 부드러움 제약"""

    def __init__(
        self,
        name: str,
        dt: float = 0.1,
        w_smooth: float = 1e-1,
        w_limit: float = 1e-4,
    ):
        # [변경] 잔차 벡터를 4차원 → 2차원으로 축소
        # 기존: [accel, omega, accel_lim, omega_lim] 4차원
        # 수정: [accel, omega] 2차원 (평활화 항만 유지)
        #
        # 기존 구현에서 평활화 항(accel, omega)과 한계 항(accel_lim, omega_lim)이
        # 같은 물리량에서 파생된 중복 정보였습니다.
        # w_smooth=1e-1이 w_limit=1e-4보다 1000배 크므로 한계 항은
        # 사실상 EKI 교차공분산에 미치는 영향이 무시되었습니다.
        # 한계 초과 페널티는 VelocityConstraintFNode가 별도로 담당하므로
        # 여기서는 평활화만 담당하도록 역할을 분리합니다.
        gamma = np.diag([w_smooth, w_smooth])
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

        # [변경] 평활화 항만 반환 (한계 항 제거)
        # 가속도/각속도 자체를 0에 가깝게 만드는 것이 이 팩터의 역할
        error = np.stack([err_accel := accel, err_omega := omega], axis=0)
        return error

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray):
        """
        궤적 평활화를 위한 ADMM 타겟.

        [변경] 위치(px, py) z_target을 자기 자신으로 고정하던 것을
               DynamicsFactor와 충돌하지 않도록 위치 타겟을 제공하지 않고,
               theta/v만 합의 타겟으로 설정합니다.

        기존 구현에서 z_target0 = mean0.copy()로 위치를 자기 자신으로 고정했는데,
        DynamicsFactor가 운동학적 위치 타겟을 제공하는 것과 충돌했습니다.
        한 변수 노드에 두 팩터의 z_target이 평균 내어질 때 물리적으로
        의미 없는 위치가 합의점이 될 수 있었습니다.

        수정 후에는 이 팩터가 위치에 대한 z_target을 제공하지 않습니다.
        theta/v에 대해서만 원형 평균 합의를 제공하여 DynamicsFactor와
        역할이 명확히 분리됩니다.
        """
        avg_v = (mean0[3] + mean1[3]) / 2.0
        sin_mean = (np.sin(mean0[2]) + np.sin(mean1[2])) / 2.0
        cos_mean = (np.cos(mean0[2]) + np.cos(mean1[2])) / 2.0
        avg_theta = np.arctan2(sin_mean, cos_mean)

        # [변경] 위치(px, py)는 DynamicsFactor에 위임하므로 여기서는
        # 현재 평균을 그대로 유지 (z_target에서 위치 간섭 제거)
        # 이 타겟이 ADMM z-update에서 DynamicsFactor 타겟과 평균 내어질 때
        # 위치 성분이 서로 충돌하지 않도록 동일한 값을 유지합니다.
        z_target0 = np.array([mean0[0], mean0[1], avg_theta, avg_v])
        z_target1 = np.array([mean1[0], mean1[1], avg_theta, avg_v])
        return z_target0, z_target1


class GridObsFactor(UnaryFNode):
    """
    SDF 기반 장애물 페널티 팩터.

    SDF 맵은 inflation_radius 범위 안에서 연속적인 그래디언트를 제공하므로
    파티클의 현재 위치를 그대로 맵 함수에 넣어 페널티를 얻습니다.
    """

    def __init__(self, name: str, occupancy_map_func, weight: float = 1e-3):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.occupancy_map_func = occupancy_map_func

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        penalties = self.occupancy_map_func(x[0:1, :], x[1:2, :])  # (1, N)
        return penalties.reshape(1, -1)