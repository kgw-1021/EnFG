import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 장애물 기본 형태(Shape) 클래스 정의
# ==========================================
class Obstacle:
    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ x, y 좌표(배열)가 장애물 내부에 있는지(True/False) 반환 """
        raise NotImplementedError

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        [SDF] 부호 있는 거리 반환.
        - 양수: 장애물 외부까지의 거리
        - 음수: 장애물 내부 깊이 (중심으로 갈수록 더 음수)
        기본 구현은 contains()를 이용한 fallback. 각 서브클래스에서 해석적으로 오버라이드.
        """
        raise NotImplementedError


class CircleObstacle(Obstacle):
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x - self.cx)**2 + (y - self.cy)**2) <= (self.radius**2)

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        원형 장애물의 해석적 SDF.
        sdf = dist_to_center - radius
          → 외부: sdf > 0 (표면까지 얼마나 남았는지)
          → 표면: sdf = 0
          → 내부: sdf < 0 (중심에 가까울수록 더 음수)
        """
        dist_to_center = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return dist_to_center - self.radius


class RectangleObstacle(Obstacle):
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        직사각형 장애물의 해석적 SDF.
        사각형 중심 기준으로 각 축의 half-extent 까지의 거리를 계산합니다.

        외부 점: 가장 가까운 표면까지의 유클리드 거리 (양수)
        내부 점: 가장 가까운 표면까지의 거리에 음수 부호 (음수)
        """
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        hx = (self.x_max - self.x_min) / 2.0
        hy = (self.y_max - self.y_min) / 2.0

        # 중심으로부터의 상대 위치 (절댓값 → 1사분면으로 접기)
        dx = np.abs(x - cx) - hx
        dy = np.abs(y - cy) - hy

        # 외부 거리: max(dx,0)^2 + max(dy,0)^2 의 루트
        # 내부 거리: min(max(dx,dy), 0) — 표면까지 얼마나 깊이 들어왔는지
        outside_dist = np.sqrt(np.maximum(dx, 0.0)**2 + np.maximum(dy, 0.0)**2)
        inside_dist  = np.minimum(np.maximum(dx, dy), 0.0)  # 내부면 음수

        return outside_dist + inside_dist


# ==========================================
# 2. 맵 환경(Environment) 매니저 클래스
# ==========================================
class EnvironmentMap:
    def __init__(self, penalty_value: float = 10.0, inflation_radius: float = 1.0):
        """
        Args:
            penalty_value:    장애물 표면(SDF=0)에서의 최대 페널티 값.
                              내부로 갈수록 선형으로 더 커지고,
                              외부에서는 inflation_radius 만큼 거리를 두고 0으로 줄어듭니다.
            inflation_radius: 장애물 표면 바깥으로 페널티가 퍼지는 반경.
                              이 범위 안에서 EKI 앙상블이 방향 정보(그래디언트 역할)를 얻습니다.
        """
        self.obstacles = []
        self.penalty_value = penalty_value
        self.inflation_radius = inflation_radius

    def add_obstacle(self, obstacle: Obstacle):
        """ 맵에 장애물 추가 """
        self.obstacles.append(obstacle)

    def _compute_sdf_penalty(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        모든 장애물의 SDF를 합성하여 점진적 페널티를 계산합니다.

        페널티 프로파일 (sdf = 장애물까지의 부호있는 거리):
        
          penalty
            ^
        P_max + k·|sdf| ─── (내부, sdf < 0, 중심으로 갈수록 선형 증가)
            |         /
        P_max ────── * ←── 표면 (sdf = 0)
            |       /
            |      /  ← 외부 감쇠 구간 (0 < sdf < inflation_radius)
            |     /       penalty = P_max * (1 - sdf/R)
            |    /
          0 +──────────────────────────────> sdf
                0     inflation_radius

        - 내부(sdf < 0):  penalty = P_max + P_max * |sdf| / inflation_radius
          → 중심으로 갈수록 선형 증가. 앙상블이 중심에 있어도 표면 방향 그래디언트 존재.

        - 표면(sdf = 0):  penalty = P_max

        - 외부 감쇠(0 < sdf < R):  penalty = P_max * (1 - sdf / R)
          → 접근 전부터 부드럽게 경고. EKI 앙상블이 미리 회피 방향을 학습.

        - 안전 구역(sdf >= R):  penalty = 0

        여러 장애물이 있을 경우: 각 장애물의 페널티를 독립 계산 후 최댓값으로 합성(union).
        """
        if not self.obstacles:
            return np.zeros_like(x, dtype=float)

        R = self.inflation_radius
        P = self.penalty_value
        combined = np.zeros_like(x, dtype=float)

        for obs in self.obstacles:
            sdf = obs.signed_distance(x, y)  # 양수=외부, 음수=내부

            penalty = np.where(
                sdf < 0,
                # 내부: 표면 페널티 + 깊이에 비례한 추가 페널티
                P + P * np.abs(sdf) / R,
                # 외부: inflation_radius 내에서 선형 감쇠
                np.where(
                    sdf < R,
                    P * (1.0 - sdf / R),
                    0.0
                )
            )
            # 여러 장애물의 페널티는 최댓값(union) 으로 합성
            combined = np.maximum(combined, penalty)

        return combined

    def get_penalty(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        [EKI 팩터용 병렬 함수 — 인터페이스 동일]
        입력된 x, y 좌표 배열에 대해 SDF 기반 점진적 페널티를 반환합니다.
        기존의 binary(0 or penalty_value) 대신 연속적인 값을 반환하므로
        장애물 내부에서도 EKI 앙상블이 탈출 방향 정보를 얻을 수 있습니다.
        """
        return self._compute_sdf_penalty(x, y)

    def generate_grid_map(self, x_range=(0, 10), y_range=(0, 10), resolution=0.1):
        """
        시각화 또는 다른 알고리즘(A*)을 위한 2D Occupancy Grid 배열 생성
        """
        x_coords = np.arange(x_range[0], x_range[1], resolution)
        y_coords = np.arange(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_penalties = self.get_penalty(xx, yy)
        return x_coords, y_coords, grid_penalties

    def visualize(self, x_range=(0, 10), y_range=(0, 10), resolution=0.05):
        """
        SDF 기반 페널티 맵을 시각화합니다.
        binary 맵과 달리 등고선(contour)으로 점진적 그래디언트를 확인할 수 있습니다.
        """
        x_coords, y_coords, grid = self.generate_grid_map(x_range, y_range, resolution)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- 왼쪽: 페널티 히트맵 ---
        ax1 = axes[0]
        im = ax1.imshow(
            grid,
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            origin='lower', cmap='hot_r', alpha=0.85
        )
        # 등고선으로 그래디언트 명시
        xx, yy = np.meshgrid(x_coords, y_coords)
        ax1.contour(xx, yy, grid, levels=10, colors='white', linewidths=0.6, alpha=0.6)
        plt.colorbar(im, ax=ax1, label='Penalty Value')
        ax1.set_title("SDF-based Penalty Map (Heatmap + Contour)")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, linestyle='--', alpha=0.2)

        # --- 오른쪽: SDF 원본 (부호 있는 거리) ---
        ax2 = axes[1]
        if self.obstacles:
            sdf_combined = np.full_like(grid, np.inf)
            for obs in self.obstacles:
                sdf_combined = np.minimum(sdf_combined, obs.signed_distance(xx, yy))

            vmax = max(abs(sdf_combined.min()), abs(sdf_combined.max()))
            im2 = ax2.imshow(
                sdf_combined,
                extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                origin='lower', cmap='RdYlGn', vmin=-vmax, vmax=vmax, alpha=0.85
            )
            ax2.contour(xx, yy, sdf_combined, levels=[0.0], colors='black', linewidths=2.0)
            ax2.contour(xx, yy, sdf_combined,
                        levels=np.linspace(-vmax, vmax, 12),
                        colors='gray', linewidths=0.5, alpha=0.5)
            plt.colorbar(im2, ax=ax2, label='Signed Distance (m)')
            ax2.set_title("Signed Distance Field (SDF)\nBlack contour = obstacle surface")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.grid(True, linestyle='--', alpha=0.2)

        plt.suptitle(
            f"EnvironmentMap  |  penalty_value={self.penalty_value},  "
            f"inflation_radius={self.inflation_radius}",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

    def draw_obstacles(self, ax, color='gray', alpha=0.5):
        """
        주어진 Matplotlib Axes(ax)에 현재 맵의 모든 장애물을 Patch로 그립니다.
        """
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                circle = patches.Circle(
                    (obs.cx, obs.cy), obs.radius,
                    edgecolor='black', facecolor=color, alpha=alpha, zorder=1
                )
                ax.add_patch(circle)
                # inflation 반경도 점선으로 표시 (선택적)
                inflation_circle = patches.Circle(
                    (obs.cx, obs.cy), obs.radius + self.inflation_radius,
                    edgecolor='gray', facecolor='none',
                    linestyle='--', linewidth=1.0, alpha=0.4, zorder=1
                )
                ax.add_patch(inflation_circle)

            elif isinstance(obs, RectangleObstacle):
                rect = patches.Rectangle(
                    (obs.x_min, obs.y_min),
                    obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                    edgecolor='black', facecolor=color, alpha=alpha, zorder=1
                )
                ax.add_patch(rect)