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

class CircleObstacle(Obstacle):
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # 원의 방정식: (x - cx)^2 + (y - cy)^2 <= r^2
        return ((x - self.cx)**2 + (y - self.cy)**2) <= (self.radius**2)

class RectangleObstacle(Obstacle):
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # 사각형 경계 조건
        return (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)


# ==========================================
# 2. 맵 환경(Environment) 매니저 클래스
# ==========================================
class EnvironmentMap:
    def __init__(self, penalty_value: float = 10.0):
        self.obstacles = []
        self.penalty_value = penalty_value

    def add_obstacle(self, obstacle: Obstacle):
        """ 맵에 장애물 추가 """
        self.obstacles.append(obstacle)

    def get_penalty(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ 
        [EKI 팩터용 병렬 함수] 
        입력된 x, y 좌표 배열(K, N)에 대해 장애물 충돌 여부를 검사하고 페널티 배열을 반환
        """
        # x와 동일한 형태의 0.0 배열 생성
        penalties = np.zeros_like(x, dtype=float)
        
        # 장애물이 없으면 바로 0 반환
        if not self.obstacles:
            return penalties

        # 모든 장애물에 대해 논리합(OR)으로 충돌 영역 마스크(Mask) 생성
        collision_mask = np.zeros_like(x, dtype=bool)
        for obs in self.obstacles:
            collision_mask = collision_mask | obs.contains(x, y)
            
        # 장애물 내부에 있는 좌표에만 페널티 부여
        penalties[collision_mask] = self.penalty_value
        
        return penalties

    def generate_grid_map(self, x_range=(0, 10), y_range=(0, 10), resolution=0.1):
        """ 
        시각화 또는 다른 알고리즘(A*)을 위한 2D Occupancy Grid 배열 생성 
        """
        # 해상도에 맞춰 x, y 축 격자 생성
        x_coords = np.arange(x_range[0], x_range[1], resolution)
        y_coords = np.arange(y_range[0], y_range[1], resolution)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # get_penalty 재사용하여 전체 그리드의 픽셀값 한 번에 계산
        grid_penalties = self.get_penalty(xx, yy)
        
        return x_coords, y_coords, grid_penalties

    def visualize(self, x_range=(0, 10), y_range=(0, 10), resolution=0.1):
        """ 생성된 장애물 맵을 화면에 그리기 """
        x_coords, y_coords, grid = self.generate_grid_map(x_range, y_range, resolution)
        
        plt.figure(figsize=(8, 8))
        # origin='lower'를 해야 y축이 아래에서 위로 정상적으로 표시됨
        plt.imshow(grid, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
                   origin='lower', cmap='Greys', alpha=0.6)
        plt.title("2D Environment Grid Map")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    def draw_obstacles(self, ax, color='gray', alpha=0.5):
        """
        주어진 Matplotlib Axes(ax)에 현재 맵의 모든 장애물을 Patch로 그립니다.
        """
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                # 원형 장애물 추가
                circle = patches.Circle((obs.cx, obs.cy), obs.radius, 
                                       edgecolor='black', facecolor=color, alpha=alpha, zorder=1)
                ax.add_patch(circle)
                
            elif isinstance(obs, RectangleObstacle):
                # 사각형 장애물 추가
                rect = patches.Rectangle((obs.x_min, obs.y_min), 
                                         obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                                         edgecolor='black', facecolor=color, alpha=alpha, zorder=1)
                ax.add_patch(rect)