import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# =============================================================================
# 0. Modules Import
# =============================================================================
# EKI
from src.graph.EKIcore import VNode as VNodeEKI, FactorGraph as GraphEKI
from src.graph.EKIcore import UnaryFNode as UnaryFNodeEKI, BinaryFNode as BinaryFNodeEKI
# GaBP
from src.graph.GBPCore import VNode as VNodeGaBP, FactorGraph as GraphGaBP
from src.graph.GBPCore import UnaryFNode as UnaryFNodeGaBP, BinaryFNode as BinaryFNodeGaBP
# D-EKI
from src.graph.DEKICore import VNode as VNodeDEKI, FactorGraph as GraphDEKI
from src.graph.DEKICore import UnaryFNode as UnaryFNodeDEKI, BinaryFNode as BinaryFNodeDEKI
# Gauss-Newton / LM
from src.graph.GNCore import VNode as VNodeGN, FactorGraph as GraphGN
from src.graph.GNCore import UnaryFNode as UnaryFNodeGN, BinaryFNode as BinaryFNodeGN
# Global LM
from src.graph.GLMCore import VNode as VNodeGLM, FactorGraph as GraphGLM
from src.graph.GLMCore import UnaryFNode as UnaryFNodeGLM, BinaryFNode as BinaryFNodeGLM



# =============================================================================
# Utilities
# =============================================================================
def wrap_angle(angle):
    """각도를 [-pi, pi] 범위로 래핑하여 최적화 발산을 방지합니다."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# 1. Factor Classes Definition
# =============================================================================

# --- EKI, GaBP, D-EKI, GN, GLM, EKIBP 기본 팩터 (기존 코드와 동일) ---

class AnchorFactorEKI(UnaryFNodeEKI):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.reshape(2, 1)
    def _error_function(self, x):
        return x - self.measured_pos

class RangeOnlyFactorEKI(BinaryFNodeEKI):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
    def _error_function(self, x0, x1):
        dist = np.linalg.norm(x1 - x0, axis=0)
        return (dist - self.measured_dist).reshape(1, -1)

class RangeBearingFactorEKI(BinaryFNodeEKI):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing):
        super().__init__(name, dims=[2], gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
    def _error_function(self, x0, x1):
        dx = x1[0, :] - x0[0, :]
        dy = x1[1, :] - x0[1, :]
        dist = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx)
        return np.vstack([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])


class AnchorFactorGaBP(UnaryFNodeGaBP):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = np.array(measured_pos)
    def error(self, x: np.ndarray) -> np.ndarray: return x - self.measured_pos
    def jacobian(self, x: np.ndarray) -> np.ndarray: return np.eye(2)

class RangeOnlyFactorGaBP(BinaryFNodeGaBP):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(x1 - x0) - self.measured_dist])
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-8 
        J0 = np.array([[-dx / dist, -dy / dist]])
        J1 = np.array([[dx / dist, dy / dist]])
        return J0, J1

class RangeBearingFactorGaBP(BinaryFNodeGaBP):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing):
        super().__init__(name, gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx)
        return np.array([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist2 = dx**2 + dy**2 + 1e-8
        dist = np.sqrt(dist2)
        J1 = np.array([[dx / dist, dy / dist], [-dy / dist2, dx / dist2]])
        return -J1, J1


class AnchorFactorDEKI(UnaryFNodeDEKI):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.reshape(2, 1)
    def _error_function(self, x): return x - self.measured_pos

class RangeOnlyFactorDEKI(BinaryFNodeDEKI):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
    def _error_function(self, x0, x1):
        dist = np.linalg.norm(x1 - x0, axis=0)
        return (dist - self.measured_dist).reshape(1, -1)
    def _compute_z_targets(self, mean0, mean1):
        diff = mean1 - mean0
        current_dist = np.linalg.norm(diff)
        u = diff / current_dist if current_dist > 1e-8 else np.array([1.0, 0.0])
        return mean1 - self.measured_dist * u, mean0 + self.measured_dist * u

class RangeBearingFactorDEKI(BinaryFNodeDEKI):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing):
        super().__init__(name, dims=[2], gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
    def _error_function(self, x0, x1):
        dx, dy = x1[0, :] - x0[0, :], x1[1, :] - x0[1, :]
        dist, bearing = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)
        return np.vstack([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])
    def _compute_z_targets(self, mean0, mean1):
        ideal_offset = self.measured_dist * np.array([np.cos(self.measured_bearing), np.sin(self.measured_bearing)])
        return mean1 - ideal_offset, mean0 + ideal_offset

# --- 새롭게 추가된 CI 기반 D-EKI 팩터 ---
class RangeOnlyFactorDEKI_CI(BinaryFNodeDEKI):
    def __init__(self, name, measured_dist, noise_std, vn0, vn1):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
        self.vn0 = vn0
        self.vn1 = vn1

    def _error_function(self, x0, x1):
        dist = np.linalg.norm(x1 - x0, axis=0)
        return (dist - self.measured_dist).reshape(1, -1)

    def _compute_z_targets(self, mean0, mean1):
        # 앙상블로부터 Covariance Trace 추출
        trace0 = np.trace(np.cov(self.vn0.ensemble))
        trace1 = np.trace(np.cov(self.vn1.ensemble))
        
        # 내가 불확실할수록(Trace가 클수록) 상대방의 타겟 제안을 수용하는 가중치 증가
        omega0 = trace0 / (trace0 + trace1 + 1e-8)  # Node 0가 Node 1의 제안을 믿는 정도
        omega1 = trace1 / (trace0 + trace1 + 1e-8)  # Node 1이 Node 0의 제안을 믿는 정도

        diff = mean1 - mean0
        current_dist = np.linalg.norm(diff)
        u = diff / current_dist if current_dist > 1e-8 else np.array([1.0, 0.0])
        ideal_offset = self.measured_dist * u
        
        # 순수 기하학적 타겟 제안
        target_for_0 = mean1 - ideal_offset
        target_for_1 = mean0 + ideal_offset
        
        # CI 가중치를 통한 보수적 타협
        z_target0 = omega0 * target_for_0 + (1 - omega0) * mean0
        z_target1 = omega1 * target_for_1 + (1 - omega1) * mean1
        
        return z_target0, z_target1

class RangeBearingFactorDEKI_CI(BinaryFNodeDEKI):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing, vn0, vn1):
        super().__init__(name, dims=[2], gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
        self.vn0 = vn0
        self.vn1 = vn1

    def _error_function(self, x0, x1):
        dx, dy = x1[0, :] - x0[0, :], x1[1, :] - x0[1, :]
        dist, bearing = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)
        return np.vstack([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])

    def _compute_z_targets(self, mean0, mean1):
        trace0 = np.trace(np.cov(self.vn0.ensemble))
        trace1 = np.trace(np.cov(self.vn1.ensemble))
        
        omega0 = trace0 / (trace0 + trace1 + 1e-8)
        omega1 = trace1 / (trace0 + trace1 + 1e-8)

        ideal_offset = self.measured_dist * np.array([np.cos(self.measured_bearing), np.sin(self.measured_bearing)])
        
        target_for_0 = mean1 - ideal_offset
        target_for_1 = mean0 + ideal_offset
        
        z_target0 = omega0 * target_for_0 + (1 - omega0) * mean0
        z_target1 = omega1 * target_for_1 + (1 - omega1) * mean1
        
        return z_target0, z_target1


class AnchorFactorGN(UnaryFNodeGN):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.astype(float)
    def _error_function(self, x): return x - self.measured_pos

class RangeOnlyFactorGN(BinaryFNodeGN):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
    def _error_function(self, x0, x1):
        return np.array([np.linalg.norm(x1 - x0) - self.measured_dist])

class RangeBearingFactorGN(BinaryFNodeGN):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing):
        super().__init__(name, dims=[2], gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
    def _error_function(self, x0, x1):
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist, bearing = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)
        return np.array([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])

class AnchorFactorGLM(UnaryFNodeGLM):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = np.array(measured_pos)
    def error(self, x: np.ndarray) -> np.ndarray: return x - self.measured_pos
    def jacobian(self, x: np.ndarray) -> np.ndarray: return np.eye(2)

class RangeOnlyFactorGLM(BinaryFNodeGLM):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(x1 - x0) - self.measured_dist])
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-8 
        J0 = np.array([[-dx / dist, -dy / dist]])
        J1 = np.array([[dx / dist, dy / dist]])
        return J0, J1

class RangeBearingFactorGLM(BinaryFNodeGLM):
    def __init__(self, name, measured_dist, measured_bearing, noise_std_dist, noise_std_bearing):
        super().__init__(name, gamma=np.diag([noise_std_dist**2, noise_std_bearing**2]))
        self.measured_dist = measured_dist
        self.measured_bearing = measured_bearing
    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist, bearing = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)
        return np.array([dist - self.measured_dist, wrap_angle(bearing - self.measured_bearing)])
    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist2 = dx**2 + dy**2 + 1e-8
        dist = np.sqrt(dist2)
        J1 = np.array([[dx / dist, dy / dist], [-dy / dist2, dx / dist2]])
        return -J1, J1


# =============================================================================
# 2. Scenario & Graph Builder
# =============================================================================

def build_scenario():
    np.random.seed(77)
    GT = {
        'V0': np.array([0, 0]),  'V1': np.array([5, 0]),  'V2': np.array([10, 0]),
        'V3': np.array([0, 5]),  'V4': np.array([5, 5]),  'V5': np.array([10, 5]),
        'V6': np.array([0, 10]), 'V7': np.array([5, 10]), 'V8': np.array([10, 10])
    }
    anchors  = ['V0', 'V5']
    unknowns = [k for k in GT if k not in anchors]
    raw_edges = [
        ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
        ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
    ]
    edges = []
    for n1, n2 in raw_edges:
        if n2 in anchors and n1 not in anchors:
            edges.append((n2, n1))
        else:
            edges.append((n1, n2))

    anchor_noise, ro_noise, bearing_noise = 0.01, 0.1, 0.05
    measurements = {}
    for n1, n2 in edges:
        dx, dy = GT[n2][0] - GT[n1][0], GT[n2][1] - GT[n1][1]
        dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * ro_noise
        if n1 in anchors or n2 in anchors:
            bearing = np.arctan2(dy, dx) + np.random.randn() * bearing_noise
            measurements[(n1, n2)] = {'dist': dist, 'bearing': wrap_angle(bearing)}
        else:
            measurements[(n1, n2)] = {'dist': dist}

    initial_guesses = {}
    for name in GT:
        if name in anchors:
            initial_guesses[name] = GT[name].copy().astype(float)
        else:
            initial_guesses[name] = GT[name].copy().astype(float) + np.random.randn(2) * 10.0

    return GT, anchors, unknowns, edges, measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise


def build_graph(algo_type, GT, anchors, unknowns, edges_list,
                measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise=0.05):
    if algo_type == 'EKI': graph = GraphEKI()
    elif algo_type == 'GaBP': graph = GraphGaBP()
    elif algo_type in ('DEKI_cov', 'DEKI_res', 'DEKI_CI'): graph = GraphDEKI()
    elif algo_type == 'GN': graph = GraphGN(lm_lambda_init=0.0, lm_adapt=False)
    elif algo_type == 'LM': graph = GraphGN(lm_lambda_init=1e-2, lm_adapt=True, lm_nu=5.0)
    elif algo_type == 'GLM': graph = GraphGLM(lm_lambda_init=1e-2)
    else: raise ValueError(f"Unknown algo_type: {algo_type}")

    vnodes = {}
    for name in GT:
        if algo_type == 'GaBP':
            vn = VNodeGaBP(name, dims=[2], prior_std=100.0)
            vn.mu = initial_guesses[name].copy()
        elif algo_type in ('GN', 'LM', 'GLM'):
            vn = VNodeGN(name, dims=[2], prior_std=100.0) if algo_type != 'GLM' else VNodeGLM(name, dims=[2], initial_mu=initial_guesses[name].copy())
            vn.mu = initial_guesses[name].copy()
        else:
            if algo_type == 'EKI': vn = VNodeEKI(name, dims=[2], n_particles=1000, noise_std=1.0)
            elif algo_type == 'DEKI_cov': vn = VNodeDEKI(name, dims=[2], n_particles=1000, noise_std=1.0, rho_init=1.0, rho_update_method='covariance')
            elif algo_type == 'DEKI_res': vn = VNodeDEKI(name, dims=[2], n_particles=1000, noise_std=1.0, rho_init=1.0, rho_update_method='residual')
            elif algo_type == 'DEKI_CI':  vn = VNodeDEKI(name, dims=[2], n_particles=1000, noise_std=1.0, rho_init=1.0, rho_update_method='residual')
            vn.ensemble = initial_guesses[name].reshape(2, 1) + np.random.randn(2, vn.n_particles) * 10.0

        graph.nodes.append(vn)
        vnodes[name] = vn

    for a_name in anchors:
        if algo_type == 'EKI': f = AnchorFactorEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type == 'GaBP': f = AnchorFactorGaBP(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type in ('DEKI_cov', 'DEKI_res', 'DEKI_CI'): f = AnchorFactorDEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type == 'GLM': f = AnchorFactorGLM(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        else: f = AnchorFactorGN(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        graph.nodes.append(f)
        graph.connect(f, vnodes[a_name])

    for (n1, n2) in edges_list:
        meas = measurements[(n1, n2)]
        dist = meas['dist']
        
        if n1 in anchors or n2 in anchors:
            bearing = meas['bearing']
            if algo_type == 'EKI': f = RangeBearingFactorEKI(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise)
            elif algo_type == 'GaBP': f = RangeBearingFactorGaBP(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise)
            elif algo_type in ('DEKI_cov', 'DEKI_res'): f = RangeBearingFactorDEKI(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise)
            elif algo_type == 'DEKI_CI': f = RangeBearingFactorDEKI_CI(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise, vnodes[n1], vnodes[n2])
            elif algo_type == 'GLM': f = RangeBearingFactorGLM(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise)
            else: f = RangeBearingFactorGN(f"F_RB_{n1}_{n2}", dist, bearing, ro_noise, bearing_noise)
        else:
            if algo_type == 'EKI': f = RangeOnlyFactorEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
            elif algo_type == 'GaBP': f = RangeOnlyFactorGaBP(f"F_RO_{n1}_{n2}", dist, ro_noise)
            elif algo_type in ('DEKI_cov', 'DEKI_res'): f = RangeOnlyFactorDEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
            elif algo_type == 'DEKI_CI': f = RangeOnlyFactorDEKI_CI(f"F_RO_{n1}_{n2}", dist, ro_noise, vnodes[n1], vnodes[n2])
            elif algo_type == 'GLM': f = RangeOnlyFactorGLM(f"F_RO_{n1}_{n2}", dist, ro_noise)
            else: f = RangeOnlyFactorGN(f"F_RO_{n1}_{n2}", dist, ro_noise)
            
        graph.nodes.append(f)
        graph.connect(f, vnodes[n1])
        graph.connect(f, vnodes[n2])

    return graph, vnodes


# =============================================================================
# 3. Main Testing & Plotting Logics
# =============================================================================

def run_test():
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise, bearing_noise = build_scenario()

    # DEKI_CI 와 EKIBP 포함하여 8개 알고리즘 구성
    algos = ['EKI', 'GaBP', 'DEKI_cov', 'DEKI_res', 'DEKI_CI', 'LM', 'GLM']

    graphs     = {}
    vnodes_d   = {}
    history    = {a: {n: [initial_guesses[n]] for n in unknowns} for a in algos}
    rmse       = {a: [] for a in algos}

    def get_estimate(vn, algo):
        if algo in ('GaBP', 'GN', 'LM', 'GLM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, algo):
        errs = [np.linalg.norm(get_estimate(vn_dict[n], algo) - GT[n]) for n in unknowns]
        return float(np.mean(errs))

    for algo in algos:
        graphs[algo], vnodes_d[algo] = build_graph(
            algo, GT, anchors, unknowns, edges_list,
            measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise
        )
        rmse[algo].append(get_rmse(vnodes_d[algo], algo))

    n_iter     = 100
    decay_rate = 0.5

    print("=" * 90)
    print(f"{'Iter':>4} | {'EKI':>6} | {'GaBP':>6} | {'DEKI_c':>6} | {'DEKI_r':>6} | {'DEKI_CI':>7} | {'LM':>6} | {'GLM':>6}")
    print("=" * 90)

    vals = " | ".join(f"{rmse[a][-1]:6.3f}" for a in algos)
    print(f"{0:4d} | {vals}")

    for i in range(n_iter):
        for algo in algos:
            graphs[algo].iterate(1)
            if algo not in ('GaBP', 'GN', 'LM', 'GLM'):
                for vn in vnodes_d[algo].values():
                    if hasattr(vn, 'noise_std'):
                        vn.noise_std *= decay_rate
            for n in unknowns:
                history[algo][n].append(get_estimate(vnodes_d[algo][n], algo).copy())
            rmse[algo].append(get_rmse(vnodes_d[algo], algo))

        if (i + 1) % 10 == 0 or i == 0:
            vals = " | ".join(f"{rmse[a][-1]:>7.3f}" if a == 'DEKI_CI' else f"{rmse[a][-1]:6.3f}" for a in algos)
            print(f"{i+1:4d} | {vals}")

    print("=" * 90)
    final = " | ".join(f"{rmse[a][-1]:>7.3f}" if a == 'DEKI_CI' else f"{rmse[a][-1]:6.3f}" for a in algos)
    print(f"{'FINAL':>4} | {final}")
    print("=" * 90)

    # 3x3 Grid (9개 Plot: Init + 8 Algos)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_traj, axes_traj = plt.subplots(3, 3, figsize=(20, 18), dpi=100)

    gt_x = [GT[k][0] for k in GT]
    gt_y = [GT[k][1] for k in GT]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unknowns)))

    def plot_network(ax, title, hist=None, color_override=None):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        
        # 1. GT 노드들을 각 궤적과 동일한 색상으로 표시
        for idx, name in enumerate(unknowns):
            c = color_override if color_override else colors[idx]
            # 투명도(alpha=0.4)를 주어 최종 도착점(*)과 겹쳐도 잘 보이게 설정
            ax.scatter(GT[name][0], GT[name][1], c=[c], s=150, zorder=2, edgecolors='dimgrey', linewidth=1.5, alpha=0.4, label='GT' if idx==0 else "")
            
        # 2. 앵커 노드 표시
        for a in anchors:
            ax.scatter(GT[a][0], GT[a][1], c='red', marker='s', s=180, edgecolors='black', linewidth=2, zorder=5)
            
        # 3. 엣지(네트워크 뼈대) 표시
        for n1, n2 in edges_list:
            ax.plot([GT[n1][0], GT[n2][0]], [GT[n1][1], GT[n2][1]], 'black', alpha=0.25, linewidth=1.2, zorder=1)
            
        # 4. 최적화 궤적 표시
        if hist:
            for idx, name in enumerate(unknowns):
                c = color_override if color_override else colors[idx]
                traj = np.array(hist[name])
                # 궤적 선
                ax.plot(traj[:, 0], traj[:, 1], c=c, marker='.', markersize=4, alpha=0.8, linewidth=2.5, zorder=4)
                # 초기 위치 (X 마커)
                ax.scatter(traj[0, 0], traj[0, 1], c=[c], marker='X', s=120, edgecolors='black', linewidth=1.5, zorder=6)
                # 최종 위치 (* 마커)
                ax.scatter(traj[-1, 0], traj[-1, 1], c=[c], marker='*', s=300, edgecolors='black', linewidth=1.5, zorder=7)
                
        ax.set_xlim(-5, 15); ax.set_ylim(-5, 15); ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=12)

    plot_network(axes_traj[0, 0], "Initial State")
    for name in unknowns:
        axes_traj[0, 0].scatter(initial_guesses[name][0], initial_guesses[name][1], c='orange', marker='X', s=90, edgecolors='k')
    
    plot_network(axes_traj[0, 1], f"Local LM  (final={rmse['LM'][-1]:.3f}m)", history['LM'])
    plot_network(axes_traj[0, 2], f"Global LM  (final={rmse['GLM'][-1]:.3f}m)", history['GLM'])
    plot_network(axes_traj[1, 0], f"GaBP  (final={rmse['GaBP'][-1]:.3f}m)", history['GaBP'])
    plot_network(axes_traj[1, 1], f"Vanilla EKI  (final={rmse['EKI'][-1]:.3f}m)", history['EKI'])
    plot_network(axes_traj[1, 2], f"DEKI-Cov  (final={rmse['DEKI_cov'][-1]:.3f}m)", history['DEKI_cov'])
    plot_network(axes_traj[2, 0], f"DEKI-Res  (final={rmse['DEKI_res'][-1]:.3f}m)", history['DEKI_res'])
    plot_network(axes_traj[2, 1], f"DEKI-CI  (final={rmse['DEKI_CI'][-1]:.3f}m)", history['DEKI_CI'])
    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=0.5)

    # Plot 2: Convergence
    fig_rmse, axes_rmse = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
    ax_r = axes_rmse[0]
    ax_r.set_title("RMSE Convergence", fontsize=16, fontweight='bold', pad=15)

    styles = {
        'EKI':      ('mediumblue',  '--',  'o', 2.5),
        'GaBP':     ('red',         '--',  'o', 2.5),
        'DEKI_cov': ('olivedrab',   '--',  'o', 2.5),
        'DEKI_res': ('purple',      '--',  'o', 2.5),
        'DEKI_CI':  ('brown',       '-.',  'd', 2.5), # DEKI_CI 스타일 지정
        'LM':       ('darkorange',  '--',  'o', 2.5),
        'GLM':      ('darkcyan',    '--',  'o', 2.5),
    }

    for algo, (col, ls, mk, lw) in styles.items():
        clipped = np.clip(rmse[algo], 0, 8.0)
        ax_r.plot(clipped, label=algo, color=col, linestyle=ls, marker=mk, markersize=5, linewidth=lw, markevery=10, alpha=0.9)

    ax_r.set_xlabel("Iteration", fontsize=16); ax_r.set_ylabel("Mean Error (m)", fontsize=16)
    ax_r.set_ylim(0, 3); ax_r.legend(fontsize=12, frameon=True, shadow=True, loc='upper right'); ax_r.grid(True, linestyle=':', alpha=0.6)

    ax_b = axes_rmse[1]
    ax_b.set_title("Final RMSE Comparison", fontsize=16, fontweight='bold', pad=15)
    final_vals = [rmse[a][-1] for a in algos]
    bar_colors = [styles[a][0] for a in algos]
    bars = ax_b.bar(algos, final_vals, color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    for bar, val in zip(bars, final_vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    ax_b.set_ylabel("Mean Position Error (m)", fontsize=16)
    ax_b.set_ylim(0, max(final_vals) * 1.3)
    ax_b.set_xticklabels(algos, fontsize=12, fontweight='bold', rotation=15); ax_b.grid(axis='y', linestyle='--', alpha=0.5)

    fig_rmse.tight_layout(pad=3.0)
    plt.show()

def run_algo_comparison(num_anchors: int, n_trials: int = 100, n_iter: int = 30, decay_rate: float = 0.5):
    algos     = ['EKI', 'GaBP', 'DEKI_cov', 'DEKI_res', 'DEKI_CI', 'LM', 'GLM']
    all_nodes = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']

    def get_estimate(vn, algo):
        if algo in ('GaBP', 'GN', 'LM', 'GLM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, unknowns_list, GT, algo):
        errs = [np.linalg.norm(get_estimate(vn_dict[n], algo) - GT[n]) for n in unknowns_list]
        return float(np.mean(errs))

    trial_rmses = {a: [] for a in algos}

    print(f"\n{'='*75}\n  Algorithm Comparison (Trials: {n_trials}, Anchors: {num_anchors})\n{'='*75}")
    for trial in range(n_trials):
        np.random.seed(trial)
        chosen_anchors = list(np.random.choice(all_nodes, size=num_anchors, replace=False))
        unknowns_trial = [v for v in all_nodes if v not in chosen_anchors]

        GT = {
            'V0': np.array([0.0,  0.0]),  'V1': np.array([5.0,  0.0]),  'V2': np.array([10.0,  0.0]),
            'V3': np.array([0.0,  5.0]),  'V4': np.array([5.0,  5.0]),  'V5': np.array([10.0,  5.0]),
            'V6': np.array([0.0, 10.0]),  'V7': np.array([5.0, 10.0]),  'V8': np.array([10.0, 10.0])
        }
        raw_edges = [
            ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
            ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
        ]
        edges = []
        for n1, n2 in raw_edges:
            if n2 in chosen_anchors and n1 not in chosen_anchors:
                edges.append((n2, n1))
            else:
                edges.append((n1, n2))

        anchor_noise, ro_noise, bearing_noise = 0.01, 0.5, 0.1
        measurements = {}
        for n1, n2 in edges:
            dx, dy = GT[n2][0] - GT[n1][0], GT[n2][1] - GT[n1][1]
            dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * ro_noise
            if n1 in chosen_anchors or n2 in chosen_anchors:
                bearing = np.arctan2(dy, dx) + np.random.randn() * bearing_noise
                measurements[(n1, n2)] = {'dist': dist, 'bearing': wrap_angle(bearing)}
            else:
                measurements[(n1, n2)] = {'dist': dist}

        initial_guesses = {}
        for name in GT:
            if name in chosen_anchors:
                initial_guesses[name] = GT[name].copy()
            else:
                initial_guesses[name] = GT[name].copy() + np.random.randn(2) * 3.0

        for algo in algos:
            graph, vnodes = build_graph(
                algo, GT, chosen_anchors, unknowns_trial,
                edges, measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise
            )
            for _ in range(n_iter):
                graph.iterate(1)
                if algo not in ('GaBP', 'GN', 'LM', 'GLM'):
                    for vn in vnodes.values():
                        if hasattr(vn, 'noise_std'): vn.noise_std *= decay_rate
            rmse_val = get_rmse(vnodes, unknowns_trial, GT, algo)
            trial_rmses[algo].append(rmse_val)

        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1:3d} completed.")

    results = {}
    for algo in algos:
        rmses = trial_rmses[algo]
        results[algo] = {'rmses': rmses, 'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))}

    print("\nResults:")
    for algo in algos:
        print(f"  {algo:<10} Mean RMSE: {results[algo]['mean']:.4f} m  (Std: {results[algo]['std']:.4f})")
    return results

def draw_confidence_ellipse(mu, cov, ax, n_std, edgecolor, **kwargs):
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
        return False
    try:
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        if np.any(eigenvals <= 0):
            return False
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvals)
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, 
                          edgecolor=edgecolor, facecolor='none', **kwargs)
        ax.add_patch(ellipse)
        return True
    except Exception:
        return False


def run_ensemble_visualization(algo='DEKI_CI'):
    """
    EKI, DEKI_cov, DEKI_res, DEKI_CI 등의 앙상블 입자 변화를 시각화합니다.
    """
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise, bearing_noise = build_scenario()

    graph, vnodes = build_graph(
        algo, GT, anchors, unknowns, edges_list,
        measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise
    )

    target_iters = [0, 1, 5, 10, 20, 100]
    snapshots = {}
    decay_rate = 0.5

    snapshots[0] = {n: vnodes[n].ensemble.copy() for n in unknowns}
    print(f"Running {algo} Ensemble Simulation...")
    
    for i in range(1, max(target_iters) + 1):
        graph.iterate(1)
        for vn in vnodes.values():
            if hasattr(vn, 'noise_std'):
                vn.noise_std *= decay_rate
        if i in target_iters:
            snapshots[i] = {n: vnodes[n].ensemble.copy() for n in unknowns}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{algo} Ensemble Evolution", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unknowns)))

    for idx, step in enumerate(target_iters):
        ax = axes[idx]
        ax.set_title(f"Iteration {step}", fontsize=14, fontweight='bold')
        for n1, n2 in edges_list:
            ax.plot([GT[n1][0], GT[n2][0]], [GT[n1][1], GT[n2][1]], 'k--', alpha=0.2, linewidth=1)
        for a in anchors:
            ax.scatter(GT[a][0], GT[a][1], c='red', marker='s', s=150, edgecolors='k', zorder=5)
        for c_idx, name in enumerate(unknowns):
            ax.scatter(GT[name][0], GT[name][1], c=[colors[c_idx]], marker='*', s=200, edgecolors='k', zorder=6)

        current_particles = snapshots[step]
        for c_idx, name in enumerate(unknowns):
            particles = current_particles[name]
            ax.scatter(particles[0, :], particles[1, :], color=colors[c_idx], s=5, alpha=0.15, zorder=3)
            mean_est = particles.mean(axis=1)
            ax.scatter(mean_est[0], mean_est[1], c=[colors[c_idx]], marker='o', s=80, edgecolors='black', zorder=4)

        ax.set_xlim(-5, 15); ax.set_ylim(-5, 15); ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def run_gabp_visualization():
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise, bearing_noise = build_scenario()

    graph, vnodes = build_graph(
        'GaBP', GT, anchors, unknowns, edges_list,
        measurements, initial_guesses, anchor_noise, ro_noise, bearing_noise
    )

    target_iters = [0, 1, 5, 10, 20, 100]
    snapshots = {}
    snapshots[0] = {n: {'mu': vnodes[n].mu.flatten(), 'cov': np.eye(2) * 5.0} for n in unknowns}

    print("Running GaBP Covariance Simulation...")
    for i in range(1, max(target_iters) + 1):
        graph.iterate(1)
        if i in target_iters:
            snapshots[i] = {}
            for n in unknowns:
                vn = vnodes[n]
                mu = vn.mu.flatten().copy()
                cov = vn.Sigma.copy() if hasattr(vn, 'Sigma') else np.eye(2) * 9999.0
                snapshots[i][n] = {'mu': mu, 'cov': cov}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("GaBP Belief Evolution (Chi-Square Confidence Ellipses)", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unknowns)))
    confidence_levels = [(1.0, 1.0, '-'), (2.0, 0.6, '--'), (3.0, 0.3, ':')]

    for idx, step in enumerate(target_iters):
        ax = axes[idx]
        ax.set_title(f"Iteration {step}", fontsize=14, fontweight='bold')
        for n1, n2 in edges_list:
            ax.plot([GT[n1][0], GT[n2][0]], [GT[n1][1], GT[n2][1]], 'k--', alpha=0.2, linewidth=1)
        for a in anchors:
            ax.scatter(GT[a][0], GT[a][1], c='red', marker='s', s=150, edgecolors='k', zorder=5)
        for c_idx, name in enumerate(unknowns):
            ax.scatter(GT[name][0], GT[name][1], c=[colors[c_idx]], marker='*', s=200, edgecolors='k', zorder=6)

        current_state = snapshots[step]
        for c_idx, name in enumerate(unknowns):
            mu = current_state[name]['mu']
            cov = current_state[name]['cov']
            ax.scatter(mu[0], mu[1], c=[colors[c_idx]], marker='o', s=80, edgecolors='black', zorder=4)

            success = True
            for n_std, alpha, linestyle in confidence_levels:
                if not draw_confidence_ellipse(mu, cov, ax, n_std=n_std, edgecolor=colors[c_idx], alpha=alpha, linestyle=linestyle, linewidth=2):
                    success = False
                    break
            if not success and idx > 0:
                ax.text(mu[0] + 0.5, mu[1] + 0.5, "Covariance Diverged!", color='red', fontsize=10, fontweight='bold')
        ax.set_xlim(-10, 20); ax.set_ylim(-10, 20); ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    run_test()
    # run_ensemble_visualization(algo='DEKI_CI')  # 'DEKI_res', 'EKI' 등으로 변경 가능
    # run_gabp_visualization()
    # run_algo_comparison(num_anchors=2)