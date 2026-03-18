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
# 1. Factor Classes Definition
# =============================================================================

# --- EKI ---
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
        dx = x1[0, :] - x0[0, :]
        dy = x1[1, :] - x0[1, :]
        dist = np.sqrt(dx**2 + dy**2)
        return (dist - self.measured_dist).reshape(1, -1)


# --- GaBP ---
class AnchorFactorGaBP(UnaryFNodeGaBP):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = np.array(measured_pos)

    def error(self, x: np.ndarray) -> np.ndarray:
        return x - self.measured_pos

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.eye(2)

class RangeOnlyFactorGaBP(BinaryFNodeGaBP):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dist - self.measured_dist])

    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-8 
        J0 = np.array([[-dx / dist, -dy / dist]])
        J1 = np.array([[dx / dist, dy / dist]])
        return J0, J1


# --- D-EKI ---
class AnchorFactorDEKI(UnaryFNodeDEKI):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.reshape(2, 1)

    def _error_function(self, x):
        return x - self.measured_pos

class RangeOnlyFactorDEKI(BinaryFNodeDEKI):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def _error_function(self, x0, x1):
        dx = x1[0, :] - x0[0, :]
        dy = x1[1, :] - x0[1, :]
        dist = np.sqrt(dx**2 + dy**2)
        return (dist - self.measured_dist).reshape(1, -1)

    def _compute_z_targets(self, mean0, mean1):
        diff = mean1 - mean0
        current_dist = np.linalg.norm(diff)
        u = diff / current_dist if current_dist > 1e-8 else np.array([1.0, 0.0])
        z_target0 = mean1 - self.measured_dist * u
        z_target1 = mean0 + self.measured_dist * u
        return z_target0, z_target1


# --- GN / LM ---
class AnchorFactorGN(UnaryFNodeGN):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.astype(float)

    def _error_function(self, x):
        return x - self.measured_pos

class RangeOnlyFactorGN(BinaryFNodeGN):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def _error_function(self, x0, x1):
        dx = x1[0] - x0[0]
        dy = x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dist - self.measured_dist])


# --- Global LM ---
class AnchorFactorGLM(UnaryFNodeGLM):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = np.array(measured_pos)

    def error(self, x: np.ndarray) -> np.ndarray:
        return x - self.measured_pos

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.eye(2)

class RangeOnlyFactorGLM(BinaryFNodeGLM):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def error(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dist - self.measured_dist])

    def jacobians(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x1[0] - x0[0], x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-8 
        J0 = np.array([[-dx / dist, -dy / dist]])
        J1 = np.array([[dx / dist, dy / dist]])
        return J0, J1
    

# =============================================================================
# 2. Scenario & Graph Builder
# =============================================================================

def build_scenario():
    np.random.seed(100)
    GT = {
        'V0': np.array([0, 0]),  'V1': np.array([5, 0]),  'V2': np.array([10, 0]),
        'V3': np.array([0, 5]),  'V4': np.array([5, 5]),  'V5': np.array([10, 5]),
        'V6': np.array([0, 10]), 'V7': np.array([5, 10]), 'V8': np.array([10, 10])
    }

    anchors  = ['V4']  # 고정된 위치를 가진 노드들
    unknowns = [k for k in GT if k not in anchors]

    edges = [
        ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
        ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
    ]

    anchor_noise = 0.01
    ro_noise     = 0.5

    measurements = {}
    for n1, n2 in edges:
        dx = GT[n2][0] - GT[n1][0]
        dy = GT[n2][1] - GT[n1][1]
        dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * ro_noise
        measurements[(n1, n2)] = dist

    initial_guesses = {}
    for name in GT:
        if name in anchors:
            initial_guesses[name] = GT[name].copy().astype(float)
        else:
            initial_guesses[name] = GT[name].copy().astype(float) + np.random.randn(2) * 1.0

    return GT, anchors, unknowns, edges, measurements, initial_guesses, anchor_noise, ro_noise


def build_graph(algo_type, GT, anchors, unknowns, edges_list,
                measurements, initial_guesses, anchor_noise, ro_noise):
    
    # --- 1. Graph Instantiation ---
    if algo_type == 'EKI':
        graph = GraphEKI()
    elif algo_type == 'GaBP':
        graph = GraphGaBP()
    elif algo_type in ('DEKI_cov', 'DEKI_res'):
        graph = GraphDEKI()
    elif algo_type == 'GN':
        graph = GraphGN(lm_lambda_init=0.0, lm_adapt=False)
    elif algo_type == 'LM':
        graph = GraphGN(lm_lambda_init=1e-2, lm_adapt=True, lm_nu=5.0)
    elif algo_type == 'GLM':
        graph = GraphGLM(lm_lambda_init=1e-2)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")

    vnodes = {}

    # --- 2. Variable Nodes ---
    for name in GT:
        if algo_type == 'GaBP':
            vn = VNodeGaBP(name, dims=[2], prior_std=100.0)
            vn.mu = initial_guesses[name].copy()
        elif algo_type in ('GN', 'LM', 'GLM'):
            # GLM uses initial_mu param (handled as mu internally like GN)
            vn = VNodeGN(name, dims=[2], prior_std=100.0) if algo_type != 'GLM' else VNodeGLM(name, dims=[2], initial_mu=initial_guesses[name].copy())
            vn.mu = initial_guesses[name].copy()
        else:  # EKI series
            if algo_type == 'EKI':
                vn = VNodeEKI(name, dims=[2], n_particles=1000, noise_std=1.0)
            elif algo_type == 'DEKI_cov':
                vn = VNodeDEKI(name, dims=[2], n_particles=1000, noise_std=1.0,
                               rho_init=1.0, rho_update_method='covariance')
            elif algo_type == 'DEKI_res':
                vn = VNodeDEKI(name, dims=[2], n_particles=1000, noise_std=1.0,
                               rho_init=1.0, rho_update_method='residual')
            mean_guess = initial_guesses[name].reshape(2, 1)
            vn.ensemble = mean_guess + np.random.randn(2, vn.n_particles) * 3.0

        graph.nodes.append(vn)
        vnodes[name] = vn

    # --- 3. Anchor Factors ---
    for a_name in anchors:
        if algo_type == 'EKI':
            f = AnchorFactorEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type == 'GaBP':
            f = AnchorFactorGaBP(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type in ('DEKI_cov', 'DEKI_res'):
            f = AnchorFactorDEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type == 'GLM':
            f = AnchorFactorGLM(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        else:  # GN, LM
            f = AnchorFactorGN(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
            
        graph.nodes.append(f)
        graph.connect(f, vnodes[a_name])

    # --- 4. Range-Only Factors ---
    for (n1, n2) in edges_list:
        dist = measurements[(n1, n2)]
        if algo_type == 'EKI':
            f = RangeOnlyFactorEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
        elif algo_type == 'GaBP':
            f = RangeOnlyFactorGaBP(f"F_RO_{n1}_{n2}", dist, ro_noise)
        elif algo_type in ('DEKI_cov', 'DEKI_res'):
            f = RangeOnlyFactorDEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
        elif algo_type == 'GLM':
            f = RangeOnlyFactorGLM(f"F_RO_{n1}_{n2}", dist, ro_noise)
        else:  # GN, LM
            f = RangeOnlyFactorGN(f"F_RO_{n1}_{n2}", dist, ro_noise)
            
        graph.nodes.append(f)
        graph.connect(f, vnodes[n1])
        graph.connect(f, vnodes[n2])

    return graph, vnodes


# =============================================================================
# 3. Main Testing & Plotting Logics
# =============================================================================

def run_test():
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise = build_scenario()

    algos = ['EKI', 'GaBP', 'DEKI_cov', 'DEKI_res', 'LM', 'GLM']

    graphs     = {}
    vnodes_d   = {}
    history    = {a: {n: [initial_guesses[n]] for n in unknowns} for a in algos}
    rmse       = {a: [] for a in algos}

    def get_estimate(vn, algo):
        if algo in ('GaBP', 'GN', 'LM', 'GLM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, algo):
        errs = [np.linalg.norm(get_estimate(vn_dict[n], algo) - GT[n])
                for n in unknowns]
        return float(np.mean(errs))

    # --- Init ---
    for algo in algos:
        graphs[algo], vnodes_d[algo] = build_graph(
            algo, GT, anchors, unknowns, edges_list,
            measurements, initial_guesses, anchor_noise, ro_noise
        )
        rmse[algo].append(get_rmse(vnodes_d[algo], algo))

    n_iter     = 100
    decay_rate = 0.5

    print("=" * 70)
    print(f"{'Iter':>4} | {'EKI':>6} | {'GaBP':>6} | {'DEKI_c':>6} | "
          f"{'DEKI_r':>6} | {'LM':>6} | {'GLM':>6}")
    print("=" * 70)

    # --- Run Optimization ---
    for i in range(n_iter):
        for algo in algos:
            graphs[algo].iterate(1)

            # EKI 계열 annealing
            if algo not in ('GaBP', 'GN', 'LM', 'GLM'):
                for vn in vnodes_d[algo].values():
                    vn.noise_std *= decay_rate

            for n in unknowns:
                history[algo][n].append(get_estimate(vnodes_d[algo][n], algo).copy())

            rmse[algo].append(get_rmse(vnodes_d[algo], algo))

        if (i + 1) % 10 == 0 or i == 0:
            vals = " | ".join(f"{rmse[a][-1]:6.3f}" for a in algos)
            print(f"{i+1:4d} | {vals}")

    print("=" * 70)
    final = " | ".join(f"{rmse[a][-1]:6.3f}" for a in algos)
    print(f"{'FINAL':>4} | {final}")
    print("=" * 70)


    # ==========================================
    # Plot 1: Trajectory Comparisons (2x3 Grid)
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_traj, axes_traj = plt.subplots(2, 3, figsize=(20, 12), dpi=100)
    # fig_traj.suptitle("Factor Graph Inference Comparison\n(2 anchors, range-only)", 
    #                   fontsize=12, fontweight='bold')

    gt_x = [GT[k][0] for k in GT]
    gt_y = [GT[k][1] for k in GT]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unknowns)))

    def plot_network(ax, title, hist=None, color_override=None):
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        
        # GT 노드: 테두리를 진하게 설정하여 명확하게 표현
        ax.scatter(gt_x, gt_y, c='lightgrey', s=100, zorder=2, 
                edgecolors='dimgrey', linewidth=1, label='GT')
        
        # Anchor: 크기를 키우고 테두리를 두껍게
        for a in anchors:
            ax.scatter(GT[a][0], GT[a][1], c='red', marker='s',
                    s=180, edgecolors='black', linewidth=2, zorder=5, label='Anchor')
        
        # 배경 연결선: 투명도를 약간 높이고(0.15 -> 0.25) 검은색 강조
        for n1, n2 in edges_list:
            ax.plot([GT[n1][0], GT[n2][0]], [GT[n1][1], GT[n2][1]],
                    'black', alpha=0.25, linewidth=1.2, zorder=1)
                    
        if hist:
            for idx, name in enumerate(unknowns):
                c = color_override if color_override else colors[idx]
                traj = np.array(hist[name])
                
                # 경로 선(Trajectory): 선 굵기를 키우고 투명도를 높여 선명하게 (alpha 0.6 -> 0.8)
                ax.plot(traj[:, 0], traj[:, 1], c=c, marker='.', markersize=4, 
                        alpha=0.8, linewidth=2.5, zorder=4)
                
                # 시작점(X): 테두리 강조
                ax.scatter(traj[0, 0], traj[0, 1], c=[c], marker='X', s=120, 
                        edgecolors='black', linewidth=1.5, zorder=6)
                
                # 종착점(별): 크기를 키우고 흰색 테두리를 추가하거나 검은 테두리를 두껍게
                ax.scatter(traj[-1, 0], traj[-1, 1], c=[c], marker='*', s=300, 
                        edgecolors='black', linewidth=1.5, zorder=7)
    
        # 축 설정
        ax.set_xlim(-5, 15)
        ax.set_ylim(-5, 15)
        ax.set_aspect('equal')
        # 눈금 폰트 크기 조정
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Assigning to specific subplots
    plot_network(axes_traj[0, 0], "Initial State")
    for name in unknowns:
        axes_traj[0, 0].scatter(initial_guesses[name][0], initial_guesses[name][1],
                                c='orange', marker='X', s=90, edgecolors='k')
    plot_network(axes_traj[0, 1], f"Naive Local LM  (final={rmse['GaBP'][-1]:.3f}m)", history['LM'])
    plot_network(axes_traj[0, 2], f"Global LM  (final={rmse['GLM'][-1]:.3f}m)", history['GLM'])
    plot_network(axes_traj[1, 0], f"Vanilla EKI  (final={rmse['EKI'][-1]:.3f}m)", history['EKI'])
    plot_network(axes_traj[1, 1], f"DEKI-Cov  (final={rmse['DEKI_cov'][-1]:.3f}m)", history['DEKI_cov'])
    plot_network(axes_traj[1, 2], f"DEKI-Res  (final={rmse['DEKI_res'][-1]:.3f}m)", history['DEKI_res'])
    
    # We plot Global LM on the last panel since it's our optimal upper bound

    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=0.5)


    # ==========================================
    # Plot 2: Convergence & RMSE (1x2 Grid) - Enhanced
    # ==========================================
    # figsize를 유지하면서 dpi를 높여 선명도를 확보합니다.
    fig_rmse, axes_rmse = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

    # [1] RMSE Convergence
    ax_r = axes_rmse[0]
    ax_r.set_title("RMSE Convergence", fontsize=16, fontweight='bold', pad=15)

    # 스타일 설정 유지 (선 굵기를 살짝 보강: lw=2.5)
    styles = {
        'EKI':      ('mediumblue',  '--',  'o', 2.5),
        'GaBP':     ('red',         '--',  'o', 2.5),
        'DEKI_cov': ('olivedrab',   '--',  'o', 2.5),
        'DEKI_res': ('purple',      '--',  'o', 2.5),
        'LM':       ('darkorange',  '--',  'o', 2.5),
        'GLM':      ('darkcyan',    '--',  'o', 2.5)
    }

    for algo, (col, ls, mk, lw) in styles.items():
        clipped = np.clip(rmse[algo], 0, 8.0)
        ax_r.plot(clipped, label=algo, color=col, linestyle=ls,
                marker=mk, markersize=5, linewidth=lw, markevery=10, 
                alpha=0.9) # 약간의 투명도로 겹치는 부분 시인성 확보

    ax_r.set_xlabel("Iteration", fontsize=16)
    ax_r.set_ylabel("Mean Error (m)", fontsize=16)
    ax_r.set_ylim(0, 3)

    # 범례(Legend) 강조: 폰트 키우고 테두리 추가
    ax_r.legend(fontsize=12, frameon=True, shadow=True, loc='upper right')
    ax_r.grid(True, linestyle=':', alpha=0.6)

    # [2] Final RMSE Bar Chart
    ax_b = axes_rmse[1]
    ax_b.set_title("Final RMSE Comparison", fontsize=16, fontweight='bold', pad=15)

    final_vals = [rmse[a][-1] for a in algos]
    bar_colors = [styles[a][0] for a in algos]

    # 바 차트: 테두리 두께 강화 (linewidth=1.5)
    bars = ax_b.bar(algos, final_vals, color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.85)

    # 바 상단 수치 표시: 폰트 크기 증대 및 위치 최적화
    for bar, val in zip(bars, final_vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='black')

    ax_b.set_ylabel("Mean Position Error (m)", fontsize=16)
    ax_b.set_ylim(0, max(final_vals) * 1.3) # 텍스트 공간 확보를 위해 y축 상단 여유 증가

    # X축 알고리즘 이름 폰트 크기 및 굵기 설정
    ax_b.set_xticklabels(algos, fontsize=16, fontweight='bold')
    ax_b.grid(axis='y', linestyle='--', alpha=0.5)

    fig_rmse.tight_layout(pad=3.0)
    plt.show()


# =============================================================================
# 4. Helper Vis Functions (Ensemble & Belief)
# =============================================================================

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


def run_ensemble_visualization():
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise = build_scenario()

    graph, vnodes = build_graph(
        'DEKI_res', GT, anchors, unknowns, edges_list,
        measurements, initial_guesses, anchor_noise, ro_noise
    )

    target_iters = [0, 1, 5, 10, 20, 100]
    snapshots = {}
    decay_rate = 0.1

    snapshots[0] = {n: vnodes[n].ensemble.copy() for n in unknowns}
    print("Running D-EKI (Covariance) Ensemble Simulation...")
    
    for i in range(1, max(target_iters) + 1):
        graph.iterate(1)
        for vn in vnodes.values():
            vn.noise_std *= decay_rate
            
        if i in target_iters:
            snapshots[i] = {n: vnodes[n].ensemble.copy() for n in unknowns}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("D-EKI (Covariance) Ensemble", 
                 fontsize=18, fontweight='bold')

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
        initial_guesses, anchor_noise, ro_noise = build_scenario()

    graph, vnodes = build_graph(
        'GaBP', GT, anchors, unknowns, edges_list,
        measurements, initial_guesses, anchor_noise, ro_noise
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
                # 리팩토링된 GBPCore.py 대응: Sigma가 내부에 존재함
                if hasattr(vn, 'Sigma'):
                    cov = vn.Sigma.copy()
                else:
                    cov = np.eye(2) * 9999.0 # 발산 시 무한대 취급
                
                snapshots[i][n] = {'mu': mu, 'cov': cov}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("GaBP Belief Evolution (Chi-Square Confidence Ellipses)", 
                 fontsize=18, fontweight='bold')

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
                if not draw_confidence_ellipse(mu, cov, ax, n_std=n_std, 
                                               edgecolor=colors[c_idx], 
                                               alpha=alpha, linestyle=linestyle, linewidth=2):
                    success = False
                    break
            
            if not success and idx > 0:
                ax.text(mu[0] + 0.5, mu[1] + 0.5, "Covariance Diverged!", 
                        color='red', fontsize=10, fontweight='bold')

        ax.set_xlim(-10, 20); ax.set_ylim(-10, 20); ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def run_anchor_sensitivity(num_anchors: int, n_trials: int = 100, n_iter: int = 30,
                           algo: str = 'DEKI_res', decay_rate: float = 0.5):
    """
    지정한 수(num_anchors)의 앵커를 9개 노드 중 랜덤으로 선택하여
    n_trials 번 독립 실험을 수행하고, 각 실험의 최종 RMSE를 집계합니다.

    Args:
        num_anchors : 매 실험마다 랜덤 선택할 앵커 수 (1~8)
        n_trials    : 반복 실험 횟수 (기본 100)
        n_iter      : 실험당 최적화 반복 횟수 (기본 30)
        algo        : 사용할 알고리즘 (기본 'DEKI_res')
        decay_rate  : EKI 계열 noise_std 감쇠율 (기본 0.5)

    Returns:
        trial_rmses : 각 실험의 최종 RMSE 리스트 (길이 n_trials)
        mean_rmse   : 전체 평균 RMSE
        std_rmse    : 전체 표준편차
    """
    all_nodes = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']

    if not (1 <= num_anchors <= len(all_nodes) - 1):
        raise ValueError(f"num_anchors는 1 이상 {len(all_nodes)-1} 이하여야 합니다.")

    def get_estimate(vn):
        if algo in ('GaBP', 'GN', 'LM', 'GLM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, unknowns_list, GT):
        errs = [np.linalg.norm(get_estimate(vn_dict[n]) - GT[n]) for n in unknowns_list]
        return float(np.mean(errs))

    trial_rmses = []

    print(f"\n{'='*60}")
    print(f"  Anchor Sensitivity Test")
    print(f"  Algorithm : {algo}")
    print(f"  Anchors   : {num_anchors} / {len(all_nodes)} nodes (random per trial)")
    print(f"  Trials    : {n_trials}  |  Iterations per trial : {n_iter}")
    print(f"{'='*60}")

    for trial in range(n_trials):
        # 매 실험마다 독립적인 랜덤 시드 (trial 번호 기반 → 재현 가능)
        np.random.seed(trial)

        # 앵커 랜덤 선택
        chosen_anchors = list(np.random.choice(all_nodes, size=num_anchors, replace=False))
        unknowns_trial = [v for v in all_nodes if v not in chosen_anchors]

        # GT / 엣지 / 측정값 생성 (매 실험마다 측정 노이즈 새로 샘플링)
        GT = {
            'V0': np.array([0.0,  0.0]),  'V1': np.array([5.0,  0.0]),  'V2': np.array([10.0,  0.0]),
            'V3': np.array([0.0,  5.0]),  'V4': np.array([5.0,  5.0]),  'V5': np.array([10.0,  5.0]),
            'V6': np.array([0.0, 10.0]),  'V7': np.array([5.0, 10.0]),  'V8': np.array([10.0, 10.0])
        }
        edges = [
            ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
            ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
        ]
        anchor_noise = 0.01
        ro_noise     = 0.5

        measurements = {}
        for n1, n2 in edges:
            dx = GT[n2][0] - GT[n1][0]
            dy = GT[n2][1] - GT[n1][1]
            dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * ro_noise
            measurements[(n1, n2)] = dist

        initial_guesses = {}
        for name in GT:
            if name in chosen_anchors:
                initial_guesses[name] = GT[name].copy()
            else:
                initial_guesses[name] = GT[name].copy() + np.random.randn(2) * 1.0

        # 그래프 빌드 및 최적화
        graph, vnodes = build_graph(
            algo, GT, chosen_anchors, unknowns_trial,
            edges, measurements, initial_guesses, anchor_noise, ro_noise
        )

        for _ in range(n_iter):
            graph.iterate(1)
            if algo not in ('GaBP', 'GN', 'LM', 'GLM'):
                for vn in vnodes.values():
                    vn.noise_std *= decay_rate

        # 최종 RMSE 기록
        final_rmse = get_rmse(vnodes, unknowns_trial, GT)
        trial_rmses.append(final_rmse)

        if (trial + 1) % 10 == 0:
            running_mean = float(np.mean(trial_rmses))
            print(f"  Trial {trial+1:3d}/{n_trials}  |  "
                  f"This RMSE: {final_rmse:.4f} m  |  "
                  f"Running mean: {running_mean:.4f} m  |  "
                  f"Anchors: {chosen_anchors}")

    mean_rmse = float(np.mean(trial_rmses))
    std_rmse  = float(np.std(trial_rmses))

    print(f"\n{'='*60}")
    print(f"  Results over {n_trials} trials")
    print(f"  Mean RMSE : {mean_rmse:.4f} m")
    print(f"  Std  RMSE : {std_rmse:.4f} m")
    print(f"  Min  RMSE : {min(trial_rmses):.4f} m")
    print(f"  Max  RMSE : {max(trial_rmses):.4f} m")
    print(f"{'='*60}\n")

    # --- 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Anchor Sensitivity  |  algo={algo},  anchors={num_anchors}/{len(all_nodes)},  "
        f"trials={n_trials},  iter={n_iter}",
        fontsize=13, fontweight='bold'
    )

    # [왼쪽] 실험별 RMSE 히스토그램
    ax_hist = axes[0]
    ax_hist.hist(trial_rmses, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax_hist.axvline(mean_rmse, color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {mean_rmse:.4f} m')
    ax_hist.axvline(mean_rmse + std_rmse, color='orange', linestyle=':', linewidth=1.5,
                    label=f'±1σ = {std_rmse:.4f} m')
    ax_hist.axvline(mean_rmse - std_rmse, color='orange', linestyle=':', linewidth=1.5)
    ax_hist.set_xlabel("Final RMSE (m)", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_title("RMSE Distribution across Trials", fontsize=13, fontweight='bold')
    ax_hist.legend(fontsize=11)
    ax_hist.grid(True, linestyle='--', alpha=0.5)

    # [오른쪽] 실험 순서별 RMSE + 누적 평균
    ax_line = axes[1]
    cumulative_mean = np.cumsum(trial_rmses) / np.arange(1, n_trials + 1)
    ax_line.plot(range(1, n_trials + 1), trial_rmses,
                 color='steelblue', alpha=0.4, linewidth=1.0, label='Per-trial RMSE')
    ax_line.plot(range(1, n_trials + 1), cumulative_mean,
                 color='red', linewidth=2.5, label='Cumulative mean')
    ax_line.axhline(mean_rmse, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax_line.set_xlabel("Trial", fontsize=12)
    ax_line.set_ylabel("Final RMSE (m)", fontsize=12)
    ax_line.set_title("Per-trial RMSE & Cumulative Mean", fontsize=13, fontweight='bold')
    ax_line.legend(fontsize=11)
    ax_line.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return trial_rmses, mean_rmse, std_rmse


def run_algo_comparison(num_anchors: int, n_trials: int = 100, n_iter: int = 30,
                        decay_rate: float = 0.5):
    """
    6개 알고리즘을 동일한 trial 조건(앵커 위치, 측정 노이즈)에서 비교합니다.
    매 trial마다 시나리오를 1회 생성하고, 6개 알고리즘이 동일한 조건을 공유합니다.

    Args:
        num_anchors : 랜덤 선택할 앵커 수 (1~8)
        n_trials    : 반복 실험 횟수 (기본 100)
        n_iter      : 실험당 최적화 반복 횟수 (기본 30)
        decay_rate  : EKI 계열 noise_std 감쇠율 (기본 0.5)

    Returns:
        results : { algo: {'rmses': [...], 'mean': float, 'std': float} }
    """
    algos     = ['EKI', 'GaBP', 'DEKI_cov', 'DEKI_res', 'LM', 'GLM']
    all_nodes = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']

    if not (1 <= num_anchors <= len(all_nodes) - 1):
        raise ValueError(f"num_anchors는 1 이상 {len(all_nodes)-1} 이하여야 합니다.")

    def get_estimate(vn, algo):
        if algo in ('GaBP', 'GN', 'LM', 'GLM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, unknowns_list, GT, algo):
        errs = [np.linalg.norm(get_estimate(vn_dict[n], algo) - GT[n])
                for n in unknowns_list]
        return float(np.mean(errs))

    trial_rmses = {a: [] for a in algos}

    print(f"\n{'='*70}")
    print(f"  Algorithm Comparison (same scenario per trial)")
    print(f"  Algorithms : {algos}")
    print(f"  Anchors    : {num_anchors} / {len(all_nodes)} nodes (random per trial)")
    print(f"  Trials     : {n_trials}  |  Iterations per trial : {n_iter}")
    print(f"{'='*70}")
    header = " | ".join(f"{a:>8}" for a in algos)
    print(f"{'Trial':>6} | {header}")
    print("-" * 70)

    for trial in range(n_trials):
        # 시드 고정 → 모든 알고리즘이 동일한 앵커/노이즈/초기값 공유
        np.random.seed(trial)

        chosen_anchors = list(np.random.choice(all_nodes, size=num_anchors, replace=False))
        unknowns_trial = [v for v in all_nodes if v not in chosen_anchors]

        GT = {
            'V0': np.array([0.0,  0.0]),  'V1': np.array([5.0,  0.0]),  'V2': np.array([10.0,  0.0]),
            'V3': np.array([0.0,  5.0]),  'V4': np.array([5.0,  5.0]),  'V5': np.array([10.0,  5.0]),
            'V6': np.array([0.0, 10.0]),  'V7': np.array([5.0, 10.0]),  'V8': np.array([10.0, 10.0])
        }
        edges = [
            ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
            ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
        ]
        anchor_noise = 0.01
        ro_noise     = 0.5

        measurements = {}
        for n1, n2 in edges:
            dx = GT[n2][0] - GT[n1][0]
            dy = GT[n2][1] - GT[n1][1]
            dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * ro_noise
            measurements[(n1, n2)] = dist

        initial_guesses = {}
        for name in GT:
            if name in chosen_anchors:
                initial_guesses[name] = GT[name].copy()
            else:
                initial_guesses[name] = GT[name].copy() + np.random.randn(2) * 1.0

        # 6개 알고리즘 동일 조건에서 순차 실행
        trial_row = {}
        for algo in algos:
            graph, vnodes = build_graph(
                algo, GT, chosen_anchors, unknowns_trial,
                edges, measurements, initial_guesses, anchor_noise, ro_noise
            )
            for _ in range(n_iter):
                graph.iterate(1)
                if algo not in ('GaBP', 'GN', 'LM', 'GLM'):
                    for vn in vnodes.values():
                        vn.noise_std *= decay_rate

            rmse_val = get_rmse(vnodes, unknowns_trial, GT, algo)
            trial_rmses[algo].append(rmse_val)
            trial_row[algo] = rmse_val

        if (trial + 1) % 10 == 0:
            row_str = " | ".join(f"{trial_row[a]:8.4f}" for a in algos)
            print(f"{trial+1:6d} | {row_str}   anchors={chosen_anchors}")

    # 통계 집계
    results = {}
    for algo in algos:
        rmses = trial_rmses[algo]
        results[algo] = {
            'rmses' : rmses,
            'mean'  : float(np.mean(rmses)),
            'std'   : float(np.std(rmses)),
            'min'   : float(np.min(rmses)),
            'max'   : float(np.max(rmses)),
        }

    print(f"\n{'='*70}")
    print(f"  Final Results ({n_trials} trials, {n_iter} iter, {num_anchors} anchors)")
    print(f"{'='*70}")
    print(f"  {'Algo':<10} {'Mean RMSE':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*46}")
    for algo in sorted(algos, key=lambda a: results[a]['mean']):
        r = results[algo]
        print(f"  {algo:<10} {r['mean']:10.4f} {r['std']:8.4f} {r['min']:8.4f} {r['max']:8.4f}")
    print(f"{'='*70}\n")

    # 시각화
    algo_styles = {
        'EKI'     : ('mediumblue',  'o'),
        'GaBP'    : ('red',         's'),
        'DEKI_cov': ('olivedrab',   '^'),
        'DEKI_res': ('purple',      'D'),
        'LM'      : ('darkorange',  'v'),
        'GLM'     : ('darkcyan',    'P'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Algorithm Comparison  |  anchors={num_anchors}/{len(all_nodes)},  "
        f"trials={n_trials},  iter={n_iter}",
        fontsize=14, fontweight='bold'
    )

    # [1] 평균 RMSE 바 차트 (오차막대 포함)
    ax_bar = axes[0]
    means  = [results[a]['mean'] for a in algos]
    stds   = [results[a]['std']  for a in algos]
    colors = [algo_styles[a][0]  for a in algos]
    bars   = ax_bar.bar(algos, means, yerr=stds, color=colors,
                        edgecolor='black', linewidth=1.2, alpha=0.85,
                        capsize=6, error_kw={'linewidth': 2})
    for bar, val in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax_bar.set_ylabel("Mean Final RMSE (m)", fontsize=12)
    ax_bar.set_title("Mean RMSE ± 1σ", fontsize=13, fontweight='bold')
    ax_bar.set_xticklabels(algos, fontsize=11, fontweight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    # [2] 누적 평균 수렴 곡선
    ax_conv = axes[1]
    for algo in algos:
        col, _ = algo_styles[algo]
        cumulative = np.cumsum(trial_rmses[algo]) / np.arange(1, n_trials + 1)
        ax_conv.plot(range(1, n_trials + 1), cumulative,
                     color=col, linewidth=2.0, label=algo)
    ax_conv.set_xlabel("Trial", fontsize=12)
    ax_conv.set_ylabel("Cumulative Mean RMSE (m)", fontsize=12)
    ax_conv.set_title("Cumulative Mean Convergence", fontsize=13, fontweight='bold')
    ax_conv.legend(fontsize=10, frameon=True)
    ax_conv.grid(True, linestyle='--', alpha=0.5)

    # [3] 박스플롯 (분포 비교)
    ax_box = axes[2]
    box_data = [trial_rmses[a] for a in algos]
    bp = ax_box.boxplot(box_data, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(algo_styles[algo][0])
        patch.set_alpha(0.75)
    ax_box.set_xticklabels(algos, fontsize=11, fontweight='bold')
    ax_box.set_ylabel("Final RMSE (m)", fontsize=12)
    ax_box.set_title("RMSE Distribution (Boxplot)", fontsize=13, fontweight='bold')
    ax_box.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # run_test()
    # run_ensemble_visualization()
    # run_gabp_visualization()
    # run_anchor_sensitivity(num_anchors=3)

    # 앵커 수를 원하는 값으로 지정하세요 (1~8)
    run_algo_comparison(num_anchors=4)
