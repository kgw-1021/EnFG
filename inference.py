import numpy as np
import matplotlib.pyplot as plt

# EKI 모듈 임포트
from src.graph.EKIcore import VNode as VNodeEKI, FactorGraph as GraphEKI
from src.graph.EKIcore import UnaryFNode as UnaryFNodeEKI, BinaryFNode as BinaryFNodeEKI

# GaBP 모듈 임포트
from src.graph.GBPCore import VNode as VNodeGaBP, FactorGraph as GraphGaBP
from src.graph.GBPCore import UnaryFNode as UnaryFNodeGaBP, BinaryFNode as BinaryFNodeGaBP

# D-EKI
from src.graph.DEKICore import VNode as VNodeDEKI, FactorGraph as GraphDEKI
from src.graph.DEKICore import UnaryFNode as UnaryFNodeDEKI, BinaryFNode as BinaryFNodeDEKI

# Gauss-Newton / LM
from src.graph.GNCore import VNode as VNodeGN, FactorGraph as GraphGN
from src.graph.GNCore import UnaryFNode as UnaryFNodeGN, BinaryFNode as BinaryFNodeGN


# ─────────────────────────────────────────────────────────────────────────────
# 1. 팩터 클래스 정의
# ─────────────────────────────────────────────────────────────────────────────

# --- EKI용 팩터 ---
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


# --- GaBP용 팩터 ---
class AnchorFactorGaBP(UnaryFNodeGaBP):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos

    def _error_function(self, x):
        return x - self.measured_pos

class RangeOnlyFactorGaBP(BinaryFNodeGaBP):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def _error_function(self, x0, x1):
        dx = x1[0] - x0[0]
        dy = x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dist - self.measured_dist])


# --- D-EKI용 팩터 ---
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


# --- GN / LM용 팩터 (동일 구현, 부모 클래스만 다름) ---
class AnchorFactorGN(UnaryFNodeGN):
    def __init__(self, name, measured_pos, noise_std):
        super().__init__(name, dims=[2], gamma=np.eye(2) * (noise_std ** 2))
        self.measured_pos = measured_pos.astype(float)

    def _error_function(self, x):
        # x: (2,)  ->  E: (2,)
        return x - self.measured_pos

class RangeOnlyFactorGN(BinaryFNodeGN):
    def __init__(self, name, measured_dist, noise_std):
        super().__init__(name, dims=[1], gamma=np.array([[noise_std ** 2]]))
        self.measured_dist = measured_dist

    def _error_function(self, x0, x1):
        # x0, x1: (2,)  ->  E: (1,)
        dx = x1[0] - x0[0]
        dy = x1[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dist - self.measured_dist])


# ─────────────────────────────────────────────────────────────────────────────
# 2. 시나리오 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_scenario():
    np.random.seed(100)
    GT = {
        'V0': np.array([0, 0]),  'V1': np.array([5, 0]),  'V2': np.array([10, 0]),
        'V3': np.array([0, 5]),  'V4': np.array([5, 5]),  'V5': np.array([10, 5]),
        'V6': np.array([0, 10]), 'V7': np.array([5, 10]), 'V8': np.array([10, 10])
    }

    anchors  = ['V6', 'V8', 'V0'] 
    unknowns = [k for k in GT if k not in anchors]

    edges = [
        ('V0','V1'), ('V1','V2'), ('V3','V4'), ('V4','V5'), ('V6','V7'), ('V7','V8'),
        ('V0','V3'), ('V3','V6'), ('V1','V4'), ('V4','V7'), ('V2','V5'), ('V5','V8')
    ]

    anchor_noise = 0.1
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. 그래프 빌더
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(algo_type, GT, anchors, unknowns, edges_list,
                measurements, initial_guesses, anchor_noise, ro_noise):
    """
    algo_type: 'EKI' | 'GaBP' | 'DEKI_cov' | 'DEKI_res' | 'GN' | 'LM'
    """

    # ── 그래프 인스턴스 ──────────────────────────────────────────────────────
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
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")

    vnodes = {}

    # ── 변수 노드 생성 ────────────────────────────────────────────────────────
    for name in GT:
        if algo_type == 'GaBP':
            vn = VNodeGaBP(name, dims=[2], prior_std=5.0)
            vn.mu = initial_guesses[name].copy()

        elif algo_type in ('GN', 'LM'):
            vn = VNodeGN(name, dims=[2], prior_std=100.0)
            vn.mu = initial_guesses[name].copy()

        else:  # EKI 계열
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

    # ── 앵커 팩터 ─────────────────────────────────────────────────────────────
    for a_name in anchors:
        if algo_type == 'EKI':
            f = AnchorFactorEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type == 'GaBP':
            f = AnchorFactorGaBP(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        elif algo_type in ('DEKI_cov', 'DEKI_res'):
            f = AnchorFactorDEKI(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        else:  # GN, LM
            f = AnchorFactorGN(f"F_Anchor_{a_name}", GT[a_name], anchor_noise)
        graph.nodes.append(f)
        graph.connect(f, vnodes[a_name])

    # ── Range-Only 팩터 ───────────────────────────────────────────────────────
    for (n1, n2) in edges_list:
        dist = measurements[(n1, n2)]
        if algo_type == 'EKI':
            f = RangeOnlyFactorEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
        elif algo_type == 'GaBP':
            f = RangeOnlyFactorGaBP(f"F_RO_{n1}_{n2}", dist, ro_noise)
        elif algo_type in ('DEKI_cov', 'DEKI_res'):
            f = RangeOnlyFactorDEKI(f"F_RO_{n1}_{n2}", dist, ro_noise)
        else:  # GN, LM
            f = RangeOnlyFactorGN(f"F_RO_{n1}_{n2}", dist, ro_noise)
        graph.nodes.append(f)
        graph.connect(f, vnodes[n1])
        graph.connect(f, vnodes[n2])

    return graph, vnodes


# ─────────────────────────────────────────────────────────────────────────────
# 4. 메인 실험
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    GT, anchors, unknowns, edges_list, measurements, \
        initial_guesses, anchor_noise, ro_noise = build_scenario()

    # 비교할 알고리즘
    algos = ['EKI', 
            #  'GaBP', 
             'DEKI_cov', 
             'DEKI_res',
            #  'GN', 
             'LM']

    graphs     = {}
    vnodes_d   = {}
    history    = {a: {n: [initial_guesses[n]] for n in unknowns} for a in algos}
    rmse       = {a: [] for a in algos}

    # ── RMSE 헬퍼 ────────────────────────────────────────────────────────────
    def get_estimate(vn, algo):
        if algo in ('GaBP', 'GN', 'LM'):
            return vn.mu
        return vn.ensemble.mean(axis=1)

    def get_rmse(vn_dict, algo):
        errs = [np.linalg.norm(get_estimate(vn_dict[n], algo) - GT[n])
                for n in unknowns]
        return float(np.mean(errs))

    # ── 초기화 ────────────────────────────────────────────────────────────────
    for algo in algos:
        graphs[algo], vnodes_d[algo] = build_graph(
            algo, GT, anchors, unknowns, edges_list,
            measurements, initial_guesses, anchor_noise, ro_noise
        )
        rmse[algo].append(get_rmse(vnodes_d[algo], algo))

    n_iter     = 100
    decay_rate = 0.9

    print("=" * 70)
    print(f"{'Iter':>4} | {'EKI':>6} | {'GaBP':>6} | {'DEKI_c':>6} | "
          f"{'DEKI_r':>6} | {'GN':>6} | {'LM':>6}")
    print("=" * 70)

    for i in range(n_iter):
        for algo in algos:
            graphs[algo].iterate(1)

            # EKI 계열 annealing
            if algo not in ('GaBP', 'GN', 'LM'):
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

    # ─────────────────────────────────────────────────────────────────────────
    # 5. 시각화 (3×3 그리드)
    # ─────────────────────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(22, 18))
    fig.suptitle("Factor Graph Inference Comparison\n"
                 "(2 anchors, range-only)", fontsize=14, fontweight='bold')

    gt_x = [GT[k][0] for k in GT]
    gt_y = [GT[k][1] for k in GT]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unknowns)))

    def plot_network(ax, title, hist=None, color_override=None):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.scatter(gt_x, gt_y, c='lightgrey', s=80, zorder=2, label='GT')
        for a in anchors:
            ax.scatter(GT[a][0], GT[a][1], c='red', marker='s',
                       s=140, edgecolors='k', zorder=5, label='Anchor')
        for n1, n2 in edges_list:
            ax.plot([GT[n1][0], GT[n2][0]], [GT[n1][1], GT[n2][1]],
                    'k--', alpha=0.15, linewidth=0.8)
        if hist:
            for idx, name in enumerate(unknowns):
                c = color_override if color_override else colors[idx]
                traj = np.array(hist[name])
                ax.plot(traj[:, 0], traj[:, 1], c=c,
                        marker='.', markersize=3, alpha=0.6)
                ax.scatter(traj[0, 0], traj[0, 1], c=[c],
                           marker='X', s=90, edgecolors='k', zorder=6)
                ax.scatter(traj[-1, 0], traj[-1, 1], c=[c],
                           marker='*', s=180, edgecolors='k', zorder=6)
        ax.set_xlim(-5, 15); ax.set_ylim(-5, 15)
        ax.set_aspect('equal')

    # [0,0] 초기 상태
    plot_network(axes[0, 0], "Initial State")
    for name in unknowns:
        axes[0, 0].scatter(initial_guesses[name][0], initial_guesses[name][1],
                           c='orange', marker='X', s=90, edgecolors='k')

    # [0,1] Vanilla EKI
    plot_network(axes[0, 1], f"Vanilla EKI  (final={rmse['EKI'][-1]:.3f}m)", history['EKI'])

    # [0,2] GaBP
    # plot_network(axes[0, 2], f"GaBP  (final={rmse['GaBP'][-1]:.3f}m)", history['GaBP'])

    # [1,0] DEKI cov
    plot_network(axes[1, 0], f"DEKI-Cov  (final={rmse['DEKI_cov'][-1]:.3f}m)", history['DEKI_cov'])

    # [1,1] DEKI res
    plot_network(axes[1, 1], f"DEKI-Res  (final={rmse['DEKI_res'][-1]:.3f}m)", history['DEKI_res'])

    # [1,2] Gauss-Newton
    # plot_network(axes[1, 2], f"Gauss-Newton  (final={rmse['GN'][-1]:.3f}m)", history['GN'])

    # [2,0] LM
    plot_network(axes[1, 2], f"Levenberg-Marquardt  (final={rmse['LM'][-1]:.3f}m)", history['LM'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    # [2,1] RMSE 수렴 곡선 (전체)
    ax_r = axes[0]
    ax_r.set_title("RMSE Convergence", fontsize=12, fontweight='bold')
    styles = {
        'EKI':      ('blue',   ':',  'o', 2),
        # 'GaBP':     ('red',    '--', 's', 2),
        'DEKI_cov': ('green',  '-',  'o', 3),
        'DEKI_res': ('purple', '-',  '^', 2),
        # 'GN':       ('darkorange', '-.', 'D', 2),
        'LM':       ('brown',  '-',  'P', 3),
    }
    for algo, (col, ls, mk, lw) in styles.items():
        clipped = np.clip(rmse[algo], 0, 8.0)
        ax_r.plot(clipped, label=algo, color=col, linestyle=ls,
                  marker=mk, markersize=3, linewidth=lw, markevery=10)
    ax_r.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='0.5m ref')
    ax_r.set_xlabel("Iteration"); ax_r.set_ylabel("Mean Error (m)")
    ax_r.set_ylim(0, 8); ax_r.legend(fontsize=10)

    # [2,2] 최종 RMSE 막대 그래프
    ax_b = axes[1]
    ax_b.set_title("Final RMSE Comparison", fontsize=12, fontweight='bold')
    final_vals = [rmse[a][-1] for a in algos]
    bar_colors = ['blue', 'red', 'green', 'purple', 'darkorange', 'brown']
    bars = ax_b.bar(algos, final_vals, color=bar_colors, edgecolor='k', alpha=0.8)
    ax_b.axhline(0.5, color='grey', linestyle='--', linewidth=1.2, label='0.5m ref (CRLB≈0.3m)')
    for bar, val in zip(bars, final_vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_b.set_ylabel("Mean Position Error (m)")
    ax_b.legend(fontsize=9)
    ax_b.set_ylim(0, max(final_vals) * 1.3 + 0.1)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run_test()