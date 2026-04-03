import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# DEKICore와 Graph가 정의된 모듈에서 임포트합니다.
from src.graph.DEKICore import VNode, UnaryFNode, BinaryFNode, FactorGraph

# -------------------------------------------------------------------------
# 1. 커스텀 팩터 정의 (SLAM 비선형 테스트용)
# -------------------------------------------------------------------------

class PriorFactor(UnaryFNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
        super().__init__(name, dims, gamma)
        self.y = y_meas

    def _error_function(self, x_ensemble: np.ndarray) -> np.ndarray:
        return x_ensemble - self.y

class OdometryFactor(BinaryFNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
        super().__init__(name, dims, gamma)
        self.y = y_meas

    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return (x1 - x0) - self.y

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m0 = mean0.reshape(-1, 1)
        m1 = mean1.reshape(-1, 1)
        z0 = m1 - self.y
        z1 = m0 + self.y
        return z0, z1

class RangeBearingFactor(BinaryFNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
        super().__init__(name, dims, gamma)
        self.y = y_meas  # shape: (2, 1), [거리, 방위각]^T

    def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x1 - x0
        
        # 1. 거리(Range) 오차
        r_calc = np.linalg.norm(diff, axis=0, keepdims=True)  
        err_r = r_calc - self.y[0, 0]
        
        # 2. 방위각(Bearing) 오차
        phi_calc = np.arctan2(diff[1, :], diff[0, :]).reshape(1, -1)  
        err_phi = phi_calc - self.y[1, 0]
        
        # 각도 래핑 (-pi ~ pi)
        err_phi = (err_phi + np.pi) % (2 * np.pi) - np.pi
        
        return np.vstack((err_r, err_phi))  

    def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m0 = mean0.reshape(-1, 1)
        m1 = mean1.reshape(-1, 1)
        
        r_meas = self.y[0, 0]
        phi_meas = self.y[1, 0]
        v_ideal = np.array([[r_meas * np.cos(phi_meas)], 
                            [r_meas * np.sin(phi_meas)]])
        
        z0 = m1 - v_ideal
        z1 = m0 + v_ideal
        return z0, z1

# -------------------------------------------------------------------------
# 2. 메인 시뮬레이션 설정 및 실행
# -------------------------------------------------------------------------
def run_slam_toy_simulation():
    np.random.seed(42)

    # --- Ground Truth ---
    gt_poses = {'x0': np.array([[0.], [0.]]), 
                'x1': np.array([[10.], [0.]]), 
                'x2': np.array([[10.], [10.]])}
    gt_lms =   {'l0': np.array([[5.], [5.]]), 
                'l1': np.array([[15.], [5.]])}
    
    all_nodes = list(gt_poses.keys()) + list(gt_lms.keys())

    graph = FactorGraph()

    # --- Variable Node 생성 ---
    vnodes = {}
    for name in all_nodes:
        vn = VNode(name=name, dims=[2], n_particles=200, init_std=5.0, 
                   rho_init=1.0, rho_update_method='residual', debug_mode=True)
        vn.ensemble += np.random.randn(2, 1) * 4.0 # 의도적인 초기 오차
        vnodes[name] = vn

    # --- Factor 연결 ---
    gamma_prior = np.eye(2) * 1e-4
    gamma_odom = np.eye(2) * 0.1
    gamma_rb = np.diag([0.1, 0.01]) 

    f_prior = PriorFactor("f_prior", [2], gamma_prior, gt_poses['x0'])
    graph.connect(f_prior, vnodes['x0'])

    f_odom1 = OdometryFactor("f_odom1", [2], gamma_odom, gt_poses['x1'] - gt_poses['x0'])
    graph.connect(f_odom1, vnodes['x0'])
    graph.connect(f_odom1, vnodes['x1']) 

    f_odom2 = OdometryFactor("f_odom2", [2], gamma_odom, gt_poses['x2'] - gt_poses['x1'])
    graph.connect(f_odom2, vnodes['x1'])
    graph.connect(f_odom2, vnodes['x2'])

    def get_rb_meas(pose, lm):
        diff = lm - pose
        r = np.linalg.norm(diff)
        phi = np.arctan2(diff[1, 0], diff[0, 0])
        return np.array([[r], [phi]])

    f_rb1 = RangeBearingFactor("f_rb_x0_l0", [2], gamma_rb, get_rb_meas(gt_poses['x0'], gt_lms['l0']))
    graph.connect(f_rb1, vnodes['x0'])
    graph.connect(f_rb1, vnodes['l0'])

    f_rb2 = RangeBearingFactor("f_rb_x1_l0", [2], gamma_rb, get_rb_meas(gt_poses['x1'], gt_lms['l0']))
    graph.connect(f_rb2, vnodes['x1'])
    graph.connect(f_rb2, vnodes['l0'])

    f_rb3 = RangeBearingFactor("f_rb_x1_l1", [2], gamma_rb, get_rb_meas(gt_poses['x1'], gt_lms['l1']))
    graph.connect(f_rb3, vnodes['x1'])
    graph.connect(f_rb3, vnodes['l1'])

    f_rb4 = RangeBearingFactor("f_rb_x2_l1", [2], gamma_rb, get_rb_meas(gt_poses['x2'], gt_lms['l1']))
    graph.connect(f_rb4, vnodes['x2'])
    graph.connect(f_rb4, vnodes['l1'])

    # --- 시뮬레이션 실행 및 앙상블 히스토리 기록 ---
    print("Starting EKI-ADMM Optimization...")
    n_iters = 100
    target_iters = [0, 1, 5, 10, 20, 100] # 시각화할 타겟 이터레이션
    
    ensemble_history = {name: [vn.ensemble.copy()] for name, vn in vnodes.items()}
    
    for i in range(1, n_iters + 1):
        graph.iterate(1)
        for name, vn in vnodes.items():
            ensemble_history[name].append(vn.ensemble.copy())
            
    print("Optimization Completed!")

    # --- 각 Iteration별 팩터 오차(Residual) 계산 ---
    factor_error_history = {f.name: [] for f in graph.fnodes}
    for i in range(n_iters):
        for f in graph.fnodes:
            if isinstance(f, PriorFactor):
                v0_mean = vnodes[f.edges[0].get_other(f).name].debug_history['mean'][i].reshape(-1, 1)
                err = f._error_function(v0_mean)
            elif isinstance(f, (OdometryFactor, RangeBearingFactor)):
                v0_mean = vnodes[f.edges[0].get_other(f).name].debug_history['mean'][i].reshape(-1, 1)
                v1_mean = vnodes[f.edges[1].get_other(f).name].debug_history['mean'][i].reshape(-1, 1)
                err = f._error_function(v0_mean, v1_mean)
            factor_error_history[f.name].append(float(np.linalg.norm(err)))


    # =========================================================================
    # Figure 1: 기존 성능 평가 지표 (최종 Map, ADMM Residual, Factor Error)
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1 = plt.figure(figsize=(18, 5))
    fig1.suptitle("Figure 1: Optimization Metrics", fontsize=16, fontweight='bold')

    # 1. 최종 Map
    plt.subplot(1, 3, 1)
    gt_px, gt_py = zip(*[gt_poses[k].flatten() for k in gt_poses])
    plt.plot(gt_px, gt_py, 'k--', marker='o', label='GT Poses', markersize=8)
    gt_lx, gt_ly = zip(*[gt_lms[k].flatten() for k in gt_lms])
    plt.scatter(gt_lx, gt_ly, c='gold', marker='*', s=200, edgecolors='k', label='GT Landmarks')

    est_px, est_py = zip(*[vnodes[k].mean.flatten() for k in gt_poses])
    plt.plot(est_px, est_py, 'b-', marker='o', label='Est Poses', alpha=0.7)
    est_lx, est_ly = zip(*[vnodes[k].mean.flatten() for k in gt_lms])
    plt.scatter(est_lx, est_ly, c='red', marker='x', s=100, label='Est Landmarks')

    plt.title("Final Map (Mean Estimates)")
    plt.legend()
    plt.axis('equal')

    # 2. ADMM 잔차 (l0 기준)
    plt.subplot(1, 3, 2)
    l0_hist = vnodes['l0'].debug_history
    plt.plot(l0_hist['primal_residual'], label='Primal Residual (r)', c='b')
    plt.plot(l0_hist['dual_residual'], label='Dual Residual (s)', c='r', linestyle='--')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log)')
    plt.title("ADMM Residuals (Landmark 'l0')")
    plt.legend()

    # print(f"Primal Residual: {l0_hist['primal_residual']}, Dual Residual: {l0_hist['dual_residual']}")

    # 3. Factor별 Error Norm
    plt.subplot(1, 3, 3)
    for f_name, history in factor_error_history.items():
        linestyle = '--' if 'prior' in f_name else '-'
        plt.plot(history, label=f_name, linestyle=linestyle, linewidth=2, alpha=0.8)
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error Norm (log)')
    plt.title("Factor Residuals Over Iterations")
    plt.legend(fontsize='small', loc='upper right')

    plt.tight_layout()


    # =========================================================================
    # Figure 2: 타겟 스텝별 전체 앙상블 변화 과정 시각화 (2x3 Grid)
    # =========================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle("Figure 2: EKI-ADMM Ensemble Evolution", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    # 노드별 고유 색상 지정
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_nodes)))

    for idx, step in enumerate(target_iters):
        ax = axes[idx]
        ax.set_title(f"Iteration {step}", fontsize=14, fontweight='bold')
        
        # Ground Truth Odometry 선 그리기
        ax.plot([gt_poses['x0'][0,0], gt_poses['x1'][0,0]], 
                [gt_poses['x0'][1,0], gt_poses['x1'][1,0]], 'k--', alpha=0.3, linewidth=1.5)
        ax.plot([gt_poses['x1'][0,0], gt_poses['x2'][0,0]], 
                [gt_poses['x1'][1,0], gt_poses['x2'][1,0]], 'k--', alpha=0.3, linewidth=1.5)

        # Ground Truth 포인트 찍기
        for n_name, pose in gt_poses.items():
            ax.scatter(pose[0,0], pose[1,0], c='none', marker='s', s=100, edgecolors='k', zorder=5, alpha=0.5)
        for l_name, lm in gt_lms.items():
            ax.scatter(lm[0,0], lm[1,0], c='gold', marker='*', s=200, edgecolors='k', zorder=5)

        # 현재 Step의 앙상블 파티클 및 평균 그리기
        for c_idx, name in enumerate(all_nodes):
            particles = ensemble_history[name][step]
            
            # 파티클 구름 (alpha값을 낮게 주어 분포 확인)
            ax.scatter(particles[0, :], particles[1, :], color=colors[c_idx], s=10, alpha=0.15, zorder=3)
            
            # 앙상블 평균
            mean_est = particles.mean(axis=1)
            ax.scatter(mean_est[0], mean_est[1], color=colors[c_idx], marker='o', s=80, edgecolors='black', zorder=6, label=name if idx==0 else "")

        # 축 설정 및 범위 고정 (일관성 유지)
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 15)
        ax.set_aspect('equal')
        if idx == 0:
            ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_slam_toy_simulation()