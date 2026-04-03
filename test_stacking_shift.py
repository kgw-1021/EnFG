import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- Shift (분할 정복) 방식 임포트 ---
from src.graph.DEKICore import VNode as VNodeShift, UnaryFNode as UnaryShift, BinaryFNode as BinaryShift, FactorGraph as GraphShift
# --- Stacking (통합) 방식 임포트 ---
from src.graph.DEKICore_vanilla import VNode as VNodeStack, UnaryFNode as UnaryStack, BinaryFNode as BinaryStack, FactorGraph as GraphStack

# -------------------------------------------------------------------------
# 1. 커스텀 팩터 팩토리 (두 모듈 베이스에 맞게 동적 생성)
# -------------------------------------------------------------------------
def create_factors(UnaryFNodeBase, BinaryFNodeBase):
    class PriorFactor(UnaryFNodeBase):
        def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
            super().__init__(name, dims, gamma)
            self.y = y_meas
        def _error_function(self, x_ensemble: np.ndarray) -> np.ndarray:
            return x_ensemble - self.y

    class OdometryFactor(BinaryFNodeBase):
        def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
            super().__init__(name, dims, gamma)
            self.y = y_meas
        def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
            return (x1 - x0) - self.y
        def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m0 = mean0.reshape(-1, 1); m1 = mean1.reshape(-1, 1)
            return m1 - self.y, m0 + self.y

    class RangeBearingFactor(BinaryFNodeBase):
        def __init__(self, name: str, dims: list, gamma: np.ndarray, y_meas: np.ndarray):
            super().__init__(name, dims, gamma)
            self.y = y_meas  
        def _error_function(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
            diff = x1 - x0
            r_calc = np.linalg.norm(diff, axis=0, keepdims=True)  
            err_r = r_calc - self.y[0, 0]
            phi_calc = np.arctan2(diff[1, :], diff[0, :]).reshape(1, -1)  
            err_phi = (phi_calc - self.y[1, 0] + np.pi) % (2 * np.pi) - np.pi
            return np.vstack((err_r, err_phi))  
        def _compute_z_targets(self, mean0: np.ndarray, mean1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m0 = mean0.reshape(-1, 1); m1 = mean1.reshape(-1, 1)
            v_ideal = np.array([[self.y[0, 0] * np.cos(self.y[1, 0])], [self.y[0, 0] * np.sin(self.y[1, 0])]])
            return m1 - v_ideal, m0 + v_ideal
            
    return PriorFactor, OdometryFactor, RangeBearingFactor

# -------------------------------------------------------------------------
# 2. 단일 모드 시뮬레이션 실행 함수
# -------------------------------------------------------------------------
def run_slam_loop(mode="shift", n_iters=100):
    np.random.seed(42) # 동등한 오차를 위해 시드 고정

    # --- Ground Truth (사각형 루프 궤적) ---
    gt_poses = {
        'x0': np.array([[0.], [0.]]), 
        'x1': np.array([[10.], [0.]]), 
        'x2': np.array([[10.], [10.]]),
        'x3': np.array([[0.], [10.]])
    }
    gt_lms = {'l0': np.array([[5.], [5.]])} # 중앙 관측 포인트
    all_nodes = list(gt_poses.keys()) + list(gt_lms.keys())

    if mode == "shift":
        GraphClass, VNodeClass = GraphShift, VNodeShift
        PriorF, OdomF, RBF = create_factors(UnaryShift, BinaryShift)
    else:
        GraphClass, VNodeClass = GraphStack, VNodeStack
        PriorF, OdomF, RBF = create_factors(UnaryStack, BinaryStack)

    graph = GraphClass()
    vnodes = {}
    
    # 노드 생성 및 의도적 초기 오차 주입
    for name in all_nodes:
        vn = VNodeClass(name=name, dims=[2], n_particles=200, init_std=5.0, 
                        rho_init=1.0, rho_update_method='residual', debug_mode=True)
        error = np.random.randn(2, 1) * 4.0
        if name == 'x0': error *= 0.0 # x0는 원점 고정
        vn.ensemble += error
        vnodes[name] = vn

    # --- Factor 연결 ---
    gamma_prior = np.eye(2) * 1e-4
    gamma_odom = np.eye(2) * 0.1
    gamma_rb = np.diag([0.1, 0.01]) 

    graph.connect(PriorF("f_prior", [2], gamma_prior, gt_poses['x0']), vnodes['x0'])

    # Odometry Loop: x0->x1->x2->x3->x0 (마지막이 Loop Closure)
    odom_edges = [('x0', 'x1'), ('x1', 'x2'), ('x2', 'x3'), ('x3', 'x0')]
    for n0, n1 in odom_edges:
        meas = gt_poses[n1] - gt_poses[n0]
        f_odom = OdomF(f"f_odom_{n0}_{n1}", [2], gamma_odom, meas)
        graph.connect(f_odom, vnodes[n0])
        graph.connect(f_odom, vnodes[n1])

    # Range-Bearing: 모든 노드가 중앙 l0 관측
    for n_name, pose in gt_poses.items():
        diff = gt_lms['l0'] - pose
        meas = np.array([[np.linalg.norm(diff)], [np.arctan2(diff[1, 0], diff[0, 0])]])
        f_rb = RBF(f"f_rb_{n_name}_l0", [2], gamma_rb, meas)
        graph.connect(f_rb, vnodes[n_name])
        graph.connect(f_rb, vnodes['l0'])

    # 최적화 실행
    print(f"Running Optimization ({mode.upper()}) ...")
    for i in range(1, n_iters + 1):
        graph.iterate(1)

    return gt_poses, gt_lms, vnodes

# -------------------------------------------------------------------------
# 3. 비교 시각화
# -------------------------------------------------------------------------
if __name__ == "__main__":
    iters = 100
    # 시뮬레이션 실행
    shift_gt_p, shift_gt_l, shift_vn = run_slam_loop("shift", iters)
    stack_gt_p, stack_gt_l, stack_vn = run_slam_loop("stack", iters)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ✅ (NEW) 모든 팩터 잔차와 전체 잔차의 합을 추출하는 헬퍼 함수
    def get_all_factor_residuals(vn_dict):
        # 첫 번째 노드를 기준으로 반복 횟수(iters) 파악
        first_vn = list(vn_dict.values())[0]
        n_iterations = len(first_vn.debug_history['factor_residuals'])
        
        factor_history = {}
        total_history = [0.0] * n_iterations
        
        for t in range(n_iterations):
            current_step_factors = {}
            for vn_name, vn in vn_dict.items():
                res_dict = vn.debug_history['factor_residuals'][t]
                for f_name, err_matrix in res_dict.items():
                    # Binary 팩터는 두 노드에 메시지를 보내므로 중복 계산 방지
                    if f_name not in current_step_factors: 
                        mean_err = np.mean(err_matrix, axis=1)
                        current_step_factors[f_name] = np.linalg.norm(mean_err)
            
            # 해당 스텝의 전체 에러 합산 및 개별 팩터 기록
            step_total = 0.0
            for f_name, f_norm in current_step_factors.items():
                if f_name not in factor_history:
                    factor_history[f_name] = [0.0] * n_iterations
                factor_history[f_name][t] = f_norm
                step_total += f_norm
                
            total_history[t] = step_total
            
        return factor_history, total_history

    # 데이터 추출
    shift_f_hist, shift_f_total = get_all_factor_residuals(shift_vn)
    stack_f_hist, stack_f_total = get_all_factor_residuals(stack_vn)
    target_node = 'x3' # Primal/Dual 잔차를 관찰할 대표 노드

    # ──────────────────────────────────────────────────────────────────
    # [Figure 1] 궤적 및 맵 (최종 수렴 결과 비교) - 기존과 동일
    # ──────────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("Loop Closure SLAM Trajectory Comparison", fontsize=16, fontweight='bold')

    for i, (mode, vn_dict) in enumerate([("1. Shift (Operator Splitting)", shift_vn), 
                                         ("2. Stacking (Tightly Coupled)", stack_vn)]):
        ax = axes1[i]
        gt_px, gt_py = zip(*[shift_gt_p[k].flatten() for k in shift_gt_p])
        ax.plot(list(gt_px)+[gt_px[0]], list(gt_py)+[gt_py[0]], 'k--', marker='o', label='GT Poses', alpha=0.5)
        ax.scatter(shift_gt_l['l0'][0,0], shift_gt_l['l0'][1,0], c='gold', marker='*', s=200, edgecolors='k', label='GT Lm')

        est_px, est_py = zip(*[vn_dict[k].mean.flatten() for k in shift_gt_p])
        ax.plot(list(est_px)+[est_px[0]], list(est_py)+[est_py[0]], 'b-', marker='o', label='Est Poses', linewidth=2)
        ax.scatter(vn_dict['l0'].mean[0], vn_dict['l0'].mean[1], c='red', marker='x', s=100, label='Est Lm')

        ax.set_title(mode, fontsize=14)
        ax.legend()
        ax.axis('equal')

    # ──────────────────────────────────────────────────────────────────
    # [Figure 2] 최적화 수렴 지표 대시보드
    # ──────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
    fig2.suptitle("Convergence Metrics Analysis", fontsize=16, fontweight='bold')

    # [1행] Primal Residual
    axes2[0, 0].plot(shift_vn[target_node].debug_history['primal_residual'], 'b-', label='Shift')
    axes2[0, 0].set_title(f"Shift: Primal Residual (Node: {target_node})")
    axes2[0, 1].plot(stack_vn[target_node].debug_history['primal_residual'], 'r-', label='Stacking')
    axes2[0, 1].set_title(f"Stacking: Primal Residual (Node: {target_node})")

    # [2행] Dual Residual
    axes2[1, 0].plot(shift_vn[target_node].debug_history['dual_residual'], 'b-', label='Shift')
    axes2[1, 0].set_title(f"Shift: Dual Residual (Node: {target_node})")
    axes2[1, 1].plot(stack_vn[target_node].debug_history['dual_residual'], 'r-', label='Stacking')
    axes2[1, 1].set_title(f"Stacking: Dual Residual (Node: {target_node})")

    # [3행] Factor Residuals (모든 팩터 겹쳐 그리기 + 전체 합계)
    # 3-1. Shift 방식
    for f_name, f_norms in shift_f_hist.items():
        axes2[2, 0].plot(f_norms, color='gray', alpha=0.4, linewidth=1.0) # 개별 팩터는 얇고 투명하게
    axes2[2, 0].plot(shift_f_total, 'k-', linewidth=3.0, label='Total Factor Residual') # 전체 잔차는 굵고 진하게
    axes2[2, 0].set_title("Shift: All Factor Residuals & Total")

    # 3-2. Stacking 방식
    for f_name, f_norms in stack_f_hist.items():
        axes2[2, 1].plot(f_norms, color='gray', alpha=0.4, linewidth=1.0)
    axes2[2, 1].plot(stack_f_total, 'k-', linewidth=3.0, label='Total Factor Residual')
    axes2[2, 1].set_title("Stacking: All Factor Residuals & Total")

    # 공통 속성 적용
    for ax in axes2.flatten():
        ax.set_yscale('log')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Norm (Log Scale)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.show()