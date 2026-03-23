import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PathCollection
import time

# 기존 모듈들 임포트
from src.graph.agent import Agent
from src.communication.shared_mem import CommunicationSharedMemory
from src.map.map_generator import EnvironmentMap, CircleObstacle, RectangleObstacle

def generate_circular_scenario(num_agents: int, center_x: float = 5.0, center_y: float = 5.0, radius: float = 5.0, initial_v: float = 0.5):
    """
    원의 중심과 반지름을 기준으로 에이전트들을 원의 둘레에 균등하게 배치하고,
    정반대 편을 목표 지점으로 설정합니다.
    """
    start_poses = []
    goal_poses = []
    
    for i in range(num_agents):
        angle = i * (2 * np.pi / num_agents)
        start_x = center_x + radius * np.cos(angle)
        start_y = center_y + radius * np.sin(angle)
        
        goal_x = center_x + radius * np.cos(angle + np.pi)
        goal_y = center_y + radius * np.sin(angle + np.pi)
        
        theta = np.arctan2(goal_y - start_y, goal_x - start_x)
        
        start_poses.append(np.array([start_x, start_y, theta, initial_v]))
        goal_poses.append(np.array([goal_x, goal_y]))
        
    return start_poses, goal_poses

def run_agent_process(agent_id: int, start_pos: np.ndarray, goal_pos: np.ndarray, 
                      num_agents: int, horizon: int, dt: float, max_iter: int, env_map: EnvironmentMap,
                      barrier: mp.Barrier, history_dict: dict):
    """
    각 로봇이 독립적인 프로세스에서 실행할 메인 워커(Worker) 함수.
    매 Iteration마다의 궤적을 history_dict에 저장합니다.
    """
    print(f"[Agent {agent_id}] Process Started.")
    
    agent = Agent(agent_id=agent_id, start_pos=start_pos, goal_pos=goal_pos, n_particles=1000, horizon=horizon, dt=dt, env_map=env_map, safe_dist=1.0)
    
    for other_id in range(num_agents):
        if other_id != agent_id:
            agent.attach_collision_factor(other_id)

    shm = CommunicationSharedMemory(num_agents, horizon, state_dim=4, create=False)
    
    mean_history = []     
    ensemble_history = []

    # 초기 궤적 기록 및 히스토리 저장
    mean_history.append(agent.extract_trajectory())
    ensemble_history.append(agent.get_ensemble_data()) # Shape: [Horizon, 4, N_Particles]

    shm.write(agent_id, agent.extract_trajectory())
    
    barrier.wait()

    for iteration in range(max_iter):
        shared_trajectories = {i: shm.array[i].copy() for i in range(num_agents)}
        
        agent.update_external_beliefs(shared_trajectories)
        agent.step(iterations=1)
        
        current_traj = agent.extract_trajectory()
        current_ensemble = agent.get_ensemble_data()
        shm.write(agent_id, current_traj)
        
        # Iteration이 끝날 때마다 현재 궤적을 히스토리에 추가
        mean_history.append(current_traj.copy())
        ensemble_history.append(current_ensemble.copy())
        barrier.wait()
        
        if agent_id == 0 and (iteration + 1) % 5 == 0:
            print(f">>> Iteration {iteration + 1}/{max_iter} completed across all agents.")

    print(f"[Agent {agent_id}] Optimization Finished.")
    
    # 프로세스 종료 전 히스토리 딕셔너리에 최종 데이터 업로드 (Manager dict 사용)
    history_dict[f"{agent_id}_mean"] = np.array(mean_history)
    history_dict[f"{agent_id}_ensemble"] = np.array(ensemble_history)
    shm.shm.close()

def main():
    # === 1. 파라미터 세팅 (에이전트 2개로 축소) ===
    HORIZON = 50
    DT = 0.1
    MAX_ITER = 20
    NUM_AGENTS = 2  
    
    start_poses, goal_poses = generate_circular_scenario(
        num_agents=NUM_AGENTS, 
        center_x=5.0, 
        center_y=5.0, 
        radius=15.0, 
        initial_v=0.0
    )

    env = EnvironmentMap(penalty_value=1000.0, inflation_radius=0.5)
    obs1 = CircleObstacle(cx=8.0, cy=7.0, radius=1.5)
    obs2 = CircleObstacle(cx=3.0, cy=3.0, radius=1.5)
    obs3 = CircleObstacle(cx=-1.0, cy=-1.0, radius=1.5)
    obs4 = CircleObstacle(cx=10.0, cy=3.0, radius=1.0)
    obs5 = RectangleObstacle(x_min=2.0, x_max=3.0, y_min=7.0, y_max=9.0)
    obs6 = CircleObstacle(cx=7.0, cy=-4.0, radius=1.5)
    obs7 = CircleObstacle(cx=-3.0, cy=6.0, radius=1.0)
    obs8 = CircleObstacle(cx=5.0, cy=11.5, radius=1.2)

    env.add_obstacle(obs1)
    env.add_obstacle(obs2)
    env.add_obstacle(obs3)
    env.add_obstacle(obs4)
    env.add_obstacle(obs5)
    env.add_obstacle(obs6)
    env.add_obstacle(obs7)
    env.add_obstacle(obs8)

    # === 2. 공유 자원 및 히스토리 저장소 생성 ===
    print("Initialize Shared Memory, Barrier, and Manager...")
    shm_main = CommunicationSharedMemory(NUM_AGENTS, HORIZON, state_dim=4, create=True)
    barrier = mp.Barrier(NUM_AGENTS)
    
    # Manager를 통해 멀티프로세스 환경에서 안전하게 리스트/딕셔너리 공유
    manager = mp.Manager()
    history_dict = manager.dict()

    # === 3. 병렬 프로세스 실행 ===
    start_time = time.time()
    processes = []
    for i in range(NUM_AGENTS):
        p = mp.Process(target=run_agent_process, args=(
            i, start_poses[i], goal_poses[i], NUM_AGENTS, HORIZON, DT, MAX_ITER, env, barrier, history_dict
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"All processes completed. Total execution time: {end_time - start_time:.2f} seconds.")
    
    shm_main.shm.close()
    shm_main.shm.unlink()

    # === 4. Iteration 애니메이션 설정 ===
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlim(-15, 25)
    plt.ylim(-15, 25)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    env.draw_obstacles(ax, color='dimgray', alpha=0.7)
    colors = plt.cm.rainbow(np.linspace(0, 1, NUM_AGENTS))
    
    # 시작점과 도착점 그리기
    for i in range(NUM_AGENTS):
        ax.scatter(start_poses[i][0], start_poses[i][1], color=colors[i], marker='s', s=100, edgecolors='black', label=f'Agent {i} Start')
        ax.scatter(goal_poses[i][0], goal_poses[i][1], color=colors[i], marker='*', s=250, edgecolors='black', label=f'Agent {i} Goal')
    ax.legend(loc='upper right')

    # 매 Iteration마다 그어질 전체 궤적 선 객체 리스트
    mean_lines = [ax.plot([], [], color=colors[i], linestyle='-', linewidth=3, alpha=0.8)[0] for i in range(NUM_AGENTS)]
    particle_clouds = []
    for i in range(NUM_AGENTS):
        # 초기에는 데이터가 없는 빈 산점도를 생성합니다.
        # 불확실성을 구름처럼 표현하기 위해 투명도(alpha)를 0.01로 매우 낮게 설정합니다.
        scat = ax.scatter([], [], color=colors[i], alpha=0.1, s=5, edgecolors='none', zorder=1)
        particle_clouds.append(scat)
    # 제목 텍스트 객체 (Iteration 표시용)
    title_text = ax.set_title("", fontsize=16, fontweight='bold')

    def init():
        for line in mean_lines:
            line.set_data([], [])
        for cloud in particle_clouds:
            cloud.set_offsets(np.empty((0, 2)))  # 빈 데이터로 초기화
        title_text.set_text("")
        return mean_lines + particle_clouds + [title_text]

    def animate(iteration):
        updated_artists = []
        
        for i in range(NUM_AGENTS):
            means = history_dict[f"{i}_mean"][iteration]             # [Horizon, 4]
            ensemble = history_dict[f"{i}_ensemble"][iteration]      # [Horizon, 4, N_Particles]
            
            # 1. 평균 궤적 (선) 업데이트
            mean_lines[i].set_data(means[:, 0], means[:, 1])
            updated_artists.append(mean_lines[i])
            
            # ==========================================================
            # [수정됨] 앙상블에서 x(인덱스 0), y(인덱스 1) 추출 후 1차원으로 쫙 펼치기
            # ==========================================================
            flat_px = ensemble[:, 0, :].ravel()
            flat_py = ensemble[:, 1, :].ravel()
            
            # Scatter plot의 데이터 업데이트
            particle_clouds[i].set_offsets(np.c_[flat_px, flat_py])
            updated_artists.append(particle_clouds[i])
            
        title_text.set_text(f"Trajectory Evolution - Iteration {iteration}/{MAX_ITER}")
        return mean_lines + particle_clouds + [title_text]

    # 프레임 수는 (MAX_ITER + 1) -> 초기 0번째 상태 포함
    ani = animation.FuncAnimation(fig, animate, frames=MAX_ITER + 1, 
                                  init_func=init, blit=True, interval=500, repeat=True)

    SAVE_ANIMATION = True
    ANIMATION_PATH = "resource/path_planning_2_agent_iteration.gif" 

    if SAVE_ANIMATION:
        writer = animation.PillowWriter(fps=5)
        print(f"Saving animation to '{ANIMATION_PATH}'...")
        ani.save(ANIMATION_PATH, writer=writer, dpi=100)
        print(f"Animation saved.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    main()