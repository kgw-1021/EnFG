import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
# 기존 모듈들 임포트
from src.graph.agent import Agent
from src.communication.shared_mem import CommunicationSharedMemory
from src.map.map_generator import EnvironmentMap, CircleObstacle, RectangleObstacle

def generate_narrow_corridor_scenario(num_agents: int, agent_width: float = 1.0, gap: float = 5.0, initial_v: float = 0.5):
    """
    원의 중심과 반지름을 기준으로 에이전트들을 원의 둘레에 균등하게 배치하고,
    정반대 편을 목표 지점으로 설정합니다.
    """
    start_poses = []
    goal_poses = []
    
    # 1. 왼쪽 그룹 (4명) -> 오른쪽으로 이동
    for i in range(num_agents // 2):
        # y 좌표는 중앙을 기준으로 위아래로 배치
        start_x = -gap
        start_y = (i - 1.5) * agent_width 
        
        goal_x = gap
        goal_y = (i - 1.5) * agent_width
        
        theta = 0.0  # 오른쪽(0도)을 바라봄
        
        start_poses.append(np.array([start_x, start_y, theta, initial_v]))
        goal_poses.append(np.array([goal_x, goal_y]))

        start_x = gap
        start_y = (i - 1.5) * agent_width
        
        goal_x = -gap
        goal_y = (i - 1.5) * agent_width
        
        theta = np.pi  # 왼쪽(180도)을 바라봄
        
        start_poses.append(np.array([start_x, start_y, theta, initial_v]))
        goal_poses.append(np.array([goal_x, goal_y]))
        
    return start_poses, goal_poses

def run_agent_process(agent_id: int, start_pos: np.ndarray, goal_pos: np.ndarray, 
                      num_agents: int, horizon: int, dt: float, max_iter: int, env_map: EnvironmentMap,
                      barrier: mp.Barrier):
    """
    각 로봇(에이전트)이 독립적인 프로세스에서 실행할 메인 워커(Worker) 함수.
    """
    print(f"[Agent {agent_id}] Process Started.")
    
    # 1. 프로세스 내부에서 자기 자신의 에이전트 객체 생성 (메모리 완전 독립)
    agent = Agent(agent_id=agent_id, start_pos=start_pos, goal_pos=goal_pos, n_particles=1000, horizon=horizon, dt=dt, env_map=env_map, safe_dist=1.0, dyn_weight=1e-4)
    
    # 타 에이전트와의 충돌 팩터 부착
    for other_id in range(num_agents):
        if other_id != agent_id:
            agent.attach_collision_factor(other_id)

    # 2. 공유 메모리 연결 (create=False)
    shm = CommunicationSharedMemory(num_agents, horizon, state_dim=4, create=False)
    
    # 3. 초기 궤적을 공유 메모리에 기록
    shm.write(agent_id, agent.extract_trajectory())
    
    # [동기화] 모든 프로세스가 초기화를 끝내고 공유 메모리에 첫 기록을 할 때까지 대기
    barrier.wait()

    # 4. Main Optimization Loop (ADMM/EKI)
    for iteration in range(max_iter):
        
        # [Step A] 공유 메모리에서 다른 로봇들의 현재 스텝 궤적 읽어오기
        # 데이터가 변경되는 것을 막기 위해 .copy() 로 로컬 메모리에 가져옵니다.
        shared_trajectories = {i: shm.array[i].copy() for i in range(num_agents)}
        
        # [Step B] 외부 정보(타 로봇 궤적) 업데이트 및 내 그래프 최적화 (1 스텝)
        agent.update_external_beliefs(shared_trajectories)
        agent.step(iterations=10)
        
        # [Step C] 계산된 나의 새로운 궤적을 공유 메모리에 브로드캐스트
        shm.write(agent_id, agent.extract_trajectory())
        
        # [동기화] 다른 로봇들이 연산을 끝내고 메모리를 업데이트할 때까지 대기
        # 이 장벽(Barrier)이 없으면 빠른 프로세스가 혼자 미래 스텝으로 달려나가 합의가 깨집니다.
        barrier.wait()
        
        if agent_id == 0 and (iteration + 1) % 5 == 0:
            print(f">>> Iteration {iteration + 1}/{max_iter} completed across all agents.")

    print(f"[Agent {agent_id}] Optimization Finished.")
    
    # 프로세스 종료 전 로컬 공유 메모리 참조 해제
    shm.shm.close()

def check_collision(trajectories: dict, robot_radius: float = 0.3) -> bool:
    """
    모든 에이전트의 궤적을 검사하여 충돌이 발생하는지 여부를 반환.
    충돌 기준: 두 에이전트 간의 유클리드 거리가 robot_radius 이하인 경우.
    """
    agent_ids = list(trajectories.keys())
    num_agents = len(agent_ids)
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            traj_i = trajectories[agent_ids[i]]
            traj_j = trajectories[agent_ids[j]]
            
            # 각 시간 스텝마다 두 궤적의 위치를 비교
            for t in range(traj_i.shape[0]):
                pos_i = traj_i[t, :2]  # x, y
                pos_j = traj_j[t, :2]  # x, y
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < robot_radius:
                    print(f"Collision detected between Agent {agent_ids[i]} and Agent {agent_ids[j]} at time step {t}.")
                    return True
    return False

def main():
    # 시뮬레이션 파라미터 세팅
    HORIZON = 50
    DT = 0.05
    MAX_ITER = 50
    NUM_AGENTS = 8
    
    start_poses, goal_poses = generate_narrow_corridor_scenario(
        num_agents=NUM_AGENTS, 
        agent_width=1.5, 
        gap=10.0, 
        initial_v=0.0
    )

    env = EnvironmentMap(penalty_value=1000.0, inflation_radius=0.5)
    obs1 = RectangleObstacle(x_min=-1.0, x_max=1.0, y_min=-10.0, y_max=-1.5)
    obs2 = RectangleObstacle(x_min=-1.0, x_max=1.0, y_min=1.5, y_max=10.0)

    env.add_obstacle(obs1)
    env.add_obstacle(obs2)

    # env.visualize(x_range=(-2, 12), y_range=(-2, 12), resolution=0.05)

    # ==========================================================
    # 1. 공유 자원(Shared Memory & Barrier) 생성
    # ==========================================================
    print("Initialize Shared Memory and Barrier...")
    shm_main = CommunicationSharedMemory(NUM_AGENTS, HORIZON, state_dim=4, create=True)
    barrier = mp.Barrier(NUM_AGENTS)

    # ==========================================================
    # 2. 멀티프로세스 생성 및 병렬 실행
    # ==========================================================
    start_time = time.time()
    processes = []
    for i in range(NUM_AGENTS):
        p = mp.Process(target=run_agent_process, args=(
            i, start_poses[i], goal_poses[i], NUM_AGENTS, HORIZON, DT, MAX_ITER, env, barrier
        ))
        processes.append(p)
        p.start()

    # 메인 프로세스는 모든 자식 프로세스가 끝날 때까지 대기
    for p in processes:
        p.join()

    end_time = time.time()
    print(f"All processes completed. Total execution time: {end_time - start_time:.2f} seconds.")
    final_trajectories = np.array([shm_main.array[i].copy() for i in range(NUM_AGENTS)])
    
    # (공유 메모리는 데이터를 다 뽑았으니 미리 닫아줍니다)
    shm_main.shm.close()
    shm_main.shm.unlink()
    print("Shared Memory cleared.")

    if not check_collision({f"A{i}": final_trajectories[i] for i in range(NUM_AGENTS)}):
        print("No collisions detected among the final trajectories.")

    # --- 애니메이션 설정 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Multi-Robot Trajectory Animation ({NUM_AGENTS} Agents)", fontsize=16, fontweight='bold')
    plt.xlim(-15, 15)
    plt.ylim(-10, 10)
    ax.set_aspect('equal') # 원형 교차가 찌그러지지 않도록 비율 고정
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    colors = plt.cm.rainbow(np.linspace(0, 1, NUM_AGENTS))

    # 0. 배경(정적 요소) 그리기: 장애물
    env.draw_obstacles(ax, color='dimgray', alpha=0.7)

    # 1. 배경(정적 요소) 그리기: 시작점과 도착점
    for i in range(NUM_AGENTS):
        c = colors[i]
        ax.scatter(start_poses[i][0], start_poses[i][1], color=c, marker='s', s=100, edgecolors='black')
        ax.scatter(goal_poses[i][0], goal_poses[i][1], color=c, marker='*', s=250, edgecolors='black')

    # 2. 동적 요소 초기화: 지나온 궤적(선)과 현재 위치(동그라미)
    lines = [ax.plot([], [], color=colors[i], linestyle='-', linewidth=2, alpha=0.5)[0] for i in range(NUM_AGENTS)]
    # 로봇 자체를 나타내는 큰 점
    robots = [ax.plot([], [], marker='o', color=colors[i], markersize=12, markeredgecolor='black', zorder=5)[0] for i in range(NUM_AGENTS)]

    def init():
        """ 애니메이션 초기화 함수 """
        for line, robot in zip(lines, robots):
            line.set_data([], [])
            robot.set_data([], [])
        return lines + robots

    def animate(t):
        """ 매 프레임(Time step t)마다 호출되어 화면을 업데이트하는 함수 """
        for i in range(NUM_AGENTS):
            # t 시점까지 지나온 궤적 데이터
            x_trail = final_trajectories[i, :t+1, 0]
            y_trail = final_trajectories[i, :t+1, 1]
            lines[i].set_data(x_trail, y_trail)
            
            # t 시점의 현재 로봇 위치
            x_curr = final_trajectories[i, t, 0]
            y_curr = final_trajectories[i, t, 1]
            # plot의 set_data는 리스트나 배열 형태를 요구하므로 []로 감쌈
            robots[i].set_data([x_curr], [y_curr])
            
        return lines + robots

    # 3. 애니메이션 객체 생성 (interval=200 은 프레임당 0.2초 대기를 의미)
    ani = animation.FuncAnimation(fig, animate, frames=HORIZON,
                                  init_func=init, blit=True, interval=200, repeat=True)

    # 4. 애니메이션 저장 (저장하지 않으려면 SAVE_ANIMATION = False 로 설정)
    SAVE_ANIMATION = True
    ANIMATION_PATH = "resource/path_planning_narrow_corridor.gif"  # .gif 또는 .mp4

    if SAVE_ANIMATION:
        # mp4로 저장하려면 ffmpeg 설치 필요: pip install ffmpeg-python 또는 conda install ffmpeg
        # gif로 저장하려면 pillow 설치 필요: pip install pillow
        ext = ANIMATION_PATH.split(".")[-1].lower()
        if ext == "mp4":
            writer = animation.FFMpegWriter(fps=5, bitrate=1800,
                                            metadata={"title": "Multi-Robot Trajectory"})
        else:  # gif
            writer = animation.PillowWriter(fps=5)

        print(f"Saving animation to '{ANIMATION_PATH}'...")
        ani.save(ANIMATION_PATH, writer=writer, dpi=100)
        print(f"Animation saved.")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    # plt.title("Decentralized Multi-Robot D-EKI (Multiprocessing)", fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_AGENTS))
    
    for i in range(NUM_AGENTS):
        # 최종 계산된 궤적을 메인 프로세스가 공유 메모리에서 읽어옴
        px, py = final_trajectories[i, :, 0], final_trajectories[i, :, 1]
        
        # 각 에이전트 고유의 색상 할당
        c = colors[i]
        
        env.draw_obstacles(plt.gca(), color='gray', alpha=0.5)
        plt.scatter(px[0], py[0], color=c, marker='s', s=100, edgecolors='black')
        # plt.scatter(goal_poses[i][0], goal_poses[i][1], color=c, marker='*', s=250, edgecolors='black', label=f"A{i} Goal")
        plt.plot(px, py, color=c, marker='o', markersize=4, linestyle='-', alpha=0.7, label=f"A{i} Path")
    plt.xlim(-15, 15)
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.tight_layout() # 그래프 잘림 방지
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    main()