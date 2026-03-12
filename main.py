import numpy as np
import concurrent.futures
import time
import matplotlib.pyplot as plt

from src.communication.shared_mem import CommunicationSharedMemory
from src.graph.agent import Agent

def run_agent_step(agent: Agent, board_config: dict, iterations: int = 3):
    """cd
    각 CPU 코어(독립 프로세스)에서 실행될 워커 함수입니다.
    이 함수는 최상단(Top-level)에 정의되어야 멀티프로세싱 Pickling 오류가 발생하지 않습니다.
    """
    # 1. 공유 메모리 연결 (create=False)
    board = CommunicationSharedMemory(
        num_agents=board_config['num_agents'],
        horizon=board_config['horizon'],
        state_dim=board_config['state_dim'],
        name=board_config['name'],
        create=False
    )
    
    # 2. 통신 (읽기): 공유 메모리에서 상대방 궤적을 안전하게 복사(copy)해옵니다.
    current_beliefs = {}
    for other_id in agent.collision_factors.keys():
        # Numpy 배열의 복사본을 만들어 로컬 연산 중 오염을 방지합니다.
        current_beliefs[other_id] = board.array[other_id].copy()
        
    agent.update_external_beliefs(current_beliefs)
    
    # 3. 최적화 연산: EKI 메시지 패싱 및 궤적 업데이트
    agent.step(iterations=iterations)
    
    # 4. 통신 (쓰기): 계산된 새로운 궤적을 공유 메모리에 기록
    my_new_traj = agent.extract_trajectory()
    board.write(agent.id, my_new_traj)
    
    # 5. 메모리 연결 해제 (워커 프로세스의 연결만 끊음)
    board.shm.close()
    
    return agent


def main():
    # ==========================================
    # 1. 시뮬레이션 환경 및 에이전트 설정
    # ==========================================
    num_agents = 8
    horizon = 10
    state_dim = 4 # [px, py, vx, vy]
    sim_steps = 50
    
    # 에이전트들의 시작점과 목표점 (서로 교차하는 시나리오)
    starts = [
        np.array([-5.0,  5.0, 0.0, 0.0]),
        np.array([ 5.0,  5.0, 0.0, 0.0]),
        np.array([ -5.0, -5.0, 0.0, 0.0]),
        np.array([ 5.0,  -5.0, 0.0, 0.0]),
        np.array([ 0.0,  -5.0, 0.0, 0.0]),
        np.array([ 0.0,  5.0, 0.0, 0.0]),
        np.array([ -5.0,  0.0, 0.0, 0.0]),
        np.array([ 5.0,  0.0, 0.0, 0.0])  
    ]
    goals = [
        np.array([ 5.0, -5.0]), 
        np.array([-5.0, -5.0]), 
        np.array([ 5.0,  5.0]),  
        np.array([ -5.0,  5.0]), 
        np.array([ 0.0, 5.0]), 
        np.array([0.0,  -5.0]), 
        np.array([5.0, 0.0]), 
        np.array([-5.0, 0.0])  
    ]

    agents = []
    for i in range(num_agents):
        agent = Agent(agent_id=i, start_pos=starts[i], goal_pos=goals[i], horizon=horizon)
        agents.append(agent)
        
    # 서로를 인식하는 분산 충돌 팩터 연결
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                agents[i].attach_collision_factor(j)
                
    # ==========================================
    # 2. 공유 메모리 보드 생성
    # ==========================================
    board_name = "multi_robot_shm_board"
    board = CommunicationSharedMemory(
        num_agents=num_agents, 
        horizon=horizon, 
        state_dim=state_dim, 
        name=board_name, 
        create=True # 메인 프로세스에서 최초 1회 생성
    )
    
    # 초기 상태를 공유 메모리에 기록
    for agent in agents:
        board.write(agent.id, agent.extract_trajectory())
        
    board_config = {
        'num_agents': num_agents,
        'horizon': horizon,
        'state_dim': state_dim,
        'name': board_name
    }
    
    history_positions = {i: [] for i in range(num_agents)}

    # ==========================================
    #  Matplotlib 실시간 애니메이션 설정
    # ==========================================
    plt.ion()  # 대화형 모드(Interactive mode) 켜기
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown'] 
    
    # 정적 요소 (시작점, 목표점) 미리 그리기
    for i in range(num_agents):
        ax.scatter(starts[i][0], starts[i][1], color=colors[i], marker='o', s=100, alpha=0.5)
        ax.scatter(goals[i][0], goals[i][1], color=colors[i], marker='X', s=150)
        
    # 동적 요소 (지나온 궤적 선, 현재 위치 마커, 미래 예상 궤적) 객체 생성
    lines = [ax.plot([], [], color=colors[i], linestyle='-', linewidth=2)[0] for i in range(num_agents)]
    pred_lines = [ax.plot([], [], color=colors[i], linestyle=':', linewidth=1)[0] for i in range(num_agents)]
    points = [ax.plot([], [], color=colors[i], marker='s', markersize=10)[0] for i in range(num_agents)]

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title("Real-Time Decentralized Collision Avoidance")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    ax.legend()
    
    plt.show(block=False) # 창을 띄우되 코드는 계속 진행

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_agents) as executor:
            for step in range(sim_steps):
                step_start_time = time.time()
                
                # 1. 분산 병렬 연산 (MPC Step)
                futures = [executor.submit(run_agent_step, a, board_config, 10) for a in agents]
                for future in concurrent.futures.as_completed(futures):
                    updated_agent = future.result()
                    agents[updated_agent.id] = updated_agent 
                    
                # 2. 데이터 기록 및 실시간 플로팅 업데이트
                for i, agent in enumerate(agents):
                    # 현재 위치 (VNode[0]의 평균)
                    current_pos = agent.vnodes[0].ensemble.mean(axis=1)[:2]
                    history_positions[i].append(current_pos.copy())
                    
                    # 지나온 궤적 선 업데이트
                    hist_arr = np.array(history_positions[i])
                    lines[i].set_data(hist_arr[:, 0], hist_arr[:, 1])
                    
                    # 현재 위치 마커 업데이트 (set_data는 리스트 형태를 요구함)
                    points[i].set_data([current_pos[0]], [current_pos[1]])
                    
                    # 미래 예측 궤적 (VNode[0] ~ VNode[H]의 평균들) 선 업데이트
                    pred_traj = agent.extract_trajectory() # (Horizon, 4)
                    pred_lines[i].set_data(pred_traj[:, 0], pred_traj[:, 1])

                # 애니메이션 화면 갱신
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.5) # UI 업데이트를 위한 짧은 휴식 (이 값이 애니메이션 속도 조절)
                
                elapsed = time.time() - step_start_time
                print(f"Step {step:02d} | ⏱️ 소요 시간: {elapsed:.4f} sec")
                
    except KeyboardInterrupt:
        print("\n 사용자에 의해 시뮬레이션이 중단되었습니다.")
    except Exception as e:
        print(f"\n 에러 발생: {e}")
        
    finally:
        print(" 공유 메모리를 안전하게 해제합니다...")
        board.shm.close()
        board.shm.unlink() 
        
        plt.ioff()
        print(" 시뮬레이션 종료. 그래프 창을 닫으면 프로그램이 종료됩니다.")
        plt.show()

if __name__ == '__main__':
    main()