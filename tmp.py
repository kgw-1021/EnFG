"""
factor_graph.py
===============
NumPy 기반 완전 벡터화 EKI Factor Graph  —  멀티로봇 경로 계획

EKI 공식 (Iglesias 2013)
-----------------------
    관측 모델:   y = h(x) + η,   η ~ N(0, Γ)

    업데이트:
        C_xh = Cov(x, h(x))           (d × m)
        C_hh = Cov(h(x), h(x))        (m × m)
        K    = C_xh (C_hh + Γ)^{-1}   (d × m)
        x_new = x + K (y - h(x))

    핵심 주의:
        r = y - h(x) 를 그대로 쓰고 Cov(x, r) 을 계산하면
        Cov(x, y-h(x)) = -Cov(x, h(x)) 로 부호가 반전 → 발산.
        반드시 h(x) 와 y 를 분리해서 C_xh = Cov(x, h(x)) 로 계산.

상태 텐서 구조
--------------
    이전 버전의 핵심 버그: X (K, T, d, N) 에서 각 (k,t) 노드를 독립 업데이트.
    DynamicsFactor 가 x_t 만 업데이트하고 x_{t+1} 을 건드리지 않아
    인접 타임스텝 연결이 끊겨 궤적이 불연속.

    수정: X (K, T*d, N) — 에이전트별 전체 궤적을 하나의 big state vector 로.
    DynamicsFactor 는 [x_t; x_{t+1}] joint state 로 두 노드를 동시 업데이트.
    이렇게 해야 공분산이 전 타임스텝에 걸쳐 계산되어 연속 궤적이 보장됨.

    X[k]  : (T*d, N)   에이전트 k 의 전체 궤적 앙상블
    X[k, t*d:(t+1)*d, :] 가 타임스텝 t 의 상태 (d, N)

DynamicsFactor 업데이트 순서
-----------------------------
    1. GoalFactor   : terminal 노드를 목표로 당김
    2. Backward pass: t = T-2 → 0  (goal 정보를 앞으로 전파)
    3. Forward pass : t = 0 → T-2  (start 정보를 뒤로 전파)
    4. Hard start   : 위치 차원만 직접 덮어씀 (velocity 는 자유 변수)

StartFactor 설계
----------------
    start_pos 만 hard fix, velocity 는 EKI 가 결정.
    만약 velocity 까지 고정하면 dynamics constraint 와 충돌해 첫 스텝 점프 발생.
    (v=0 고정 시 dt=0.1 한 스텝에 이동 가능 거리 = 0 → t=1 이 크게 점프)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# =============================================================================
#  EKI 커널
# =============================================================================

def _eki_update(
    X:           np.ndarray,   # (d, N)
    HX:          np.ndarray,   # (m, N)  forward model h(X)
    y:           np.ndarray,   # (m,)    관측값 / 목표값
    gamma_scale: float = 1e-3,
) -> np.ndarray:               # (d, N)
    """
    단일 노드 EKI 업데이트.

    C_xh = Cov(x, h(x))     (d, m)
    C_hh = Var(h(x))         (m, m)
    Γ    = gamma_scale * trace(C_hh)/m * I
    K    = C_xh (C_hh + Γ)^{-1}
    """
    N   = X.shape[1]
    Xc  = X  - X.mean(axis=1, keepdims=True)
    HXc = HX - HX.mean(axis=1, keepdims=True)

    if np.abs(HXc).max() < 1e-12:
        return X

    C_xh = (Xc  @ HXc.T) / (N - 1)
    C_hh = (HXc @ HXc.T) / (N - 1)

    m   = C_hh.shape[0]
    eps = gamma_scale * np.trace(C_hh) / max(m, 1) + 1e-10
    C_hh_reg = C_hh + eps * np.eye(m)

    K = C_xh @ np.linalg.inv(C_hh_reg)
    return X + K @ (y[:, None] - HX)


def _joint_dyn_update(
    x_t:         np.ndarray,   # (d, N)  현재 상태
    x_t1:        np.ndarray,   # (d, N)  다음 상태
    f_xt:        np.ndarray,   # (d, N)  f(x_t) — dynamics 예측
    gamma_scale: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamics constraint h = f(x_t) - x_{t+1} → 0 을 만족하도록
    [x_t; x_{t+1}] joint state 를 동시 업데이트.

    h(joint) = f(x_t) - x_{t+1}        (d, N)
    y        = 0                         (d,)

    joint 의 C_xh 를 통해 x_t 와 x_{t+1} 이 서로를 고려하며 이동.
    """
    N     = x_t.shape[1]
    joint = np.vstack([x_t, x_t1])          # (2d, N)
    hx    = f_xt - x_t1                     # (d,  N)  위반량

    jc = joint - joint.mean(axis=1, keepdims=True)
    hc = hx    - hx.mean(axis=1,    keepdims=True)

    if np.abs(hc).max() < 1e-12:
        return x_t, x_t1

    d     = x_t.shape[0]
    C_xh  = (jc @ hc.T) / (N - 1)          # (2d, d)
    C_hh  = (hc @ hc.T) / (N - 1)          # (d,  d)
    eps   = gamma_scale * np.trace(C_hh) / max(d, 1) + 1e-10
    C_hh += eps * np.eye(d)

    K     = C_xh @ np.linalg.inv(C_hh)     # (2d, d)
    delta = K @ (-hx)                       # (2d, N)  y=0 이므로 0-hx

    return x_t + delta[:d], x_t1 + delta[d:]


# =============================================================================
#  Shared Memory 통신 레이어
# =============================================================================

class CommunicationSharedMemory:
    """
    멀티프로세스 간 에이전트 궤적 공유.

    shape : (num_agents, horizon, state_dim)
    각 에이전트는 자신의 슬롯(agent_id)에 평균 궤적을 write 하고,
    상대방의 슬롯을 read 한다.
    """

    def __init__(
        self,
        num_agents: int,
        horizon:    int,
        state_dim:  int  = 4,
        name:       str  = "agent_trajectories",
        create:     bool = False,
    ) -> None:
        self.name  = name
        self.shape = (num_agents, horizon, state_dim)
        self.dtype = np.float64

        d_size = int(np.dtype(self.dtype).itemsize * np.prod(self.shape))

        if create:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=d_size
                )
            except FileExistsError:
                tmp = shared_memory.SharedMemory(name=self.name)
                tmp.unlink()
                self.shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=d_size
                )
            self.array = np.ndarray(self.shape, dtype=self.dtype,
                                    buffer=self.shm.buf)
            self.array[:] = 0.0
        else:
            self.shm   = shared_memory.SharedMemory(name=self.name)
            self.array = np.ndarray(self.shape, dtype=self.dtype,
                                    buffer=self.shm.buf)

    def write(self, agent_id: int, trajectory: np.ndarray) -> None:
        """trajectory : (horizon, state_dim)"""
        self.array[agent_id] = trajectory

    def read(self, target_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """Returns {agent_id: trajectory (horizon, state_dim)}"""
        return {tid: self.array[tid].copy() for tid in target_ids}

    def close(self) -> None:
        self.shm.close()

    def cleanup(self) -> None:
        self.shm.close()
        self.shm.unlink()


# =============================================================================
#  팩터 기반 클래스
# =============================================================================

class FactorBase(ABC):
    """
    모든 팩터의 공통 기반.

    X[k] shape: (T*d, N)  — 에이전트 k 의 전체 궤적 앙상블

    팩터는 apply(X_k, k, d, T) 를 구현해
    X_k 를 in-place 로 업데이트한다.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight    = weight

    @abstractmethod
    def apply(
        self,
        X_k:         np.ndarray,   # (T*d, N)  in-place 업데이트
        k:           int,
        d:           int,
        T:           int,
        gamma_scale: float,
    ) -> None:
        """에이전트 k 의 궤적 앙상블 X_k 를 in-place 로 업데이트."""


# =============================================================================
#  Dynamics Factor
# =============================================================================

class DynamicsFactor(FactorBase):
    """
    시간축 Dynamics 팩터: x_{t+1} = f(x_t)

    [x_t; x_{t+1}] joint state 를 동시에 업데이트해
    인접 타임스텝 연결을 보장. 이것이 smooth 궤적의 핵심.

    업데이트 방향:
        forward  (t=0 → T-2): start 정보를 뒤로 전파
        backward (t=T-2 → 0): goal  정보를 앞으로 전파

    Subclass 예시
    -------------
    class SingleIntegrator(DynamicsFactor):
        def __init__(self, dt=0.1, weight=1.0):
            super().__init__(weight)
            self.dt = dt

        def _f(self, x_t):           # (d, N) → (d, N)
            xp = x_t.copy()
            xp[:2] += self.dt * x_t[2:4]
            return xp
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    @abstractmethod
    def _f(self, x_t: np.ndarray) -> np.ndarray:
        """
        단일 타임스텝 예측.

        x_t  : (d, N)  현재 상태 앙상블
        Returns f(x_t) : (d, N)
        """

    def apply(self, X_k, k, d, T, gamma_scale):
        # gamma 를 weight 로 조정: weight 높을수록 constraint 강하게
        g = gamma_scale / max(self.weight, 1e-6)

        # Forward pass: start → goal 방향으로 dynamics 전파
        for t in range(T - 1):
            x_t  = X_k[t*d:(t+1)*d, :]
            x_t1 = X_k[(t+1)*d:(t+2)*d, :]
            f_xt = self._f(x_t)
            new_t, new_t1 = _joint_dyn_update(x_t, x_t1, f_xt, g)
            X_k[t*d:(t+1)*d, :]     = new_t
            X_k[(t+1)*d:(t+2)*d, :] = new_t1

        # Backward pass: goal → start 방향으로 dynamics 전파
        for t in range(T - 2, -1, -1):
            x_t  = X_k[t*d:(t+1)*d, :]
            x_t1 = X_k[(t+1)*d:(t+2)*d, :]
            f_xt = self._f(x_t)
            new_t, new_t1 = _joint_dyn_update(x_t, x_t1, f_xt, g)
            X_k[t*d:(t+1)*d, :]     = new_t
            X_k[(t+1)*d:(t+2)*d, :] = new_t1


# =============================================================================
#  Obstacle Factor
# =============================================================================

class ObstacleFactor(FactorBase):
    """
    장애물 회피 팩터 (Unary, 특정 타임스텝 부분집합에 적용).

    h(x)  = 침투 측도  (≥ 0)
    y     = 0
    잔차 → 침투를 0으로 줄이는 방향으로 수렴

    Subclass 예시
    -------------
    class CircleObstacle(ObstacleFactor):
        def __init__(self, center, radius, **kw):
            super().__init__(**kw)
            self.center = np.array(center)
            self.radius = radius

        def _h(self, x):   # (d, N) → (m, N)
            diff = x[:2] - self.center[:, None]
            dist = np.linalg.norm(diff, axis=0, keepdims=True)  # (1, N)
            return np.maximum(0.0, self.radius - dist)
    """

    def __init__(
        self,
        weight:    float = 5.0,
        timesteps: Optional[List[int]] = None,
        agents:    Optional[List[int]] = None,
    ) -> None:
        super().__init__(weight)
        self.timesteps = timesteps
        self.agents    = agents

    @abstractmethod
    def _h(self, x: np.ndarray) -> np.ndarray:
        """x : (d, N) → (m, N)  침투 측도. 0 이 되길 원함."""

    def apply(self, X_k, k, d, T, gamma_scale):
        if self.agents is not None and k not in self.agents:
            return
        ts = self.timesteps if self.timesteps is not None else range(T)
        g  = gamma_scale / max(self.weight, 1e-6)
        y  = None   # probe 후 결정

        for t in ts:
            x_t  = X_k[t*d:(t+1)*d, :]
            hx   = self._h(x_t)              # (m, N)
            if y is None:
                y = np.zeros(hx.shape[0])
            X_k[t*d:(t+1)*d, :] = _eki_update(x_t, hx, y, g)


# =============================================================================
#  Collision Factor  (SHM 기반 Unary)
# =============================================================================

class CollisionFactor(FactorBase):
    """
    Inter-robot 충돌 회피 팩터 — Shared Memory 기반 단방향 Unary 팩터.

    SHM 에서 읽은 상대방 평균 궤적을 y 로 삼아 자신만 업데이트.

    h(x_t) = x_t[:2]  (자신의 위치)
    y      = 상대방 위치에서 safe_dist 만큼 떨어진 안전 목표 위치

    Parameters
    ----------
    self_id    : 이 에이전트의 인덱스
    target_ids : 충돌 회피 대상 에이전트 인덱스 목록
    safe_dist  : 유지해야 할 최소 거리
    comm       : CommunicationSharedMemory 인스턴스
    weight     : EKI 정보 가중치

    Runtime attach/detach
    ---------------------
    solver.add_factor(collision_fac)        # 활성화
    solver.remove_factor(collision_fac)     # 비활성화

    Subclass 예시
    -------------
    class MyCollision(CollisionFactor):
        def _h_pos(self, x_t):     # (d, N) → (2, N)
            return x_t[:2]
    """

    def __init__(
        self,
        self_id:    int,
        target_ids: List[int],
        safe_dist:  float,
        comm:       Optional[CommunicationSharedMemory] = None,
        weight:     float = 5.0,
        timesteps:  Optional[List[int]] = None,
    ) -> None:
        super().__init__(weight)
        self.self_id    = self_id
        self.target_ids = list(target_ids)
        self.safe_dist  = safe_dist
        self.comm       = comm
        self.timesteps  = timesteps

    @abstractmethod
    def _h_pos(self, x_t: np.ndarray) -> np.ndarray:
        """x_t : (d, N) → (2, N)  위치 추출"""

    def _safe_target(self, self_pos: np.ndarray, other_pos: np.ndarray) -> np.ndarray:
        """
        상대방으로부터 safe_dist 만큼 떨어진 방향의 목표 위치.

        self_pos, other_pos : (2,)
        Returns : (2,)
        """
        diff = self_pos - other_pos
        norm = np.linalg.norm(diff) + 1e-8
        if norm >= self.safe_dist:
            return self_pos   # 이미 안전 → 잔차 없음
        return other_pos + diff / norm * self.safe_dist

    def apply(self, X_k, k, d, T, gamma_scale):
        if k != self.self_id:
            return
        if self.comm is None:
            return

        other_trajs = self.comm.read(self.target_ids)  # {id: (T, d)}
        if not other_trajs:
            return

        ts = self.timesteps if self.timesteps is not None else range(T)
        g  = gamma_scale / max(self.weight, 1e-6)

        for t in ts:
            x_t      = X_k[t*d:(t+1)*d, :]
            self_pos = self._h_pos(x_t).mean(axis=1)   # (2,)

            # 가장 가까운 상대방 기준으로 y 결정
            min_dist = np.inf
            best_y   = self_pos.copy()
            for oid, traj in other_trajs.items():
                if t >= len(traj): continue
                other_pos = traj[t, :2]
                dist      = np.linalg.norm(self_pos - other_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_y   = self._safe_target(self_pos, other_pos)

            hx = self._h_pos(x_t)             # (2, N)
            X_k[t*d:(t+1)*d, :] = _eki_update(x_t, hx, best_y, g)


# =============================================================================
#  Goal Factor
# =============================================================================

class GoalFactor(FactorBase):
    """
    터미널 목표 팩터 (t = T-1).

    h(x_{T-1})  = 상태의 목표 관련 부분
    y           = goal

    Subclass 예시
    -------------
    class PositionGoal(GoalFactor):
        def __init__(self, goals, **kw):
            # goals : (K, 2) 또는 단일 에이전트는 (2,)
            super().__init__(**kw)
            self.goals = np.array(goals)

        def _h(self, x_term):          # (d, N) → (m, N)
            return x_term[:2]

        def _y(self, k):               # → (m,)
            return self.goals[k] if self.goals.ndim == 2 else self.goals
    """

    def __init__(
        self,
        weight:  float = 10.0,
        agents:  Optional[List[int]] = None,
    ) -> None:
        super().__init__(weight)
        self.agents = agents

    @abstractmethod
    def _h(self, x_term: np.ndarray) -> np.ndarray:
        """(d, N) → (m, N)"""

    @abstractmethod
    def _y(self, k: int) -> np.ndarray:
        """에이전트 k 의 목표값 → (m,)"""

    def apply(self, X_k, k, d, T, gamma_scale):
        if self.agents is not None and k not in self.agents:
            return
        g    = gamma_scale / max(self.weight, 1e-6)
        idx  = (T - 1) * d
        x_T  = X_k[idx:idx+d, :]
        hx   = self._h(x_T)
        y    = self._y(k)
        X_k[idx:idx+d, :] = _eki_update(x_T, hx, y, g)


# =============================================================================
#  Start Factor
# =============================================================================

class StartFactor(FactorBase):
    """
    출발점 고정 팩터 (t = 0).

    hard=True (기본값, 권장):
        위치 차원을 매 이터레이션 후 직접 덮어씀.
        velocity 는 자유 변수로 놓아 dynamics constraint 와 충돌 방지.
        (v=0 고정 시 dt 한 스텝에 이동 가능 = 0 → 첫 스텝 점프 발생)

    hard=False:
        EKI 소프트 제약. 관측 노이즈가 있는 출발점에 적합.

    Subclass 예시
    -------------
    class PositionStart(StartFactor):
        def __init__(self, start_pos, **kw):
            # start_pos: (2,)  위치만 고정
            super().__init__(**kw)
            self.start_pos = np.array(start_pos)

        def _pos_indices(self):   # 고정할 상태 차원 인덱스
            return slice(0, 2)    # px, py

        def _start_value(self):   # → (len(pos_indices), N) 또는 (len,)
            return self.start_pos
    """

    def __init__(
        self,
        weight:  float = 20.0,
        agents:  Optional[List[int]] = None,
        hard:    bool  = True,
    ) -> None:
        super().__init__(weight)
        self.agents = agents
        self.hard   = hard

    @abstractmethod
    def _pos_indices(self) -> slice:
        """고정할 상태 차원 슬라이스. 예: slice(0,2) → px,py"""

    @abstractmethod
    def _start_value(self) -> np.ndarray:
        """→ (m,)  출발 목표값"""

    def apply(self, X_k, k, d, T, gamma_scale):
        if self.agents is not None and k not in self.agents:
            return

        idx = self._pos_indices()
        val = self._start_value()   # (m,)

        if self.hard:
            # 직접 덮어씀: 모든 파티클을 동일 값으로
            X_k[idx, :] = val[:, None]
        else:
            g   = gamma_scale / max(self.weight, 1e-6)
            x_0 = X_k[:d, :]
            # h(x) = x[idx] → y = val
            hx  = x_0[idx, :]
            X_k[:d, :] = _eki_update(x_0, hx, val, g)


# =============================================================================
#  Solver
# =============================================================================

class MultiRobotEKISolver:
    """
    완전 벡터화 EKI 기반 멀티로봇 경로 계획 Solver.

    상태 구조
    ---------
    X : (K, T*d, N)
        K       — 에이전트 수
        T*d     — 전체 궤적 (타임스텝 × 상태차원)
        N       — 앙상블 파티클 수

    업데이트 순서 (매 이터레이션)
    ----------------------------
    1. GoalFactor      : terminal 노드를 목표로 당김
    2. DynamicsFactor  : backward + forward pass (양방향 전파)
    3. 기타 팩터       : ObstacleFactor, CollisionFactor 등
    4. StartFactor     : 출발 위치 고정 (hard 모드 시 직접 덮어씀)
    5. Inflation       : 앙상블 collapse 방지 노이즈

    팩터 런타임 탈착
    ----------------
    solver.add_factor(fac)
    solver.remove_factor(fac)
    solver.remove_factor_by_type(CollisionFactor)
    solver.active_factors

    Parameters
    ----------
    K, T, d     : 에이전트 수, 타임스텝 수, 단일 상태 차원
    N           : 앙상블 파티클 수
    noise_std   : inflation 노이즈 표준편차
    gamma_scale : EKI 정규화 기준값 (팩터 weight 로 나눠서 적용)
    max_workers : 에이전트 병렬 처리 스레드 수
    """

    def __init__(
        self,
        K:           int,
        T:           int,
        d:           int,
        N:           int   = 200,
        noise_std:   float = 1e-4,
        gamma_scale: float = 1e-3,
        max_workers: int   = 8,
    ) -> None:
        self.K, self.T, self.d, self.N = K, T, d, N
        self.noise_std   = noise_std
        self.gamma_scale = gamma_scale
        self.max_workers = max_workers
        self._factors:   List[FactorBase] = []

        # X[k] : (T*d, N) — 에이전트별 전체 궤적 앙상블
        self.X = np.random.randn(K, T * d, N)

    # ── 초기화 헬퍼 ──────────────────────────────────────────────────────────

    def init_linear(
        self,
        starts: np.ndarray,   # (K, d) 또는 (K, 2)  출발 위치/상태
        goals:  np.ndarray,   # (K, 2)               도착 위치
        pos_scale: float = 0.2,
        vel_scale: float = 0.1,
    ) -> None:
        """
        출발점 → 도착점 선형 보간으로 앙상블 초기화.
        완전 랜덤 초기화보다 수렴이 훨씬 빠름.
        """
        K, T, d, N = self.K, self.T, self.d, self.N
        s = np.array(starts)
        g = np.array(goals)
        if s.ndim == 1: s = s[None].repeat(K, axis=0)

        for k in range(K):
            for t in range(T):
                alpha = t / max(T - 1, 1)
                # 위치: 선형 보간
                pos_mean = (1 - alpha) * s[k, :2] + alpha * g[k, :2]
                self.X[k, t*d:t*d+2, :] = (
                    pos_mean[:, None] + pos_scale * np.random.randn(2, N)
                )
                # 속도: 상수 속도 추정
                if d > 2:
                    vel_mean = (g[k] - s[k, :2]) / max((T - 1) * 0.1, 1e-6)
                    self.X[k, t*d+2:t*d+4, :] = (
                        vel_mean[:, None] + vel_scale * np.random.randn(2, N)
                    )

    # ── 팩터 관리 ─────────────────────────────────────────────────────────────

    def add_factor(self, factor: FactorBase) -> None:
        if factor not in self._factors:
            self._factors.append(factor)

    def remove_factor(self, factor: FactorBase) -> None:
        try:
            self._factors.remove(factor)
        except ValueError:
            pass

    def remove_factor_by_type(self, factor_type: type) -> int:
        before = len(self._factors)
        self._factors = [f for f in self._factors
                         if not isinstance(f, factor_type)]
        return before - len(self._factors)

    @property
    def active_factors(self) -> List[FactorBase]:
        return list(self._factors)

    # ── EKI 스텝 ─────────────────────────────────────────────────────────────

    def _eki_step_agent(self, k: int) -> None:
        """
        에이전트 k 의 궤적 앙상블 X[k] 를 한 번 업데이트.

        업데이트 순서:
          1. GoalFactor   — terminal 당기기
          2. DynamicsFactor — backward + forward (내부에서 처리)
          3. 나머지 팩터  — Obstacle, Collision 등
          4. StartFactor  — 출발 위치 고정 (마지막에 적용해야 다른 팩터가 안 건드림)
        """
        X_k  = self.X[k]                   # view (T*d, N)
        d, T = self.d, self.T
        gs   = self.gamma_scale

        # 팩터를 타입별로 분리
        goal_facs  = [f for f in self._factors if isinstance(f, GoalFactor)]
        dyn_facs   = [f for f in self._factors if isinstance(f, DynamicsFactor)]
        start_facs = [f for f in self._factors if isinstance(f, StartFactor)]
        other_facs = [f for f in self._factors
                      if not isinstance(f, (GoalFactor, DynamicsFactor, StartFactor))]

        for f in goal_facs:  f.apply(X_k, k, d, T, gs)
        for f in dyn_facs:   f.apply(X_k, k, d, T, gs)
        for f in other_facs: f.apply(X_k, k, d, T, gs)
        for f in start_facs: f.apply(X_k, k, d, T, gs)

    def _eki_step(self) -> None:
        """모든 에이전트를 병렬로 한 스텝 업데이트."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(self._eki_step_agent, k) for k in range(self.K)]
        for f in futs:
            f.result()

        self.X += self.noise_std * np.random.randn(*self.X.shape)

        # StartFactor hard 모드 재적용 (inflation 이 덮어쓰기 전에 고정)
        for fac in self._factors:
            if isinstance(fac, StartFactor) and fac.hard:
                for k in range(self.K):
                    fac.apply(self.X[k], k, self.d, self.T, self.gamma_scale)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def solve(
        self,
        n_iter:  int   = 50,
        tol:     float = 1e-5,
        verbose: bool  = False,
    ) -> np.ndarray:
        """
        EKI 최적화.

        Returns
        -------
        X : (K, T*d, N)
        """
        for i in range(n_iter):
            X_prev = self.X.copy()
            self._eki_step()
            delta = np.abs(self.X - X_prev).max()
            if verbose:
                print(f"[iter {i+1:3d}]  max|ΔX| = {delta:.3e}  "
                      f"factors={len(self._factors)}")
            if delta < tol:
                if verbose:
                    print(f"  → Converged at iter {i+1}")
                break
        return self.X

    def step(self) -> np.ndarray:
        """단일 EKI 이터레이션 (온라인 루프용)."""
        self._eki_step()
        return self.X

    def mean_trajectory(self) -> np.ndarray:
        """
        앙상블 평균 궤적.

        Returns
        -------
        traj : (K, T, d)
        """
        return self.X.mean(axis=-1).reshape(self.K, self.T, self.d)

    def std_trajectory(self) -> np.ndarray:
        """앙상블 표준편차. Returns (K, T, d)"""
        return self.X.std(axis=-1).reshape(self.K, self.T, self.d)

    def check_consistency(self, dynamics_factor: DynamicsFactor) -> np.ndarray:
        """
        Dynamics 일관성 검사: pos[t+1] - (pos[t] + dt*vel[t]) 의 max 절댓값.

        Returns (K, T-1) 오차 행렬.
        """
        traj = self.mean_trajectory()    # (K, T, d)
        errs = np.zeros((self.K, self.T - 1))
        for k in range(self.K):
            for t in range(self.T - 1):
                x_t  = traj[k, t]
                x_t1 = traj[k, t + 1]
                f_xt = dynamics_factor._f(x_t[:, None])[:, 0]
                errs[k, t] = np.abs(x_t1 - f_xt).max()
        return errs


# =============================================================================
#  사용 예시
# =============================================================================

if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 구체 팩터 정의 ────────────────────────────────────────────────────────

    class SingleIntegratorDynamics(DynamicsFactor):
        """x_{t+1} = x_t + dt * v_t   state: [px, py, vx, vy]"""
        def __init__(self, dt=0.1, **kw):
            super().__init__(**kw); self.dt = dt

        def _f(self, x_t):   # (d, N) → (d, N)
            xp = x_t.copy()
            xp[:2] += self.dt * x_t[2:4]
            return xp

    class CircleObstacle(ObstacleFactor):
        def __init__(self, center, radius, **kw):
            super().__init__(**kw)
            self.center = np.array(center, dtype=float)
            self.radius = radius

        def _h(self, x):     # (d, N) → (1, N)
            diff = x[:2] - self.center[:, None]
            dist = np.linalg.norm(diff, axis=0, keepdims=True)
            return np.maximum(0.0, self.radius - dist)

    class PositionGoal(GoalFactor):
        def __init__(self, goals, **kw):
            super().__init__(**kw)
            self.goals = np.array(goals, dtype=float)   # (K, 2)

        def _h(self, x_term):   # (d, N) → (2, N)
            return x_term[:2]

        def _y(self, k):        # → (2,)
            return self.goals[k]

    class PositionStart(StartFactor):
        """출발 위치만 고정 (velocity 는 자유 — dynamics 가 결정)."""
        def __init__(self, start_pos, **kw):
            super().__init__(**kw)
            self.start_pos = np.array(start_pos, dtype=float)   # (2,)

        def _pos_indices(self):
            return slice(0, 2)   # px, py

        def _start_value(self):
            return self.start_pos

    class PositionCollision(CollisionFactor):
        def _h_pos(self, x_t):   # (d, N) → (2, N)
            return x_t[:2]

    # ── 문제 설정 ─────────────────────────────────────────────────────────────

    K, T, d, N = 4, 20, 4, 300
    dt = 0.1

    starts = np.array([[0., 0.], [0., 2.], [0., -2.], [0., 4.]])
    goals  = np.array([[5., 2.], [5., 0.], [5., 4.], [5., -2.]])

    solver = MultiRobotEKISolver(
        K=K, T=T, d=d, N=N,
        noise_std=1e-4, gamma_scale=1e-3, max_workers=K
    )
    solver.init_linear(starts, goals)   # 선형 보간 초기화

    dyn_fac = SingleIntegratorDynamics(dt=dt, weight=1.0)
    solver.add_factor(dyn_fac)
    solver.add_factor(CircleObstacle([2.5, 1.], radius=0.7, weight=5.0))
    solver.add_factor(PositionGoal(goals, weight=10.0))   # 모든 에이전트, 각자 다른 goal
    for k in range(K):
        solver.add_factor(PositionStart(starts[k], hard=True, agents=[k]))

    print(f"State shape : {solver.X.shape}  (K={K}, T={T}, d={d}, N={N})")
    print(f"Active factors: {len(solver.active_factors)}")

    t0 = time.perf_counter()
    solver.solve(n_iter=50, verbose=True)
    elapsed = time.perf_counter() - t0

    traj = solver.mean_trajectory()   # (K, T, d)
    errs = solver.check_consistency(dyn_fac)

    print(f"\nSolve time : {elapsed:.2f}s")
    print(f"Max dynamics err : {errs.max():.5f}")
    print(f"Mean dynamics err: {errs.mean():.5f}")
    for k in range(K):
        print(f"  Agent {k}"
              f"  start={traj[k,0,:2].round(3)} (target={starts[k]})"
              f"  →  goal={traj[k,-1,:2].round(3)} (target={goals[k]})")

    # ── 시각화 ────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: XY 궤적
    ax = axes[0]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k in range(K):
        xs = traj[k, :, 0]
        ys = traj[k, :, 1]
        ax.plot(xs, ys, "-o", color=colors[k], markersize=4,
                label=f"Agent {k}")
        ax.plot(xs[0],  ys[0],  "s", color=colors[k], markersize=10)
        ax.plot(xs[-1], ys[-1], "*", color=colors[k], markersize=12)
        # 앙상블 퍼짐 (일부 파티클)
        for n in range(0, N, N // 20):
            px = solver.X[k, ::d, n]    # px 채널만
            py = solver.X[k, 1::d, n]
            ax.plot(px, py, alpha=0.05, color=colors[k], linewidth=0.5)

    # 장애물
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(2.5 + 0.7*np.cos(theta), 1. + 0.7*np.sin(theta),
            alpha=0.3, color="gray", label="Obstacle")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("XY Trajectories (mean ± ensemble)")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True)

    # 오른쪽: Dynamics consistency error per timestep
    ax2 = axes[1]
    for k in range(K):
        ax2.plot(errs[k], "-o", color=colors[k], markersize=4,
                 label=f"Agent {k}")
    ax2.set_xlabel("timestep t"); ax2.set_ylabel("|x_{t+1} - f(x_t)|_∞")
    ax2.set_title("Dynamics Consistency Error")
    ax2.legend(); ax2.grid(True); ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=150)
    print("\nPlot saved: trajectories.png")