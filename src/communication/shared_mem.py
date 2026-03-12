import numpy as np
from multiprocessing import shared_memory
from typing import Dict, List, Iterable

class CommunicationSharedMemory:
    def __init__(self, num_agents: int, horizon: int, state_dim: int = 4, name: str = "agent_trajectories", create: bool = False):
        self.name = name
        self.shape = (num_agents, horizon, state_dim)
        self.dtype = np.float64
        
        d_size = int(np.dtype(self.dtype).itemsize * np.prod(self.shape))
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=d_size)
            except FileExistsError:
                temp_shm = shared_memory.SharedMemory(name=self.name)
                temp_shm.unlink()
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=d_size)
                
            self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            self.array[:] = 0.0
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)
            self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def write(self, agent_id: int, trajectory: np.ndarray):
        self.array[agent_id] = trajectory

    def read(self, target_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        return {tid: self.array[tid].copy() for tid in target_ids}

    def close(self):
        self.shm.close()

    def cleanup(self):
        self.shm.close()
        self.shm.unlink()