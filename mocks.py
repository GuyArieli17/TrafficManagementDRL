class Agent:

    def __init__(self) -> None:
        self.memory = []
    
    def remember(self,state, action_phase, reward, next_state) -> None:
        self.memory.append(state, action_phase, reward, next_state)

    def choose_action(self,state) -> int:
        return 0
    
    def update_target_network() -> None:
        pass