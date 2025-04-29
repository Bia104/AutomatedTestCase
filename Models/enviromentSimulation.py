import random

import gymnasium as gym
from gym import spaces

from Models.Enumerations.actionEnumeration import ActionEnum
from Models.Enumerations.locationEnumeration import LocationEnum
from Models.finiteStateMachine import FSM


class TaxiGymEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self):
        super().__init__()

        # Env Details
        self.grid_size = 5
        self.action_space = spaces.Discrete(6)  # 6 actions
        self.observation_space = spaces.MultiDiscrete([5, 5])

        # Unchanging Stats
        self.start_pos = (random.choice(range(0,4)), random.choice(range(0,4)))

        # Randomized Stats
        self.final_dest, self.pass_loc = get_random_pickup_and_dropoff()
        self.fsm = FSM(self.final_dest, self.pass_loc)

        # Changing Stats
        self.taxi_loc = self.start_pos
        self.steps = 0
        self.visited = set()
        self.current_objective = self.pass_loc.value
        self.track = 0
        self.hovering = 0

    def reset(self, *, seed=None, options=None) -> (tuple[int, int], dict[str, str]):
        super().reset(seed=seed)

        self.final_dest, self.pass_loc = get_random_pickup_and_dropoff()
        self.fsm = FSM(self.final_dest, self.pass_loc)

        self.start_pos = (random.choice(range(0,4)), random.choice(range(0,4)))
        self.taxi_loc = self.start_pos
        self.steps = 0
        self.visited = set()
        self.current_objective = self.pass_loc.value
        self.track = 0
        self.hovering = 0
        return self.taxi_loc, {"passenger_status": self.fsm.pass_location.name,
                               "destination": self.fsm.destination.name,
                               "steps": self.steps}

    def step(self, action_id: int) -> (tuple[int, int], int, bool, bool, dict[str, str]):
        # Initializing
        old_loc = self.taxi_loc
        action = ActionEnum(action_id)

        # Making the Move in the FSM
        self.taxi_loc = self.fsm.transition(self.taxi_loc, action)
        self.steps += 1

        # Assuring the Agent is not Looping on the +1000 Reward
        if self.taxi_loc == self.current_objective:
            self.hovering += 1

        # Reward / Penalty
        reward = (-100 if self.fsm.current_state == self.fsm.idle
                  else -200 if looping_move(self.taxi_loc, self.visited, self.current_objective, action, old_loc)
                  else -150 if invalid_special_move(self.pass_loc, action)
                  else 1000 if self.taxi_loc == self.current_objective and self.hovering < 3
                  else 50 if manhattan(self.taxi_loc, self.current_objective) < manhattan(old_loc, self.current_objective)
                  else -10)

        # Valid Pickup Case -> Reset List and Update Objective
        if valid_pickup(self.taxi_loc, self.current_objective, action):
            self.visited.clear()
            self.current_objective = self.final_dest.value
            self.hovering = 0
            reward = 2000

        # Add Visited
        self.visited.add(self.taxi_loc)

        # Assign Boolean Sone / Truncated
        done = self.fsm.current_state == self.fsm.completed
        truncated = self.steps > 200

        # Done / Truncated Cases
        if done:
            print("COMPLETED!")
            if self.steps < 100:
                reward = 100000000
            else:
                reward = 50000
        elif truncated:
            reward = -10000000

        # Rewards Scale Based on How Many Correct Moves in a Row are Made
        if reward > 0:
            self.track += 1
            reward *= self.track
        else:
            self.track = 0

        reward -= int(self.steps / 10)

        info = {
            "passenger_status": self.fsm.pass_location.name,
            "destination": self.fsm.destination.name,
            "steps": self.steps
        }

        return self.taxi_loc, reward, done, truncated, info


def looping_move(taxi_loc: tuple[int, int], visited: set[tuple[int, int]],
                 current_objective: tuple[int, int], action: ActionEnum, old_loc: tuple[int, int]) -> bool:
    return (taxi_loc in visited
            and action != ActionEnum.pickup
            and action != ActionEnum.dropoff
            and manhattan(taxi_loc, current_objective) >= manhattan(old_loc, current_objective))

def invalid_special_move(pass_loc: LocationEnum, action: ActionEnum) -> bool:
    return ((action == ActionEnum.dropoff and pass_loc != LocationEnum.InTaxi) or
            (action == ActionEnum.pickup and pass_loc == LocationEnum.InTaxi))

def valid_pickup(taxi_loc: tuple[int, int], current_objective: tuple[int, int], action: ActionEnum) -> bool:
    return (action == ActionEnum.pickup
            and taxi_loc == current_objective)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_random_pickup_and_dropoff() -> tuple[LocationEnum, LocationEnum]:
    valid = [loc for loc in LocationEnum if loc != LocationEnum.InTaxi]
    pickup = random.choice(valid)
    dropoff = random.choice([l for l in valid if l != pickup])
    return pickup, dropoff