from typing import Any, Callable

import numpy as np

import gymnasium
from gymnasium import spaces

from Models.Enumerations.actions import ActionEnum
from Models.Enumerations.locations import LocationEnum
from Models.finite_state_machine import FSM


class TaxiGymEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self):
        super().__init__()

        # Env Details
        self.grid_size = 5
        self.action_space = spaces.Discrete(6)  # 6 actions
        self.observation_space = spaces.Dict({
            "taxi_loc": spaces.MultiDiscrete([5, 5]),
            "destination": spaces.MultiDiscrete([5, 5]),
            "pickup_location": spaces.MultiDiscrete([6, 6]),
            "passenger_status": spaces.Discrete(2),  # 0 = not picked up, 1 = in taxi
            "current_objective": spaces.MultiDiscrete([5, 5]),
        })

        # Randomized Stats
        self.final_dest, self.pass_loc = get_random_pickup_and_dropoff()
        self.fsm = FSM(self.final_dest, self.pass_loc)
        self.start_pos = (int(np.random.choice(range(0, 5))), int(np.random.choice(range(0, 5))))

        # Changing Stats
        self.taxi_loc = self.start_pos
        self.steps = 0
        self.visited = set()
        self.current_objective = self.pass_loc.value
        self.track = 0
        self.hovering = 0

    def reset(self, *, seed=None, options=None) -> tuple[dict[str, tuple[int, int] | int], dict[str, str | int]]:
        super().reset(seed=seed)

        self.final_dest, self.pass_loc = get_random_pickup_and_dropoff()
        self.fsm = FSM(self.final_dest, self.pass_loc)

        self.start_pos = (int(np.random.choice(range(0,5))), int(np.random.choice(range(0,5))))
        self.taxi_loc = self.start_pos
        self.steps = 0
        self.visited = set()
        self.current_objective = self.pass_loc.value
        self.track = 0
        self.hovering = 0
        return {"taxi_loc": self.taxi_loc,
                "destination": self.final_dest.value,
                "pickup_location": self.pass_loc.value,
                "passenger_status": 0,
                "current_objective": self.current_objective}, {"passenger_status": self.fsm.pass_location.name,
                                       "destination": self.fsm.destination.name,
                                       "steps": self.steps}

    def step(self, action_id: int) -> tuple[
                                          dict[str, Callable[[], Any] | Callable[[], Any] | int | Any], int, bool, bool,
                                          dict[str, str | int]] | tuple[dict[str, Callable[[], Any] | Callable[[], Any] | LocationEnum | int | Any],
                                          int, bool, bool, dict[str, str | int]]:

        old_loc = self.taxi_loc
        action = ActionEnum(action_id)
        self.steps += 1

        try:
            self.taxi_loc = self.fsm.transition(self.taxi_loc, action)

            if self.taxi_loc == self.current_objective:
                self.hovering += 1

            # Reward / Penalty
            reward = (-200 if self.fsm.current_state == self.fsm.idle
                  else -250 if looping_move(self.taxi_loc, self.visited, self.current_objective, action, old_loc)
                  else -150 if invalid_special_move(self.pass_loc, action)
                  else 150 if self.taxi_loc == self.current_objective and self.hovering < 3
                  else 200 if new_corner(self.taxi_loc, self.visited)
                  else 100 if manhattan(self.taxi_loc, self.current_objective) < manhattan(old_loc, self.current_objective)
                  else 70 if closest_unvisited_corner(self.taxi_loc, self.visited) < closest_unvisited_corner(old_loc, self.visited)
                  else -5)

            if self.hovering > 2:
                reward = -60

            # Valid Pickup Case -> Reset List and Update Objective
            if valid_pickup(self.taxi_loc, self.current_objective, action) and self.pass_loc != LocationEnum.InTaxi:
                self.visited.clear()
                self.current_objective = self.final_dest.value
                self.pass_loc = LocationEnum.InTaxi
                self.hovering = 0
                reward = 500

            # Add Visited
            self.visited.add(self.taxi_loc)

            # Assign Boolean Done / Truncated
            done = self.fsm.current_state == self.fsm.completed
            truncated = self.steps > 250

            # Done / Truncated Cases
            if done:
                self.pass_loc = self.final_dest
                if self.steps < 180:
                    reward = 5000
                else:
                    reward = 1000
            elif truncated:
                reward = -5000

            # Rewards Scale Based on How Many Correct Moves in a Row are Made
            if reward > 0:
                self.track += 1
                reward += self.track * 50
            else:
                self.track = 0

            reward -= int(self.steps / 10)

            info = {
                "passenger_status": self.pass_loc.name,
                "destination": self.fsm.destination.name,
                "steps": self.steps
            }

            return {"taxi_loc": self.taxi_loc,
                "destination": self.final_dest.value,
                "pickup_location": self.pass_loc.value,
                "passenger_status": 0 if self.pass_loc.value != LocationEnum.InTaxi else 1,
                "current_objective": self.current_objective}, reward, done, truncated, info
        except ValueError:
            # Invalid Action
            reward = -100
            info = {
                "passenger_status": self.pass_loc.name,
                "destination": self.final_dest.name,
                "steps": self.steps
            }
            return {"taxi_loc": self.taxi_loc,
                "destination": self.final_dest.value,
                "pickup_location": self.pass_loc.value,
                "passenger_status": 0 if self.pass_loc.value != LocationEnum.InTaxi else 1,
                "current_objective": self.current_objective}, reward, False, False, info


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
    pickup = np.random.choice(valid)
    valid = [l for l in valid if l != pickup]
    dropoff = np.random.choice(valid)
    return pickup, dropoff

# Check if the taxi is in a new corner
def new_corner(taxi_loc: tuple[int, int], visited: set[tuple[int, int]]) -> bool:
    return taxi_loc not in visited and taxi_loc in {loc for loc in LocationEnum if loc != LocationEnum.InTaxi}

def closest_unvisited_corner(current_loc: tuple[int, int], visited: set[tuple[int, int]]):
    unvisited_corners = [loc for loc in LocationEnum if loc != LocationEnum.InTaxi and loc not in visited]
    if not unvisited_corners:
        return float('inf')  # No unvisited corners left
    return min(manhattan(current_loc, corner.value) for corner in unvisited_corners)
