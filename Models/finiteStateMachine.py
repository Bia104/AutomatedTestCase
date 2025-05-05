from statemachine import StateMachine, State

from Models.Enumerations.actionEnumeration import ActionEnum
from Models.Enumerations.locationEnumeration import LocationEnum


class FSM(StateMachine):
    idle = State("Idle", initial=True)
    active = State("Active")
    completed = State("Completed", final=True)

    start = idle.to(active)
    pause = active.to(idle)
    finish = active.to(completed)

    def __init__(self, final_state: LocationEnum, pass_loc: LocationEnum):
        super().__init__()
        if final_state != LocationEnum.InTaxi and pass_loc != LocationEnum.InTaxi:
            self.pass_location = pass_loc
            self.destination = final_state
        else:
            raise ValueError("Passenger Location and Destination can't be InTaxi")

    def transition(self, data: tuple[int, int], action: ActionEnum):

        # Starting the FSM
        if self.current_state == self.idle:
            self.start()
        moved = True

        # If Action is Move
        if any(action == loc for loc in ActionEnum if (loc != ActionEnum.pickup and loc != ActionEnum.dropoff)):
            data, moved = move_case(action, data)

        # If Action is Pickup or Dropoff
        else:
            if action == ActionEnum.pickup and data == self.pass_location.value and self.pass_location != LocationEnum.InTaxi:
                self.pass_location = LocationEnum.InTaxi
            elif action == ActionEnum.dropoff and data == self.destination.value and self.pass_location == LocationEnum.InTaxi:
                self.pass_location = self.destination
                self.finish()

        # If None of the Above are Satisfied
        if not moved:
            self.pause()
            raise ValueError("Invalid Action")
        return data

def move_case(action: ActionEnum, data: tuple[int, int]) -> (tuple[int, int], bool):
    col, row = data
    match action:
        case ActionEnum.south if col < 4:
            col += 1
        case ActionEnum.north if col > 0:
            col -= 1
        case ActionEnum.east if row < 4:
            row += 1
        case ActionEnum.west if row > 0:
            row -= 1
        case _:
            return data, False
    return (col, row), True
