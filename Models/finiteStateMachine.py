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
        row, col = data
        if self.current_state == self.idle:
            self.start()
        match action:
            case ActionEnum.south if row < 4:
                row += 1
            case ActionEnum.north if row > 0:
                row -= 1
            case ActionEnum.east if col < 4:
                col += 1
            case ActionEnum.west if col > 0:
                col -= 1
            case ActionEnum.pickup if data == self.pass_location.value and self.pass_location != LocationEnum.InTaxi:
                self.pass_location = LocationEnum.InTaxi
                return data
            case ActionEnum.dropoff if data == self.destination.value and self.pass_location == LocationEnum.InTaxi:
                self.pass_location = self.destination
                self.finish()
                return data
            case _ if self.current_state == self.active:
                self.pause()
                return data
        return (row, col)

