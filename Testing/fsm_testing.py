from Models.Enumerations.actions import ActionEnum
from Models.Enumerations.locations import LocationEnum
from Models.finite_state_machine import FSM


fsm = FSM(final_state=LocationEnum.Blue, pass_loc=LocationEnum.Red)
pos = (0, 0)

pos = fsm.transition(pos, ActionEnum.pickup)

for i in range(4):
    pos = fsm.transition(pos, ActionEnum.south)

for i in range(4):
    pos = fsm.transition(pos, ActionEnum.east)

pos = fsm.transition(pos, ActionEnum.dropoff)

print(fsm)
print("Final position:", pos)
print("FSM state:", fsm.current_state)  # Should be "Completed" if at destination
