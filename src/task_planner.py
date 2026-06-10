from dataclasses import dataclass
from collections import deque

@dataclass(frozen=True)
class State: # Class to function as a container for the different stated that we can have
    objects_locations: tuple
    holding: str | None
    armLocation: str

    @staticmethod # Static method to convert from dictionary to state.
    def from_dict(objects_locations: dict, holding: str | None, armLocation: str) -> "State":
        return State(
            objects_locations=tuple(sorted(objects_locations.items())),
            holding=holding,
            armLocation=armLocation,
        )

    def locations_dict(self) -> dict: # Method to convert the actual state locations into a dictionary.
        return dict(self.objects_locations)
    
'''
Creation of the methods that will be mapped to the series of different actions that can be performed.
'''
# Action to move arm to a given destination -> Return None if already there.
def action_move_to(state: State, newLocation: str) -> State | None:
    if state.armLocation == newLocation:
        return None
    else:
        return State(
            objects_locations=state.objects_locations,
            holding=state.holding,
            armLocation=newLocation
        )

# Action to pick an object -> Can only pick it if not picked, gripper empty, and gripper and object at location.
def action_pick(state: State, obj: str, objectLocation: str) -> State | None:

    if (state.holding) or (state.armLocation != objectLocation) \
    or (state.locations_dict().get(obj) != objectLocation): # Preconditions to meet.
        return None
    else:                                                   # Resulting effects -> Holding object, object not at that location anymore, arm at object location.
        holding = obj
        temp_locations = state.locations_dict()
        temp_locations.pop(obj)

        return State.from_dict(temp_locations, holding, objectLocation)

# Action to place an object.
def action_place(state: State, obj: str, new_object_location: str) -> State | None:

    if (state.holding != obj): # Precondition -> The gripper must be holding the object.
        return None
    else:                      # Effects -> Object at destined location, arm at that location too and gripper empty.
        new_locations = state.locations_dict()
        new_locations[obj]= new_object_location

        return State.from_dict(new_locations, None, new_object_location)

'''
Now let's get with the brain of the task planner, the BFS algorithm to search for the actions to accomplish the goal.
'''

# Function to check if the goal has been satisfied -> All objects are at the desired location.
def goal_satisfied(state: State, goal: dict) -> bool:
    actual_locations = state.locations_dict()
    for obj, location in goal.items(): # Use .items() to get keys and values.
        if actual_locations.get(obj) != location:
            return False
    
    return True


# Generator that yields all (action_description, successor_state) pairs valid from the current state.
def get_successors(state: State, objects: list, locations: list):

    for loc in locations:
        result = action_move_to(state, loc)
        if result is not None:
            yield ("move_to", loc), result

    for obj in objects:
        for loc in locations:
            result = action_pick(state, obj, loc)
            if result is not None:
                yield ("pick", obj, loc), result

    if state.holding is not None:
        for loc in locations:
            result = action_place(state, state.holding, loc)
            if result is not None:
                yield ("place", state.holding, loc), result

# Now, go with the plan function, where the BFS algorithm will get executed.
def plan(initialState: State, goal: dict, objects: list, locations: list) -> list | None:

    if goal_satisfied(initialState, goal): # If goal already satisfied -> No actions to take.
        return []
    
    # If goal not already satisfied, let's search for the set of actions that will satisfy it.
    queue = deque()
    queue.append((initialState, [])) # Append (state, action taken so far).
    visited = {initialState} # Already visited states.

    while queue:
        state, actions = queue.popleft()
        for action_description, succesor in get_successors(state, objects, locations):
            if not (succesor in visited): # If succesor hasn't been visited (that given action hasn't been tried).
                visited.add(succesor)
                new_actions = actions + [action_description]
                if goal_satisfied(succesor, goal):
                    return new_actions
                else:
                    queue.append((succesor, new_actions))
    
    return None # If no valid set of actions found, return None.


