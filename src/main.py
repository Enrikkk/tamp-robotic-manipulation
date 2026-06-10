from task_planner import State, plan
import motion_planner as mp
from motion_planner import SafetyCheckFailed

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
OBJECTS   = ["red_cube"]
LOCATIONS = ["table_pos_1", "bin_A", "unknown"]

OBJECT_HANDLES_PATHS = {
    "red_cube": "/CubeToMove",
}
LOCATION_HANDLES_PATHS = {
    "table_pos_1": "/CubeToMove",          # cube's own position = pick reference
    "bin_A":       "/cubeTargetPosition",  # destination dummy
}

# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------
def execute_action(action: tuple, object_handles: dict, location_handles: dict) -> None:
    verb = action[0]

    if verb == "move_to":
        print(f"  [motion] move_to({action[1]}) — implicit, skipping.")

    elif verb == "pick":
        _, obj, loc = action
        print(f"  [motion] pick({obj}) from {loc}")
        cube_handle = object_handles[obj]
        expected_pos = mp.sim.getObjectPosition(cube_handle, mp.sim.handle_world)

        def target_still_at_expected_position():
            current_pos = mp.sim.getObjectPosition(cube_handle, mp.sim.handle_world)
            return _distance(expected_pos, current_pos) < 0.1

        return mp.moveToAndPickOrReleaseObject(
            pickObject=True,
            cube=cube_handle,
            aboveApproximationDistance=0.1,
            heightThreshold=0.08,
            safety_check=target_still_at_expected_position,
        )

    elif verb == "place":
        _, obj, loc = action
        print(f"  [motion] place({obj}) at {loc}")
        mp.moveToAndPickOrReleaseObject(
            pickObject=False,
            cube=object_handles[obj],
            aboveApproximationDistance=0.1,
            heightThreshold=0.12,
            releaseDestination=location_handles[loc],
        )

# Auxiliary functions.
def verify_pick(threshold=0.005) -> bool:
    return mp.sim.getJointPosition(mp.fingersJoint) > threshold

def verify_place(cube_handle, destination_handle, tolerance=0.1) -> bool:
    import math
    cube_pos = mp.sim.getObjectPosition(cube_handle, mp.sim.handle_world)
    dest_pos = mp.sim.getObjectPosition(destination_handle, mp.sim.handle_world)
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(cube_pos, dest_pos)))
    return distance < tolerance

def _distance(pos1, pos2):
    import math
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

def sense_world(object_handles: dict, location_handles: dict) -> State:
    '''
    Reads the current world state from the simulator: where each object is,
    what the gripper is holding, and where the arm is. Each object is mapped to
    its nearest known location.
    '''
    sim = mp.sim

    # Build the state object
    obj_locs = {}
    holding = None

    for obj_name, obj_handle in object_handles.items():
        if sim.getObjectParent(obj_handle) == mp.gripperBase:
            holding = obj_name
            continue
        obj_pos = sim.getObjectPosition(obj_handle, sim.handle_world)
        closest = min(
            location_handles.items(),
            key=lambda kv: _distance(obj_pos, sim.getObjectPosition(kv[1], sim.handle_world))
        )
        obj_locs[obj_name] = closest[0]

    # Where is the arm right now?
    tip_pos = sim.getObjectPosition(mp.tip, sim.handle_world)
    closest_name, closest_handle = min(
        location_handles.items(),
        key=lambda kv: _distance(tip_pos, sim.getObjectPosition(kv[1], sim.handle_world))
    )
    closest_dist = _distance(tip_pos, sim.getObjectPosition(closest_handle, sim.handle_world))
    arm_at = closest_name if closest_dist < 0.3 else "unknown"

    return State.from_dict(obj_locs, holding, arm_at)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sim = mp.sim

    mp.initial_config = [sim.getJointPosition(j) for j in mp.joint_handles]

    object_handles = {}
    for name, path in OBJECT_HANDLES_PATHS.items():
        try:
            object_handles[name] = sim.getObject(path)
        except Exception:
            print(f"ERROR: Object '{name}' not found in scene (expected at '{path}'). Aborting.")
            return

    location_handles = {}
    for name, path in LOCATION_HANDLES_PATHS.items():
        try:
            location_handles[name] = sim.getObject(path)
        except Exception:
            print(f"ERROR: Location '{name}' not found in scene (expected at '{path}'). Aborting.")
            return

    goal = {"red_cube": "bin_A"}
    max_replans = 10

    print("=== TAMP Execution Loop ===")
    print(f"Goal: {goal}")

    for attempt in range(1, max_replans + 1):
        print(f"\n--- Plan attempt {attempt}/{max_replans} ---")
        current_state = sense_world(object_handles, location_handles)
        action_sequence = plan(current_state, goal, OBJECTS, LOCATIONS)

        if action_sequence is None:
            print("ERROR: No plan found from current state. Aborting.")
            break

        print(f"Plan ({len(action_sequence)} steps):")
        for i, a in enumerate(action_sequence, 1):
            print(f"  {i}. {a[0]}({', '.join(a[1:])})")

        print("\n=== Motion Execution ===")
        replan_needed = False
        for action in action_sequence:
            try:
                result = execute_action(action, object_handles, location_handles)
            except SafetyCheckFailed as e:
                print(f"  [safety] Motion aborted mid-path: {e}")
                replan_needed = True
                break

            if action[0] == "pick":
                if result is False or not verify_pick():
                    print("  [verify] Pick failed — replanning.")
                    replan_needed = True
                    break
                print("  [verify] Pick succeeded.")

            elif action[0] == "place":
                obj = action[1]
                loc = action[2]
                if not verify_place(object_handles[obj], location_handles[loc]):
                    print("  [verify] Place failed — replanning.")
                    replan_needed = True
                    break
                print("  [verify] Place succeeded.")

        if not replan_needed:
            print("\nGoal achieved!")
            break
    else:
        print("\nFailed to achieve goal after maximum replan attempts.")

    print("\n=== Returning to initial pose ===")
    mp.returnToInitialPose()
    print("Done.")


if __name__ == "__main__":
    main()
