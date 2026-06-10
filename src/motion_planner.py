from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

class SafetyCheckFailed(Exception):
    pass

class ObjectDetected(Exception):
    def __init__(self, world_position):
        self.world_position = world_position

client = RemoteAPIClient()
sim = client.require('sim')
ompl = client.require('simOMPL')
ik = client.require('simIK')

# OMPL algorithm. RRTConnect validates collisions throughout the path —
# safer than the lazy variants, which can route the arm through walls.
OMPL_ALGORITHM_NAME = "KPIECE1"


def _ompl_algorithm():
    """Resolve the configured algorithm name to an Algorithm enum value."""
    return getattr(ompl.Algorithm, OMPL_ALGORITHM_NAME, ompl.Algorithm.RRTConnect)

# Robot parts needed throughout the code.
joint_handles = [sim.getObject('/Franka/joint', {'index': i}) for i in range(7)]
tip = sim.getObject("/Franka/link8_resp/connection")
armTarget = sim.getObject("/armTarget")
gripperBase = sim.getObject("/Franka/FrankaGripper")
fingersJoint = sim.getObject("/Franka/FrankaGripper/openCloseJoint")
centerJoint = sim.getObject("/Franka/FrankaGripper/centerJoint")

def moveToAndPickOrReleaseObject(pickObject, cube, aboveApproximationDistance, heightThreshold, releaseDestination=None, safety_check=None):

    # Get the target pose depending on pick or release.
    if not releaseDestination:
        targetPose = sim.getObjectPose(cube, sim.handle_world)
    else:
        targetPose = sim.getObjectPose(releaseDestination, sim.handle_world)

    targetPose[2] += aboveApproximationDistance + heightThreshold
    targetPose[3:] = [0.0, 1.0, 0.0, 0.0] # Tip pointing downwards.

    if pickObject:
        # PICK flow:
        # Disable collidable before any movement toward the cube — arm geometry overlaps
        # with the cube at the goal state, so OMPL would reject it as invalid.

        moveArmDynamicallyToPose(targetPose, safety_check=safety_check)  # Move above cube.
        targetPose[2] -= aboveApproximationDistance             # Calculate descending movement to pick it.

        prop = sim.getObjectSpecialProperty(cube)
        sim.setObjectSpecialProperty(cube, prop & ~sim.objectspecialproperty_collidable) # Deactivate cube collision before going down so we can make a path towards it.
        moveArmToFixedPose(targetPose)
        success = pick(cube)                                    # Attach + re-enable collidable.
        targetPose[2] += aboveApproximationDistance
        moveArmToFixedPose(targetPose)                    # Ascend (cube collidable, included in OMPL).
        return success

    else:
        # RELEASE flow:
        # Cube stays collidable during transit to destination — OMPL accounts for its volume.
        moveArmDynamicallyToPose(targetPose)                    # Move above destination (cube collidable).
        targetPose[2] -= aboveApproximationDistance

        # Now disable collidable before the final descent — cube touching the surface
        # would make the goal state invalid for OMPL.
        prop = sim.getObjectSpecialProperty(cube)
        sim.setObjectSpecialProperty(cube, prop & ~sim.objectspecialproperty_collidable)
        moveArmToFixedPose(targetPose)                    # Descend to release height.
        release(cube)                                           # Detach cube.
        targetPose[2] += aboveApproximationDistance
        moveArmToFixedPose(targetPose)                    # Ascend.

# Auxiliar function to move arm to a determinate pose.
def moveArmToFixedPose(targetPose):
    sim.moveToPose({
        "joints": joint_handles,
        "ik": {"tip": tip, "target": armTarget},
        "targetPose": targetPose,
        "maxVel": [1.5, 1.5, 1.5, 1.5],
        "maxAccel": [0.8, 0.8, 0.8, 0.8],
        "maxJerk": [0.4, 0.4, 0.4, 0.4]
    })

def moveArmDynamicallyToPose(targetPose, max_attempts=6, safety_check=None):
    '''
    Function that given an object handle it moves the robotic arm to that position -> Dynamically traces a path and avoids obstacles to reach the goal destination.
    The arm reaches the object at around 20cm over it and with the tip pointing downwards -> Good configuration to pick the object (or release it).
    Parameters:
     - target: handle for the target object.
     - max_attempts: number of times to retry OMPL if it fails (default 3).
    '''

    # Compute goal joint configuration using simIK.
    sim.setObjectPose(armTarget, targetPose, sim.handle_world) # Place armTarget dummy at desired location.

    ikEnv = ik.createEnvironment()
    ikGroup = ik.createGroup(ikEnv)
    ik.setGroupCalculation(ikEnv, ikGroup, ik.method_damped_least_squares, 0.1, 99) # Configure solver: DLS method, damping=0.1, max 99 iterations.
    ikElement, simToIkMap, ikToSimMap = ik.addElementFromScene(ikEnv, ikGroup, sim.getObject("/Franka"), tip, armTarget, ik.constraint_pose)
    ik.syncFromSim(ikEnv, [ikGroup]) # Sync IK environment with current simulation state.

    ik_joint_handles = [simToIkMap[j] for j in joint_handles]
    goal_config = ik.findConfig(ikEnv, ikGroup, ik_joint_handles, 0.65, 3.0, [1, 1, 1, 0.1])
    ik.eraseEnvironment(ikEnv)

    if not goal_config:
        raise RuntimeError("IK failed — no goal config found.")

    # Retry loop for OMPL planning (probabilistic, may fail on first attempt).
    for attempt in range(1, max_attempts + 1):
        try:
            # First, we have to create a task.
            task = ompl.createTask("move_object")
            ompl.setAlgorithm(task, _ompl_algorithm())

            # Now we define the state space.
            # We need an state space for each joint -> This defines the search space to complete the task.

            state_spaces = [] # Total state space.
            for i, j in enumerate(joint_handles):
                cyclic, interval = sim.getJointInterval(j)
                ss = ompl.createStateSpace(               # per-joint state space
                    f'joint{i}',                          # name
                    ompl.StateSpaceType.joint_position,   # type
                    j,                                    # joint handle
                    [interval[0]],                        # min
                    [interval[0] + interval[1]],          # max
                    1                                     # weight
                )
                state_spaces.append(ss)

            ompl.setStateSpace(task, state_spaces)

            # Now we need to define the collections for obstacle avoidance.
            # We must get the obstacles and the robot collections -> This will tell the algorithm the 2 things than can't collide.

            # Robot collection.
            robot_colletion = sim.createCollection(0)
            sim.addItemToCollection(robot_colletion, sim.handle_tree, sim.getObject("/Franka"), 0) # 0 means include in collection -> handle_tree to add the whole tree of the object.

            # Obstacles collection -> Everything but the robot.
            obstacles_colletion = sim.createCollection(0)
            sim.addItemToCollection(obstacles_colletion, sim.handle_all, -1, 0) # handle_all selects all objects, -1 to get every object, 0 to add it to the collection.
            sim.addItemToCollection(obstacles_colletion, sim.handle_tree, sim.getObject("/Franka"), 1) # 1 to exclude the robotic arm from the collection.

            ompl.setCollisionPairs(task, [robot_colletion, obstacles_colletion]) # Finally, add both as the collision pairs to the task.

            # Now, let's get the inital and final poses for the joints.
            # The first part, the initial pose -> Easy, we are already there.
            start_config = [sim.getJointPosition(i) for i in joint_handles]
            ompl.setStartState(task, start_config)

            # Set the goal state.
            ompl.setGoalState(task, goal_config)

            # Now, let's use the task to compute the path from origin to destiny.
            ompl.setup(task) # We have to set up the task. This resembles finishing it, ready for execution.

            # Validate start and goal BEFORE compute. Feeding OMPL an invalid goal
            # (e.g. arm geometry overlapping with an obstacle at the goal config)
            # is the main cause of C++-side crashes that take the simulator down.
            try:
                start_valid = ompl.isStateValid(task, start_config)
                goal_valid  = ompl.isStateValid(task, goal_config)
            except Exception:
                # If isStateValid is not available in this simOMPL version, skip the check.
                start_valid, goal_valid = True, True
            if not start_valid:
                raise RuntimeError("Start config is in collision — cannot plan.")
            if not goal_valid:
                raise RuntimeError("Goal config is in collision — cannot plan.")

            ompl.compute(task, 5.0, -1, 1) # 1 to allow path simplification -> Avoids having to follow excessive waypoints.

            # After we get the path, we need to get all of the joint angles
            # for every single step in the path and follow them.
            path = ompl.getPath(task)
            num_joints = 7
            num_waypoints = len(path) // num_joints
            print(f"Number of waypoints: {num_waypoints}")

            # Now, let's get the robot to go through all of them.
            for i in range(num_waypoints):
                if safety_check is not None and not safety_check():
                    ompl.destroyTask(task)
                    sim.destroyCollection(robot_colletion)
                    sim.destroyCollection(obstacles_colletion)
                    raise SafetyCheckFailed("Safety check failed mid-path — aborting motion and replanning.")
                config = path[i*num_joints : (i+1)*num_joints] # Get the configurations for the actual path.
                sim.moveToConfig({
                    "joints": joint_handles,
                    "targetPos": config,
                    'maxVel': [1.5] * 7,
                    'maxAccel': [0.8] * 7,
                    'maxJerk': [0.4] * 7,
                })

            # Finally, destroy task and collections to avoid resource leaks.
            ompl.destroyTask(task)
            sim.destroyCollection(robot_colletion)
            sim.destroyCollection(obstacles_colletion)
            return  # Success — exit function.

        except SafetyCheckFailed:
            raise  # Propagate immediately — do not retry OMPL.

        except ObjectDetected:
            raise  # Object found mid-sweep — propagate to scan loop.

        except Exception as e:
            # Clean up on failure.
            try:
                ompl.destroyTask(task)
                sim.destroyCollection(robot_colletion)
                sim.destroyCollection(obstacles_colletion)
            except:
                pass  # Ignore cleanup errors.

            print(f"[OMPL] Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise RuntimeError(f"OMPL planning failed after {max_attempts} attempts.") from e
            # Otherwise, loop to next attempt.


def pick(target, proximity_threshold=0.15):
    '''
    Closes the gripper, attaches target to the gripper, re-enables its collidable flag,
    then lifts ~5cm to clear the pickup surface so transit OMPL planning includes the cube geometry.
    Assumes the arm is already positioned at the grasp pose (placed there by moveArmDynamicallyToPose).
    The caller must disable the target's collidable flag before calling moveArmDynamicallyToPose to approach.
    Returns True on success, False if the cube is too far from the gripper to attach.
    '''
    import math
    tip_pos  = sim.getObjectPosition(tip, sim.handle_world)
    cube_pos = sim.getObjectPosition(target, sim.handle_world)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(tip_pos, cube_pos)))
    if dist > proximity_threshold:
        print(f"  [pick] Cube too far from gripper ({dist:.3f}m > {proximity_threshold}m) — skipping attach.")
        return False

    sim.setJointTargetForce(fingersJoint, 10)
    sim.setJointTargetForce(centerJoint, 10)
    sim.setJointTargetPosition(fingersJoint, 0.0)
    sim.setJointTargetPosition(centerJoint, 0.0)
    sim.setObjectParent(target, gripperBase, True)
    time.sleep(0.5)
    sim.setObjectInt32Param(target, sim.shapeintparam_static, 1) # Freeze physics so object moves with gripper.

    # Re-enable collidable now that the cube is attached — OMPL will include it in collision geometry during transit.
    prop = sim.getObjectSpecialProperty(target)
    sim.setObjectSpecialProperty(target, prop | sim.objectspecialproperty_collidable)
    return True


def release(target):
    '''
    Opens the gripper and detaches the object.
    Assumes the arm is already positioned at the release pose (placed there by moveArmDynamicallyToPose).
    '''
    sim.setJointTargetForce(fingersJoint, 10)
    sim.setJointTargetForce(centerJoint, 10)
    sim.setJointTargetPosition(fingersJoint, 0.08)
    sim.setJointTargetPosition(centerJoint, 0.04)
    sim.setObjectInt32Param(target, sim.shapeintparam_static, 0) # Restore physics.
    sim.setObjectParent(target, -1, True) # Detach from gripper.
    time.sleep(0.5)


initial_config = None


def returnToInitialPose(max_attempts=6):
    '''
    Plans and executes a collision-free path back to the joint configuration captured at startup.
    Uses the same OMPL/RRTConnect approach as moveArmDynamicallyToPose.
    '''
    global initial_config
    if initial_config is None:
        raise RuntimeError("Initial configuration not saved.")

    for attempt in range(1, max_attempts + 1):
        try:
            task = ompl.createTask("return_home")
            ompl.setAlgorithm(task, _ompl_algorithm())

            state_spaces = []
            for i, j in enumerate(joint_handles):
                cyclic, interval = sim.getJointInterval(j)
                ss = ompl.createStateSpace(
                    f'joint{i}',
                    ompl.StateSpaceType.joint_position,
                    j,
                    [interval[0]],
                    [interval[0] + interval[1]],
                    1
                )
                state_spaces.append(ss)
            ompl.setStateSpace(task, state_spaces)

            robot_collection = sim.createCollection(0)
            sim.addItemToCollection(robot_collection, sim.handle_tree, sim.getObject("/Franka"), 0)
            obstacles_collection = sim.createCollection(0)
            sim.addItemToCollection(obstacles_collection, sim.handle_all, -1, 0)
            sim.addItemToCollection(obstacles_collection, sim.handle_tree, sim.getObject("/Franka"), 1)
            ompl.setCollisionPairs(task, [robot_collection, obstacles_collection])

            start_config = [sim.getJointPosition(j) for j in joint_handles]
            ompl.setStartState(task, start_config)
            ompl.setGoalState(task, initial_config)
            ompl.setup(task)

            try:
                start_valid = ompl.isStateValid(task, start_config)
                goal_valid  = ompl.isStateValid(task, initial_config)
            except Exception:
                start_valid, goal_valid = True, True
            if not start_valid:
                raise RuntimeError("Start config is in collision — cannot plan return-home.")
            if not goal_valid:
                raise RuntimeError("Initial config is in collision — cannot plan return-home.")

            ompl.compute(task, 5.0, -1, 1)

            path = ompl.getPath(task)
            num_joints = 7
            num_waypoints = len(path) // num_joints
            print(f"[Home] Waypoints: {num_waypoints}")

            for i in range(num_waypoints):
                config = path[i*num_joints : (i+1)*num_joints]
                sim.moveToConfig({
                    'joints': joint_handles,
                    'targetPos': config,
                    'maxVel': [1.5] * 7,
                    'maxAccel': [0.8] * 7,
                    'maxJerk': [0.4] * 7,
                })

            ompl.destroyTask(task)
            sim.destroyCollection(robot_collection)
            sim.destroyCollection(obstacles_collection)
            return

        except Exception as e:
            try:
                ompl.destroyTask(task)
                sim.destroyCollection(robot_collection)
                sim.destroyCollection(obstacles_collection)
            except:
                pass
            print(f"[OMPL/Home] Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise RuntimeError(f"Return to initial pose failed after {max_attempts} attempts.") from e


def __main__():
    global initial_config
    cube = sim.getObject("/CubeToMove")
    releaseDestination = sim.getObject("/cubeTargetPosition")

    initial_config = [sim.getJointPosition(j) for j in joint_handles]  # Save starting pose.

    # Now, move arm to destination and pick object.
    aboveApproximationThreshold = 0.1
    heightThreshold = 0.08
    moveToAndPickOrReleaseObject(True, cube, aboveApproximationThreshold, heightThreshold) # Go and pick object.
    moveToAndPickOrReleaseObject(False, cube, aboveApproximationThreshold, heightThreshold+0.04, releaseDestination) # Go to destination and release object.
    returnToInitialPose() # Go to back to initial position.


if __name__ == "__main__": __main__()
