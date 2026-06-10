from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.require("sim")

joint_handles = [sim.getObject("/Franka/joint", {"index": i}) for i in range(7)]
tip = sim.getObject('/Franka/link8_resp/connection')
armTarget = sim.getObject("/armTarget")  # dummy used as IK target

# Function to move the robot arm to a world pose.
def moveArmToDestination(worldPose):
    sim.setObjectPose(armTarget, worldPose, sim.handle_world)
    sim.moveToPose({
        "joints": joint_handles,
        "ik": {"tip": tip, "target": armTarget},
        "targetPose": worldPose,
        "maxVel": [0.4, 0.4, 0.4, 0.4],
        "maxAccel": [0.2, 0.2, 0.2, 0.2],
        "maxJerk": [0.1, 0.1, 0.1, 0.1]
    })

def getObjectSize(objectHandle):
    size, _ = sim.getShapeBB(objectHandle)  # size = [sizeX, sizeY, sizeZ]
    return size[0], size[1], size[2]  # width, depth, height

# In this code we will move a cube from where it is at to a target position.

maxGripJointValue = 0.08
minGripJointValue = 0.0

# Initially, we get the starting position of the robotic am so that we can 
# go back to it after moving the object.
initialArmPose = sim.getObjectPose(tip, sim.handle_world)

# First, we get the target, cube and gripper handles.
cube = sim.getObject("/CubeToMove")
cubePosition = sim.getObjectPose(cube, sim.handle_world) # Get cube pose.
target = sim.getObject("/cubeTargetPosition")
targetPosition = sim.getObjectPose(target, sim.handle_world) # Get position target position.
gripperBase = sim.getObject("/Franka/FrankaGripper")
fingersJoint = sim.getObject("/Franka/FrankaGripper/openCloseJoint")
centerJoint = sim.getObject("/Franka/FrankaGripper/centerJoint")

# Set joint max forces.
sim.setJointTargetForce(fingersJoint, 10) # Both forces initially set to 20, we reduce them to 5.
sim.setJointTargetForce(centerJoint, 10)

# First, open gripper.
sim.setJointPosition(fingersJoint, maxGripJointValue)
sim.setJointPosition(centerJoint, maxGripJointValue/2)
sim.setJointTargetPosition(fingersJoint, minGripJointValue)
sim.setJointTargetPosition(centerJoint, minGripJointValue/2)

time.sleep(1.0)  # Wait for gripper to fully close.
gripper_val = sim.getJointPosition(fingersJoint)
print(f"Gripper joint value when fully closed (no object): {gripper_val}")

# Move arm above the cube first.
aboveCube = cubePosition[:]
cubeWidth, cubeDepth, cubeHeight = getObjectSize(cube)
aboveCube[2] += cubeHeight + 0.2  # 20cm above the cube
aboveCube[3:] = [0.0, 1.0, 0.0, 0.0]
moveArmToDestination(aboveCube)

# Now, move arm downwards to effectively be in a position to pick up the cube.
aboveCube[2] -= 0.18
moveArmToDestination(aboveCube)

# Now, close gripper.
time.sleep(0.5) # Wait a little bit of time before closing the gripper (smooth behaviour).
sim.setJointTargetPosition(fingersJoint, minGripJointValue)
sim.setJointTargetPosition(centerJoint, minGripJointValue/2)

# Since we are having a problem, which is that the cube slips from the fingers in the physics engine, 
# in order to move it around we will make the cube a child of the gripper, so that it will move with it.
# This will allow us to effectively move the cube in the physics engine, without cheating with respecto to what would 
# happen in the real world, since in the real world it would get effectively picked up by the gripper.
sim.setObjectParent(cube, gripperBase, True)
sim.setObjectInt32Param(cube, sim.shapeintparam_static, 1) # Freeze physics on cube too so it will now fall.

time.sleep(0.5) # Wait some time so the gripper can effectively pick the object (smooth behaviour).

# Move the arm upwards to smoothly extract the object.
aboveCube[2] += 0.18
moveArmToDestination(aboveCube)

# Move the arm above the final destination.
targetPosition[2] += aboveCube[2] # Use the same height as the one used to puick the object. This ensures correct deployment regardless of the object's height.
targetPosition[3:] = [0.0, 1.0, 0.0, 0.0]
moveArmToDestination(targetPosition)

# Move it downwards to gently place the object.
targetPosition[2] -= 0.16
moveArmToDestination(targetPosition)

# Now, open the gripper to release the object.
time.sleep(0.5) # Wait a little bit before releasing the object (smooth behaviour).
sim.setJointTargetPosition(fingersJoint, maxGripJointValue)
sim.setJointTargetPosition(centerJoint, maxGripJointValue/2)
sim.setObjectInt32Param(cube, sim.shapeintparam_static, 0) # Enable cube physics.
sim.setObjectParent(cube, -1, True) # Release the cube from the gripper.
time.sleep(0.5) # Wait a little bit before lifting the arm (smooth behaviour).

# Move the arm upwards to smoothly retrieve it from the deployment destination.
targetPosition[2] += 0.4
moveArmToDestination(targetPosition)

# Finally, go to the initial arm position.
moveArmToDestination(initialArmPose)