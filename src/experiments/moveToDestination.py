from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

client = RemoteAPIClient()
sim = client.require('sim')

# Move to the position of, in this case, a dummy object.

# First, get the dummy object's position.
dummy_handle = sim.getObject("/dummy")
dummy_pose = sim.getObjectPose(dummy_handle, sim.handle_world)

# Now, we get the object handles.
joint_handles = [sim.getObject("/Franka/joint", {"index": i}) for i in range(7)] # i in range(7) -> Robotic arm with 7 DOF.

# Establish velocity, acceleration and jerk parameters. We know what acceleration and velocity do, jerk controls how big the change in acceleration is.
# moveToPose uses Cartesian space: 4 values (x, y, z, rotation)
maxVelocity     = [0.3, 0.3, 0.3, 0.3]
maxAcceleration = [0.15, 0.15, 0.15, 0.15]
maxJerk         = [0.1, 0.1, 0.1, 0.1]

# Get the tip (end-effector) handle
tip = sim.getObject('/Franka/link8_resp/connection')

sim.moveToPose({
    "joints": joint_handles,
    "ik": {"tip": tip, "target": dummy_handle},
    "targetPose": dummy_pose,
    "maxVel": maxVelocity,
    "maxAccel": maxAcceleration,
    "maxJerk": maxJerk
})

'''
# Code to place the gripper at the tip of the robotic arm.
# Move the gripper to the tip of the robotic arm.
frankaGripper = sim.getObject("/FrankaGripper") # Get it's handle.
connetion = sim.getObject("/Franka/connection") # Get the connection's handle.

connectionPose = sim.getObjectPose(connetion, sim.handle_world) # Get the connection's position.
sim.setObjectPose(frankaGripper, connectionPose, sim.handle_world) # Place gripper at the tip of the robotic arm.
sim.setObjectParent(frankaGripper, connetion, True) # Parent the gripper to the robotic arm. Use True at the end so that it doesn't teleport to the connection's tip, we assume it is already there.
'''
'''
# One way of doing it, setting the angles for every single joint of the robot.

# Get handles to all 7 joints of the Franka Panda
joint_handles = []
for i in range(7):
    handle = sim.getObject('/Franka/joint', {'index': i})
    joint_handles.append(handle)

# Motion profile: controls how fast and smoothly the arm moves
vel   = 90 * math.pi / 180   # max velocity (deg/s -> rad/s)
accel = 40 * math.pi / 180   # max acceleration
jerk  = 80 * math.pi / 180   # max jerk (smoothness)

max_vel   = [vel]   * 7
max_accel = [accel] * 7
max_jerk  = [jerk]  * 7

# Target joint configuration: home position (all angles in radians)
target = [0, 0, 0, -90*math.pi/180, 0, 90*math.pi/180, 0]

print("Moving arm to home position...")
sim.moveToConfig({
    'joints': joint_handles,
    'targetPos': target,
    'maxVel': max_vel,
    'maxAccel': max_accel,
    'maxJerk': max_jerk,
})
print("Done.")
'''