from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.require("sim")

# We will move the FrankaGripper's fingers -> Used to pick objects.

# It uses a single joint for both finders (easier).
# However, we have to use the middle joint to balance that.
fingersJoint = sim.getObject("/Franka/FrankaGripper/openCloseJoint") # Get the joint's handle.
centerJoint = sim.getObject("/Franka/FrankaGripper/centerJoint")
actualPosition = sim.getJointPosition(fingersJoint) # Get the actual joint position.
print(f"Actual Gripper Fingers Position: {actualPosition}")

# Force position directly, bypassing physics
targetJointPosition = 0.0
sim.setJointPosition(fingersJoint, targetJointPosition)
sim.setJointPosition(centerJoint, targetJointPosition/2)
time.sleep(0.5)
actualPosition = sim.getJointPosition(fingersJoint)
print(f"New Gripper Fingers Position: {actualPosition}")