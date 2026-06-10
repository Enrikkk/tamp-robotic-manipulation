from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

# End-effector tip is the 'connection' object under link8_resp
tip = sim.getObject('/Franka/link8_resp/connection')

pose = sim.getObjectPose(tip, sim.handle_world)
print(f"Position (x,y,z): {pose[:3]}")
print(f"Quaternion (x,y,z,w): {pose[3:]}")