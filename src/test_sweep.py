from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import motion_planner as mp
import perception

sim = mp.sim

# Arm base position and initial tip height.
arm_base = sim.getObjectPosition(sim.getObject('/Franka'), sim.handle_world)
tip_z    = sim.getObjectPosition(mp.tip, sim.handle_world)[2]
print(f"Arm base: {arm_base}")
print(f"Initial tip Z: {tip_z:.3f}m")

# Sweep parameters.
radius     = 0.5                      # Horizontal orbit radius (meters).
tilts      = [0.0, 0.3, 0.6]          # Three rings: down → slightly out → wider out.
n_azimuths = 6                        # 4 directions per ring (every 90°) for quick test.

def dummy_detection():
    return True  # Never fires — pure motion test.

print("\nStarting sweep test...")
result = perception.scan_for_object(arm_base, tip_z, radius, tilts, n_azimuths, dummy_detection)
print(f"\nSweep complete. Result: {result}")
