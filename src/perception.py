import math
import numpy as np
import cv2
import motion_planner as mp
from motion_planner import ObjectDetected

MAX_TILT_DEG = 70  # Hard limit — camera never tilts more than this from vertical.


def _tilt_quaternion(azimuth, tilt):
    '''
    Computes the tip quaternion for a given azimuth direction and outward tilt angle.
    Tilt=0 means straight down [0,1,0,0]. Tilt>0 rotates the camera outward (away
    from the arm base) around the tangent axis at that azimuth.

    The tilt axis is tangent to the sweep circle: [-sin(az), cos(az), 0].
    Quaternion = q_tilt * q_down, where q_down = [0,1,0,0] (CoppeliaSim: qx,qy,qz,qw).
    '''
    s = -math.sin(tilt / 2)  # Negative → rotates outward (away from arm base).
    c =  math.cos(tilt / 2)
    # Tilt axis components: ax = -sin(az), ay = cos(az), az_comp = 0
    ax = -math.sin(azimuth)
    ay =  math.cos(azimuth)
    # q_tilt * q_down (derived analytically — see comments in perception.py)
    qx = 0.0
    qy = c
    qz = ax * s          # -sin(az) * sin(tilt/2)
    qw = -ay * s         # -cos(az) * sin(tilt/2)
    return [qx, qy, qz, qw]


def generate_viewpoints(arm_base, tip_z, radius, tilts, n_azimuths, start_angle_deg=90):
    '''
    Generates camera poses at a fixed height (tip_z), sweeping azimuth in rings.
    Each ring uses a different outward tilt angle — ring 1 looks straight down,
    later rings tilt the camera progressively outward for a wider field of view.

    Parameters:
     - arm_base: [x, y, z] world position of the robot arm base
     - tip_z: fixed Z height for all viewpoints (use initial tip Z at startup)
     - radius: horizontal distance from arm_base to each viewpoint (meters)
     - tilts: list of tilt angles in radians, e.g. [0.0, 0.3, 0.6] (0 = down)
     - n_azimuths: number of azimuth steps per ring (e.g. 8 → every 45°)

    Returns:
     - list of poses [x, y, z, qx, qy, qz, qw]
    '''
    poses = []
    bx, by, _ = arm_base

    start_offset = math.radians(start_angle_deg)

    for tilt in tilts:
        clamped_tilt = min(tilt, math.radians(MAX_TILT_DEG))
        for i in range(n_azimuths):
            azimuth = start_offset + i * (2 * math.pi / n_azimuths)
            x = bx + radius * math.cos(azimuth)
            y = by + radius * math.sin(azimuth)
            z = tip_z  # Fixed height throughout.
            qx, qy, qz, qw = _tilt_quaternion(azimuth, clamped_tilt)
            poses.append([x, y, z, qx, qy, qz, qw])

    return poses


def scan_for_object(arm_base, tip_z, radius, tilts, n_azimuths, detection_callback, start_angle_deg=90, nudge_deg=10, max_nudges=3):
    '''
    Sweeps the arm around arm_base at a fixed height. At each azimuth position the arm
    arrives looking straight down, then performs a fast tilt sweep in place (no OMPL)
    before moving to the next azimuth. Detection is checked after each tilt stop and
    during transit between azimuth positions.

    Parameters:
     - arm_base: [x, y, z] world position of the robot arm base
     - tip_z: fixed Z height for all viewpoints (use initial tip Z)
     - radius: horizontal orbit radius in meters
     - tilts: list of outward tilt angles in radians, e.g. [0.0, 0.3, 0.6] (0 = straight down)
     - n_azimuths: number of azimuth positions per sweep
     - detection_callback: callable — raises ObjectDetected if object found, else returns True
     - start_angle_deg: starting azimuth offset in degrees (default 90°)
     - nudge_deg: degrees to offset azimuth when OMPL fails
     - max_nudges: how many nudges to try before skipping a viewpoint

    Returns:
     - world_position [x, y, z] if object found
     - None if full sweep completed without finding anything
    '''
    bx, by, _ = arm_base
    start_offset = math.radians(start_angle_deg)

    for i in range(n_azimuths - 1):  # -1 so we don't revisit the start position.
        base_azimuth = start_offset + i * (2 * math.pi / n_azimuths)

        # --- Transit to azimuth position (OMPL, detection runs per-waypoint) ---
        reached = False
        for azimuth in candidate_azimuths(base_azimuth, nudge_deg, max_nudges):
            x = bx + radius * math.cos(azimuth)
            y = by + radius * math.sin(azimuth)
            z = tip_z
            pose = [x, y, z, 0.0, 1.0, 0.0, 0.0]  # Arrive looking straight down.

            try:
                mp.moveArmDynamicallyToPose(pose, safety_check=detection_callback)
                reached = True
                actual_azimuth = azimuth
                break
            except ObjectDetected:
                raise  # Found during transit — propagate immediately.
            except RuntimeError:
                print(f"  [scan] OMPL failed at az={math.degrees(azimuth):.0f}° — nudging.")

        if not reached:
            print(f"  [scan] Skipping az={math.degrees(base_azimuth):.0f}° — no reachable pose found.")
            continue

        # --- Tilt sweep in place (fast IK, no OMPL) ---
        print(f"  [scan] az={math.degrees(actual_azimuth):.0f}° reached — sweeping tilts.")
        for tilt in tilts:
            clamped_tilt = min(tilt, math.radians(MAX_TILT_DEG))
            x = bx + radius * math.cos(actual_azimuth)
            y = by + radius * math.sin(actual_azimuth)
            qx, qy, qz, qw = _tilt_quaternion(actual_azimuth, clamped_tilt)
            tilt_pose = [x, y, tip_z, qx, qy, qz, qw]

            mp.moveArmToFixedPose(tilt_pose)  # Fast — no OMPL needed.

            # Check detection at this tilt stop.
            try:
                detection_callback()
            except ObjectDetected:
                raise  # Found during tilt sweep — propagate immediately.

    return None  # Full sweep completed, object not found.


def candidate_azimuths(base_azimuth, nudge_deg=10, max_nudges=3):
    '''
    Generator that yields candidate azimuth angles around a base angle.
    Used when OMPL fails — tries small offsets before giving up.
    Order: base → base-nudge → base+nudge → base-2*nudge → base+2*nudge → ...
    '''
    nudge_rad = math.radians(nudge_deg)
    yield base_azimuth
    for i in range(1, max_nudges + 1):
        yield base_azimuth - i * nudge_rad
        yield base_azimuth + i * nudge_rad


# ---------------------------------------------------------------------------
# Vision Utilities
# ---------------------------------------------------------------------------

def capture_image(vision_sensor_handle):
    '''Reads RGB image from simulator, converts to OpenCV BGR format, and flips it correctly.'''
    image_data, resolution = mp.sim.getVisionSensorImg(vision_sensor_handle)
    rgb_data = np.frombuffer(image_data, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
    bgr_data = cv2.flip(bgr_data, 0)
    return bgr_data, resolution[0], resolution[1]

def get_sensor_intrinsics(vision_sensor_handle, W, H):
    '''Computes pinhole camera intrinsic parameters (fx, fy, cx, cy) from simulator FOV.'''
    fov = mp.sim.getObjectFloatParam(vision_sensor_handle, mp.sim.visionfloatparam_perspective_angle)
    fy = (H / 2) / math.tan(fov / 2)
    fx = fy * W / H
    cx = W / 2
    cy = H / 2
    return fx, fy, cx, cy

def quaternion_to_rotation_matrix(q):
    '''Converts CoppeliaSim quaternion [qx,qy,qz,qw] into a 3x3 rotation matrix.'''
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ])

def backproject_to_world(u, v, Z_world, cam_pos, R, fx, fy, cx, cy):
    '''
    Converts a 2D pixel coordinate (u, v) back into a 3D world position [X, Y, Z_world].
    Assumes the object is resting on a flat plane at Z = Z_world.
    '''
    # 1. Undo pinhole projection to get ray direction in camera space (assume depth=1)
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    ray_cam = np.array([x_cam, y_cam, 1.0])

    # 2. Rotate ray from camera space into world space
    ray_world = R @ ray_cam

    # 3. Find how far along the ray we need to travel to intersect the Z_world plane
    t = (Z_world - cam_pos[2]) / ray_world[2]

    # 4. Final 3D world position = camera_position + t * ray_direction
    return cam_pos + t * ray_world
