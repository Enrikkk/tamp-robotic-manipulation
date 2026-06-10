"""
Vision-driven pick-and-place (HSV colour segmentation, single-threaded).

User picks figure + colour via arrow-key TUI, then the arm sweeps overhead
observation poses. At every OMPL waypoint boundary the safety_check captures
a frame, runs HSV segmentation, updates a live OpenCV overlay window, and
raises ObjectDetected the moment a matching coloured blob is found. The
centroid is back-projected to world coordinates, the closest matching scene
object is picked, and it's placed at /cubeTargetPosition.

Single-threaded by design: avoids ZMQ socket contention with the main thread.
"""
import os
import sys
import math
import time
import threading
import termios
import tty

import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import motion_planner as mp
from motion_planner import ObjectDetected
import perception


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DESTINATION_PATH   = "/cubeTargetPosition"
TABLE_Z            = 0.2

SWEEP_RADIUS       = 0.5
SWEEP_TILTS        = [0.0, 0.4, 0.8]
N_AZIMUTHS         = 8
OBSERVATION_TIP_Z  = 0.7

DEBUG_FRAMES_DIR   = "vision_debug"
SHOW_CV2_WINDOW    = True
DISPLAY_FPS        = 20    # read-only display thread tick rate

# HSV detection params.
MIN_S              = 80
MIN_V              = 60
MIN_DETECTION_AREA = 250
MORPH_KERNEL       = 5

SELECTABLE_FIGURES = ["cube", "cylinder", "cone"]

COLORS_PER_FIGURE = {
    "cube":     ["red", "cyan", "any"],
    "cone":     ["yellow", "any"],
    "cylinder": ["any"],
}

SCENE_INDEX = {
    ("cube", "red"):     ["/CubeToMove"],
    ("cube", "cyan"):    ["/Cuboid"],
    ("cube", "any"):     ["/CubeToMove", "/Cuboid", "/Box1", "/Box2", "/Box3"],
    ("cone", "yellow"):  ["/Cone[0]"],
    ("cone", "any"):     ["/Cone[0]", "/Cone[1]", "/Cone[2]", "/Cone[3]"],
    ("cylinder", "any"): [],
}

COLOR_HSV = {
    "red":    [(0, 12), (165, 180)],
    "cyan":   [(80, 105)],
    "yellow": [(18, 38)],
    "green":  [(40, 80)],
    "blue":   [(105, 130)],
}

COLOR_BGR = {
    "red":    (0, 0, 255),
    "cyan":   (255, 255, 0),
    "yellow": (0, 255, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "any":    (255, 255, 255),
}


# ---------------------------------------------------------------------------
# Arrow-key TUI prompt (Linux/POSIX)
# ---------------------------------------------------------------------------
def _read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def select_menu(title: str, options: list[str], allow_other: bool = True) -> str:
    items = list(options) + (["Other (type custom)"] if allow_other else [])
    cursor = 0
    print(f"\n{title}")
    print("(↑/↓ to navigate, Enter to select)")
    for _ in items:
        print()

    def render():
        sys.stdout.write(f"\033[{len(items)}A")
        for i, opt in enumerate(items):
            marker = "▶ " if i == cursor else "  "
            sys.stdout.write(f"\033[K{marker}{opt}\n")
        sys.stdout.flush()

    render()
    while True:
        key = _read_key()
        if key == "\x1b[A":
            cursor = (cursor - 1) % len(items)
            render()
        elif key == "\x1b[B":
            cursor = (cursor + 1) % len(items)
            render()
        elif key in ("\r", "\n"):
            if allow_other and cursor == len(items) - 1:
                custom = input("Type custom value: ").strip().lower()
                return custom
            return items[cursor]
        elif key == "\x03":
            raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# HSV colour detection
# ---------------------------------------------------------------------------
def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    if color == "any":
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for c in ("red", "cyan", "yellow"):
            m = cv2.bitwise_or(m, _color_mask(hsv, c))
        return m

    ranges = COLOR_HSV.get(color)
    if ranges is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        part = cv2.inRange(
            hsv,
            np.array([lo, MIN_S, MIN_V], dtype=np.uint8),
            np.array([hi, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, part)
    return mask


def detect_color(bgr: np.ndarray, color: str):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _color_mask(hsv, color)

    if MORPH_KERNEL > 0:
        k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_DETECTION_AREA:
        return None, None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, mask
    cu  = M["m10"] / M["m00"]
    cv_ = M["m01"] / M["m00"]
    return (cu, cv_), largest, mask


def render_overlay(bgr, color, centroid_px, contour, mask, status, banner_bgr,
                   target_color, target_figure):
    H, W = bgr.shape[:2]
    overlay_color = COLOR_BGR.get(color, (255, 255, 255))
    area = int(cv2.contourArea(contour)) if contour is not None else 0

    if mask is not None and mask.any():
        tint = np.zeros_like(bgr)
        tint[:] = overlay_color
        tint = cv2.bitwise_and(tint, tint, mask=mask)
        bgr = cv2.addWeighted(bgr, 1.0, tint, 0.35, 0)

    if contour is not None:
        cv2.drawContours(bgr, [contour], -1, overlay_color, 2)
        if centroid_px is not None:
            cu, cv_ = int(centroid_px[0]), int(centroid_px[1])
            cv2.circle(bgr, (cu, cv_), 6, (0, 0, 0), -1)
            cv2.circle(bgr, (cu, cv_), 4, overlay_color, -1)
            label = f"{target_color} {target_figure} ({area}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(bgr, (cu + 8, cv_ - th - 6),
                          (cu + 8 + tw + 4, cv_), overlay_color, -1)
            cv2.putText(bgr, label, (cu + 10, cv_ - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(bgr, (0, 0), (W, 26), banner_bgr, -1)
    cv2.putText(bgr, status, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if mask is not None:
        # Big inset so the colour-filter output is actually visible.
        inset_h = max(120, H // 3)
        inset_w = max(120, W // 3)
        thumb = cv2.resize(mask, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        # Tint white pixels in the mask with the target colour so it pops.
        if thumb.any():
            tint = np.zeros_like(thumb_bgr)
            tint[:] = overlay_color
            thumb_bgr = cv2.bitwise_and(tint, tint, mask=thumb)
        x0 = W - inset_w - 6
        y0 = 32
        bgr[y0:y0 + inset_h, x0:x0 + inset_w] = thumb_bgr
        cv2.rectangle(bgr, (x0 - 1, y0 - 1),
                      (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)
        header = f"colour filter: {target_color}"
        cv2.rectangle(bgr, (x0, y0), (x0 + inset_w, y0 + 18), (0, 0, 0), -1)
        cv2.putText(bgr, header, (x0 + 4, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if not thumb.any():
            text = "no pixels match"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = x0 + (inset_w - tw) // 2
            ty = y0 + inset_h // 2 + th // 2
            cv2.putText(bgr, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return bgr


# ---------------------------------------------------------------------------
# Detection state — minimal: just whether we already raised once
# ---------------------------------------------------------------------------
class DetectionState:
    def __init__(self, figure: str, color: str):
        self.target_figure = figure
        self.target_color = color
        self.found = False
        self.world_pos = None
        self.search_mode = True
        self.frames_seen = 0
        self.last_log_t = 0.0
        self.stop_event = threading.Event()


def display_thread_fn(state: DetectionState, sensor_path: str):
    """
    Read-only frame-display loop. Owns its own ZMQ client. Never writes back
    to the simulator (no setVisionSensorImg, no handleVisionSensor) so it
    cannot starve or contend with the main thread's OMPL traffic. Detection
    happens here for the overlay only — the authoritative detection (the one
    that aborts motion) still runs in safety_check on the main thread.
    """
    try:
        client = RemoteAPIClient()
        sim_d = client.require("sim")
        sensor_handle = sim_d.getObject(sensor_path)
    except Exception as e:
        print(f"  [display] could not connect: {e}")
        return

    period = 1.0 / max(1, DISPLAY_FPS)
    consec_errors = 0

    while not state.stop_event.is_set():
        t0 = time.time()
        try:
            image_data, resolution = sim_d.getVisionSensorImg(sensor_handle)
            W, H = resolution[0], resolution[1]
            rgb = np.frombuffer(image_data, dtype=np.uint8).reshape(H, W, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.flip(bgr, 0)

            centroid_px, contour, mask = detect_color(bgr, state.target_color)
            if state.found:
                status = f"FOUND  {state.target_color} {state.target_figure}"
                banner = (0, 200, 0)
            elif centroid_px is not None:
                area = int(cv2.contourArea(contour))
                status = f"DETECTED  {state.target_color} {state.target_figure}  ({area}px)"
                banner = (0, 200, 0)
            else:
                status = f"SEARCHING  {state.target_color} {state.target_figure}"
                banner = (40, 40, 200)

            annotated = render_overlay(bgr, state.target_color,
                                       centroid_px, contour, mask,
                                       status, banner,
                                       state.target_color, state.target_figure)
            if SHOW_CV2_WINDOW:
                cv2.imshow("HSV Live", annotated)
                cv2.waitKey(1)
            consec_errors = 0
        except Exception as e:
            consec_errors += 1
            if consec_errors == 1 or consec_errors % 50 == 0:
                print(f"  [display] read error ({consec_errors}): {e}")

        elapsed = time.time() - t0
        state.stop_event.wait(timeout=max(0.0, period - elapsed))


def make_safety_check(state: DetectionState, sensor_handle):
    """
    Returns a callable that motion_planner invokes between waypoints.
    The callable: captures a frame, runs HSV, updates the live overlay,
    and raises ObjectDetected if a matching colour blob is found.
    """
    def cb():
        try:
            bgr, W, H = perception.capture_image(sensor_handle)
        except Exception as e:
            print(f"  [vision] capture failed: {e}")
            return True

        if state.frames_seen < 3:
            os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
            cv2.imwrite(f"{DEBUG_FRAMES_DIR}/frame_{state.frames_seen:02d}.png", bgr)
        state.frames_seen += 1

        centroid_px, contour, mask = detect_color(bgr, state.target_color)
        # No cv2.imshow here — the display thread owns the window. This callback
        # is only the authoritative "should we abort and pick?" path.

        now = time.time()
        if now - state.last_log_t > 0.5:
            state.last_log_t = now
            has = "YES" if centroid_px is not None else "no"
            print(f"  [vision] frame {state.frames_seen}: detection={has}")

        if state.search_mode and centroid_px is not None and not state.found:
            fx, fy, cx, cy = perception.get_sensor_intrinsics(sensor_handle, W, H)
            cam_pose = mp.sim.getObjectPose(sensor_handle, mp.sim.handle_world)
            cam_pos  = np.array(cam_pose[:3])
            R        = perception.quaternion_to_rotation_matrix(cam_pose[3:])
            world_pos = perception.backproject_to_world(
                float(centroid_px[0]), float(centroid_px[1]),
                TABLE_Z, cam_pos, R, fx, fy, cx, cy,
            )
            state.world_pos = world_pos
            state.found = True
            print(f"  [vision] TARGET acquired at {world_pos}")
            raise ObjectDetected(world_pos)

        return True
    return cb


# ---------------------------------------------------------------------------
# Match detection -> scene handle (within figure+color subset)
# ---------------------------------------------------------------------------
def match_to_scene_handle(world_pos, figure: str, color: str, sim):
    paths = SCENE_INDEX.get((figure, color))
    if paths is None or len(paths) == 0:
        paths = SCENE_INDEX.get((figure, "any"), [])

    candidates = []
    for path in paths:
        try:
            handle = sim.getObject(path)
        except Exception:
            continue
        pos = sim.getObjectPosition(handle, sim.handle_world)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, world_pos)))
        candidates.append((d, path, handle))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, path, handle = candidates[0]
    return path, handle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    figure = select_menu("Which figure should I pick?", SELECTABLE_FIGURES, allow_other=True)
    color_options = COLORS_PER_FIGURE.get(figure, ["any"])
    color  = select_menu(f"Which color of {figure}?", color_options, allow_other=True)
    print(f"\n>>> Target: {color} {figure}")

    sim = mp.sim
    # Verify simulation is running before we start.
    sim_state = sim.getSimulationState()
    if sim_state == sim.simulation_stopped:
        print("\nERROR: simulation is not running. Press the play button in CoppeliaSim and re-run.")
        return

    mp.initial_config = [sim.getJointPosition(j) for j in mp.joint_handles]
    arm_base = sim.getObjectPosition(sim.getObject("/Franka"), sim.handle_world)
    sensor_handle = sim.getObject("/Franka/visionSensor")

    state = DetectionState(figure, color)
    safety_check = make_safety_check(state, sensor_handle)

    # Read-only display thread for smooth live overlay.
    display_thread = threading.Thread(
        target=display_thread_fn,
        args=(state, "/Franka/visionSensor"),
        daemon=True,
    )
    display_thread.start()
    time.sleep(0.3)  # let the first frame come up

    found_pos = None

    try:
        print("\n=== Sweeping for target ===")
        try:
            perception.scan_for_object(
                arm_base=arm_base,
                tip_z=OBSERVATION_TIP_Z,
                radius=SWEEP_RADIUS,
                tilts=SWEEP_TILTS,
                n_azimuths=N_AZIMUTHS,
                detection_callback=safety_check,
            )
            print(f"\n'{color} {figure}' not found in the scene.")
        except ObjectDetected as od:
            found_pos = od.world_position
            print(f"\nDetected '{color} {figure}' at world position "
                  f"[{found_pos[0]:.3f}, {found_pos[1]:.3f}, {found_pos[2]:.3f}]")

        if found_pos is not None:
            state.search_mode = False  # don't abort future motions on detection

            path, handle = match_to_scene_handle(found_pos, figure, color, sim)
            if handle is None:
                print(f"Could not match detection to a scene object for ({figure}, {color}).")
            else:
                print(f"Matched to scene object {path}.")
                try:
                    destination_handle = sim.getObject(DESTINATION_PATH)
                except Exception:
                    print(f"ERROR: destination '{DESTINATION_PATH}' not found.")
                    destination_handle = None

                if destination_handle is not None:
                    print("\n=== Picking ===")
                    pick_ok = mp.moveToAndPickOrReleaseObject(
                        pickObject=True, cube=handle,
                        aboveApproximationDistance=0.1, heightThreshold=0.08,
                    )
                    if pick_ok is False:
                        print("Pick failed — gripper could not attach.")
                    else:
                        print("\n=== Placing ===")
                        mp.moveToAndPickOrReleaseObject(
                            pickObject=False, cube=handle,
                            aboveApproximationDistance=0.1, heightThreshold=0.12,
                            releaseDestination=destination_handle,
                        )

        print("\n=== Returning to initial pose ===")
        try:
            mp.returnToInitialPose()
        except Exception as e:
            print(f"Return-home failed: {e}")

    finally:
        # Minimal cleanup only — anything that can block (thread join on a
        # stuck ZMQ call, cv2.destroyAllWindows on Wayland) is skipped.
        # os._exit() below tears down Qt windows, ZMQ contexts, and the
        # daemon display thread atomically.
        state.stop_event.set()
        print("Done.")


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 130
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
