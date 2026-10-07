#!/usr/bin/env python3
"""
patrol.py: fly the AWARE lawnmower patrol with MAVSDK (v4, native asyncio API).

Run inside the AWARE venv, with PX4 already running:
    aware && python3 aware_ws/src/aware_sim/scripts/patrol.py
    python3 patrol.py --dry-run      # build the mission, don't connect
    python3 patrol.py --loops 1      # override patrol.yaml
    python3 patrol.py --start enter  # how the flight begins (default: takeoff)

Start modes (--start):
  takeoff  (default) script prepares everything, then WAITS for YOU to take off
           (type `commander takeoff` in the PX4 terminal); the patrol begins
           once the drone reaches patrol altitude
  enter    script waits for you to press Enter, then takes off by itself
  auto     takes off immediately (old behaviour)

Flight phases (like a pilot's checklist):
  1. CONNECT   find PX4 on the MAVLink UDP port
  2. PREFLIGHT wait until PX4 has a GPS position and a home position
  3. UPLOAD    send the waypoint list (the "flight plan") to PX4
  4. TAKEOFF   arm the motors and climb to patrol altitude
  5. MISSION   PX4 flies the waypoints itself; we watch progress and the AI:
               a candidate / confirmed person PAUSES the patrol, and the
               drone flies to LOOK AT the person (the AI sends where they
               stand on /aware/target) and follows them if they walk;
               a rejection / lost target RESUMES the patrol.
               The drone's position is published on /aware/drone/state
               (JSON, 10 times per second) so the AI / dashboard can
               work out WHERE people stand.
  6. RETURN    return to the launch pad and land (or hover)
Ctrl+C at any time = return to launch (a safe default, never "just stop").
"""
import argparse
import asyncio
import math
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patrol_plan as P  # noqa: E402

try:
    import mavsdk
    from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
    from mavsdk.asyncio.plugins.action.action import ActionAsync
    from mavsdk.asyncio.plugins.mission.mission import MissionAsync, MissionItem, MissionPlan
    from mavsdk.asyncio.plugins.telemetry.telemetry import TelemetryAsync
    from mavsdk.asyncio.plugins.param.param import ParamAsync
except ImportError as e:
    sys.exit(f"MAVSDK v4 not available ({e}).\n"
             f"Activate the venv (aware) and run: pip install --upgrade \"mavsdk>=4\"")

NAN = math.nan
DRONE_STATE_TOPIC = "/aware/drone/state"
DRONE_STATE_PERIOD_S = 0.1          # publish the drone state ten times per second
TARGET_TOPIC = "/aware/target"      # where the AI's candidate stands

# Looking at / following the candidate
CAMERA_PITCH_DEG = 45.0             # must match pitch_deg in aware_sim/config/drone.yaml
FOLLOW_MAX_DISTANCE_M = 80.0        # never fly further than this to look at someone
FOLLOW_MIN_MOVE_M = 2.0             # re-aim only if the person moved at least this much
FOLLOW_MIN_PERIOD_S = 1.0           # at most one new flight command per second
FOLLOW_TARGET_MAX_AGE_S = 3.0       # ignore target positions older than this
METERS_PER_DEG_LAT = 111_320.0


def mission_item(lat, lon, alt, speed):
    """One waypoint. MAVSDK v4 needs every field; NaN = 'use PX4's default'."""
    return MissionItem(
        latitude_deg=lat, longitude_deg=lon, relative_altitude_m=alt, speed_m_s=speed,
        is_fly_through=True,             # don't stop at each waypoint, curve through it
        gimbal_pitch_deg=NAN, gimbal_yaw_deg=NAN,
        camera_action=MissionItem.CameraAction.NONE,
        loiter_time_s=NAN, camera_photo_interval_s=NAN, acceptance_radius_m=NAN,
        yaw_deg=NAN,                     # NaN: PX4 points the nose toward the next waypoint
        camera_photo_distance_m=NAN,
        vehicle_action=MissionItem.VehicleAction.NONE)


def build_mission(venue, patrol, loops):
    p = P.plan(venue, patrol)
    items = []
    for _ in range(loops):
        for x, y in p["waypoints"]:
            lat, lon = P.enu_to_geo(x, y, venue["venue"]["location"])
            items.append(mission_item(lat, lon, patrol["altitude_m"], patrol["speed_m_s"]))
    return p, items


def start_ros(loop, queue, target_holder=None):
    """Start ROS in a background thread:
      - listen to /aware/search_state (published by the AI) and forward each state
        ("candidate", "confirmed", "rejected", "lost", ...) into the asyncio queue
        (if queue is None, the AI states are ignored: --no-hold)
      - listen to /aware/target (where the candidate stands) and keep the newest
        one in target_holder["target"]
      - publish the drone position on /aware/drone/state
    Returns a publish(dict) function, or None if ROS isn't available
    (the patrol still works without ROS)."""
    try:
        import json
        import threading
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import String
    except ImportError:
        return None

    holder = {}
    ready = threading.Event()

    def run():
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("aware_patrol_listener")
        holder["pub"] = node.create_publisher(String, DRONE_STATE_TOPIC, 10)

        def on_msg(msg):
            if queue is None:
                return
            try:
                state = json.loads(msg.data).get("state")
            except ValueError:
                return
            loop.call_soon_threadsafe(queue.put_nowait, state)

        node.create_subscription(String, "/aware/search_state", on_msg, 10)

        def on_target(msg):
            if target_holder is None:
                return
            try:
                data = json.loads(msg.data)
            except ValueError:
                return
            target_holder["target"] = data
            target_holder["received"] = time.monotonic()

        node.create_subscription(String, TARGET_TOPIC, on_target, 10)
        ready.set()
        rclpy.spin(node)

    threading.Thread(target=run, daemon=True).start()
    ready.wait(5.0)

    def publish(data):
        pub = holder.get("pub")
        if pub is None:
            return
        msg = String()
        msg.data = json.dumps(data)
        pub.publish(msg)

    return publish


def _local_m(origin, point):
    """(north, east) metres from origin to point, both (lat, lon)."""
    north = (point[0] - origin[0]) * METERS_PER_DEG_LAT
    east = (point[1] - origin[1]) * METERS_PER_DEG_LAT * math.cos(math.radians(origin[0]))
    return north, east


def _offset(origin, north, east):
    """(lat, lon) of origin moved by north / east metres."""
    lat = origin[0] + north / METERS_PER_DEG_LAT
    lon = origin[1] + east / (METERS_PER_DEG_LAT * math.cos(math.radians(origin[0])))
    return lat, lon


def viewpoint(target, drone, alt_rel_m, heading_deg):
    """Where the drone should hover, and which way it should face, so the
    (fixed, tilted) camera looks straight at target.

    The camera looks down at CAMERA_PITCH_DEG, so it sees the ground
    alt / tan(pitch) metres ahead of the drone. The viewpoint is that far from
    the target, on the side the drone is already on (shortest flight).
    Returns (lat, lon, yaw_deg)."""
    ahead_m = alt_rel_m / math.tan(math.radians(CAMERA_PITCH_DEG))
    n, e = _local_m(target, drone)                # target -> drone
    d = math.hypot(n, e)
    if d < 1.0:                                   # right above: back off against the heading
        yaw = math.radians(heading_deg if heading_deg is not None else 0.0)
        n, e, d = -math.cos(yaw), -math.sin(yaw), 1.0
    un, ue = n / d, e / d
    view = _offset(target, un * ahead_m, ue * ahead_m)
    yaw_deg = math.degrees(math.atan2(-ue, -un)) % 360.0   # face the target
    return view[0], view[1], yaw_deg


async def until_stopped(coro, stop):
    """Run coro, but give up early if the user pressed Ctrl+C (stop is set).
    Returns (finished: bool, result)."""
    task = asyncio.ensure_future(coro)
    stopper = asyncio.ensure_future(stop.wait())
    done, _ = await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
    for t in (task, stopper):
        if t not in done:
            t.cancel()
    if task in done:
        return True, task.result()
    return False, None


async def wait_until(gen, condition, label, timeout_s):
    """Read an async telemetry stream until condition(value) is true."""
    print(f"  waiting: {label} ...")
    async def _run():
        async for value in gen:
            if condition(value):
                return value
    return await asyncio.wait_for(_run(), timeout_s)


async def fly(venue, patrol, loops, start_mode, hold_on_target=True):
    p, items = build_mission(venue, patrol, loops)
    alt = patrol["altitude_m"]

    # 1. CONNECT ----------------------------------------------------------
    print("[1/6] CONNECT", patrol["connection"])
    config = Configuration.create_with_component_type(ComponentType.GROUND_STATION)
    async with Mavsdk(config) as sdk:
        await sdk.add_any_connection(patrol["connection"])
        system = await sdk.first_autopilot(timeout_s=20.0)
        if system is None:
            sys.exit("No PX4 found. Is PX4 running (T2) and did it finish starting?")
        action, mission = ActionAsync(system), MissionAsync(system)
        telemetry, param = TelemetryAsync(system), ParamAsync(system)
        print("  connected to PX4")

        # 2. PREFLIGHT ----------------------------------------------------
        print("[2/6] PREFLIGHT")
        await wait_until(telemetry.subscribe_health(),
                         lambda h: h.is_global_position_ok and h.is_home_position_ok,
                         "GPS position + home position", 120)
        # Return-to-launch at patrol altitude (PX4's default climbs much higher)
        await param.set_param_float("RTL_RETURN_ALT", float(alt))
        print("  ready")

        # 3. UPLOAD -------------------------------------------------------
        print(f"[3/6] UPLOAD {len(items)} waypoints "
              f"({p['strips']} strips x {loops} loop(s), {alt} m, {patrol['speed_m_s']} m/s)")
        await mission.set_return_to_launch_after_mission(patrol["end_with"] == "rtl")
        await mission.upload_mission(MissionPlan(items))

        # Ctrl+C from here on is handled by us (not an instant crash)
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, stop.set)

        # 4. TAKEOFF ------------------------------------------------------
        # commander takeoff / action.takeoff both climb to this altitude
        await action.set_takeoff_altitude(float(alt))
        if start_mode == "takeoff":
            print("[4/6] TAKEOFF: waiting for YOU. In the PX4 terminal (T2) type:")
            print("          commander takeoff")
            print("      (Ctrl+C here cancels before anything flies)")
            ok, _ = await until_stopped(
                wait_until(telemetry.subscribe_in_air(), lambda air: air,
                           "drone in the air", 24 * 3600), stop)
            if not ok:
                print("  cancelled before takeoff. Nothing flew.")
                return
            print("  takeoff detected")
        else:
            if start_mode == "enter":
                print("[4/6] TAKEOFF: press Enter to take off (Ctrl+C cancels)")
                ok, _ = await until_stopped(loop.run_in_executor(None, sys.stdin.readline), stop)
                if not ok:
                    print("  cancelled before takeoff. Nothing flew.")
                    return
            print("[4/6] TAKEOFF: arming and climbing")
            await action.arm()
            await action.takeoff()

        ok, _ = await until_stopped(
            wait_until(telemetry.subscribe_position(),
                       lambda pos: pos.relative_altitude_m >= 0.9 * alt,
                       f"climb to {alt} m", 120), stop)
        if not ok:
            print("[6/6] Ctrl+C during climb -> RETURN TO LAUNCH")
            await action.return_to_launch()
            await wait_until(telemetry.subscribe_in_air(), lambda air: not air, "landing", 300)
            print("  landed.")
            return

        # 5. MISSION ------------------------------------------------------
        print("[5/6] MISSION")
        await mission.start_mission()

        # The AI tells us what it's doing on /aware/search_state:
        #   candidate / confirmed -> HOLD position (keep the person in view)
        #   rejected / lost       -> RESUME the patrol where it stopped
        # And we tell the AI where the drone is, on /aware/drone/state.
        states = asyncio.Queue()
        target_holder = {}
        publish = start_ros(loop, states if hold_on_target else None, target_holder)
        if publish is not None:
            print(f"  publishing the drone position on {DRONE_STATE_TOPIC}")
            if hold_on_target:
                print("  listening to the AI on /aware/search_state (hold on candidate/confirmed)")

        # Latest drone state (filled by the telemetry subscriptions).
        drone = {"heading_deg": None, "lat": None, "lon": None,
                 "alt_abs_m": None, "alt_rel_m": None,
                 "roll_deg": 0.0, "pitch_deg": 0.0}

        async def look_at_target(follow):
            """While paused on a candidate: fly to where the camera sees them,
            and follow them if they walk. follow = {"goal": (lat, lon) or None,
            "time": last command time, "warned": bool}."""
            t = target_holder.get("target")
            now = time.monotonic()
            if (t is None or t.get("lat") is None
                    or t.get("state") not in ("candidate", "confirmed")
                    or now - target_holder.get("received", 0.0) > FOLLOW_TARGET_MAX_AGE_S):
                return
            if None in (drone["lat"], drone["alt_abs_m"], drone["alt_rel_m"]):
                return
            goal = (t["lat"], t["lon"])
            if now - follow["time"] < FOLLOW_MIN_PERIOD_S:
                return
            if follow["goal"] is not None:
                moved = math.hypot(*_local_m(follow["goal"], goal))
                if moved < FOLLOW_MIN_MOVE_M:
                    return                                    # person barely moved
            here = (drone["lat"], drone["lon"])
            dist = math.hypot(*_local_m(here, goal))
            if dist > FOLLOW_MAX_DISTANCE_M:
                if not follow["warned"]:
                    print(f"  AI: person is {dist:.0f} m away (> {FOLLOW_MAX_DISTANCE_M:.0f} m): "
                          "holding instead of flying there")
                    follow["warned"] = True
                return
            lat, lon, yaw = viewpoint(goal, here, drone["alt_rel_m"], drone["heading_deg"])
            try:
                await action.goto_location(lat, lon, drone["alt_abs_m"], yaw)
            except Exception as exc:
                print(f"  AI: could not fly to the person ({exc}); holding position")
                follow["time"] = now
                return
            first = follow["goal"] is None
            follow["goal"], follow["time"] = goal, now
            print(f"  AI: {'flying to look at' if first else 'following'} the person "
                  f"({dist:.0f} m away), facing {yaw:.0f} deg")

        async def follow_ai():
            paused = False
            follow = {"goal": None, "time": 0.0, "warned": False}
            while True:
                try:
                    state = await asyncio.wait_for(states.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    state = None
                if state is not None:
                    print(f"  AI state: {state}")
                    if state in ("candidate", "confirmed") and not paused:
                        await mission.pause_mission()
                        paused = True
                        follow = {"goal": None, "time": 0.0, "warned": False}
                        print(f"  AI: {state.upper()} -> pausing the patrol to look at the person")
                    elif state == "confirmed":
                        print("  AI: CONFIRMED -> keep following (Ctrl+C = return and land)")
                    elif state in ("rejected", "lost", "searching") and paused:
                        await mission.start_mission()
                        paused = False
                        print(f"  AI: {state.upper()} -> RESUMING the patrol")
                if paused:
                    await look_at_target(follow)

        async def track_heading():
            try:
                async for h in telemetry.subscribe_heading():
                    drone["heading_deg"] = h.heading_deg
            except Exception as exc:                   # position still works without it
                print(f"  (no heading telemetry: {exc})")

        async def track_attitude():
            try:
                async for attitude in telemetry.subscribe_attitude_euler():
                    drone["roll_deg"] = attitude.roll_deg
                    drone["pitch_deg"] = attitude.pitch_deg
            except Exception as exc:
                print(f"  (no attitude telemetry: {exc})")

        async def track_position():
            try:
                async for pos in telemetry.subscribe_position():
                    drone["lat"], drone["lon"] = pos.latitude_deg, pos.longitude_deg
                    drone["alt_abs_m"] = pos.absolute_altitude_m
                    drone["alt_rel_m"] = pos.relative_altitude_m
            except Exception as exc:
                print(f"  (drone position telemetry stopped: {exc})")

        async def publish_drone_state():
            # Publish on a timer so a slower GPS stream does not limit the rate.
            try:
                while True:
                    if drone["lat"] is not None:
                        publish({**drone, "time": time.time()})
                    await asyncio.sleep(DRONE_STATE_PERIOD_S)
            except Exception as exc:
                print(f"  (drone state publishing stopped: {exc})")

        async def progress():
            async for mp in mission.subscribe_mission_progress():
                print(f"  waypoint {mp.current}/{mp.total}")
                if mp.total and mp.current >= mp.total:
                    return

        tasks = [asyncio.ensure_future(follow_ai())]
        if publish is not None:
            tasks.append(asyncio.ensure_future(track_heading()))
            tasks.append(asyncio.ensure_future(track_attitude()))
            tasks.append(asyncio.ensure_future(track_position()))
            tasks.append(asyncio.ensure_future(publish_drone_state()))
        await until_stopped(progress(), stop)
        for t in tasks:
            t.cancel()

        # 6. RETURN -------------------------------------------------------
        if stop.is_set():
            print("[6/6] Ctrl+C -> RETURN TO LAUNCH")
            await action.return_to_launch()
        elif patrol["end_with"] == "rtl":
            print("[6/6] mission complete -> PX4 returns to the launch pad")
        else:
            print("[6/6] mission complete -> hovering (end_with: hold)")
            return
        await wait_until(telemetry.subscribe_in_air(), lambda air: not air, "landing", 300)
        print("  landed. Patrol finished.")


def main():
    ap = argparse.ArgumentParser(description="Fly the AWARE lawnmower patrol")
    ap.add_argument("--loops", type=int, help="override loops in patrol.yaml")
    ap.add_argument("--dry-run", action="store_true", help="build the mission only")
    ap.add_argument("--no-hold", action="store_true",
                    help="ignore the AI: never pause the patrol on a candidate/confirmed target")
    ap.add_argument("--start", choices=["takeoff", "enter", "auto"], default="takeoff",
                    help="takeoff: wait for YOUR commander takeoff (default) | "
                         "enter: wait for Enter | auto: take off immediately")
    args = ap.parse_args()
    venue, patrol = P.load()
    loops = args.loops or patrol["loops"]
    if args.dry_run:
        p, items = build_mission(venue, patrol, loops)
        print(f"MAVSDK {mavsdk.__version__}: {len(items)} mission items OK "
              f"({p['strips']} strips x {loops} loops)")
        for i, it in enumerate(items[:len(p['waypoints'])], 1):
            it.to_c_struct()   # same conversion the upload uses
            print(f"  item {i}: {it.latitude_deg:.7f}, {it.longitude_deg:.7f}, {it.relative_altitude_m} m")
        return
    try:
        asyncio.run(fly(venue, patrol, loops, args.start, hold_on_target=not args.no_hold))
    except asyncio.TimeoutError:
        sys.exit("Timed out waiting for the drone (see the last 'waiting:' line).")


if __name__ == "__main__":
    main()