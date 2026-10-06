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
               a candidate / confirmed person makes the drone HOLD position,
               a rejection / lost target RESUMES the patrol
  6. RETURN    return to the launch pad and land (or hover)
Ctrl+C at any time = return to launch (a safe default, never "just stop").
"""
import argparse
import asyncio
import math
import signal
import sys
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


def start_search_listener(loop, queue):
    """Listen to /aware/search_state (published by the AI) in a background thread
    and forward each state ("candidate", "confirmed", "rejected", "lost", ...) into
    an asyncio queue. Returns False if ROS isn't available (patrol still works)."""
    try:
        import json
        import threading
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import String
    except ImportError:
        return False

    def run():
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("aware_patrol_listener")

        def on_msg(msg):
            try:
                state = json.loads(msg.data).get("state")
            except ValueError:
                return
            loop.call_soon_threadsafe(queue.put_nowait, state)

        node.create_subscription(String, "/aware/search_state", on_msg, 10)
        rclpy.spin(node)

    threading.Thread(target=run, daemon=True).start()
    return True


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
        states = asyncio.Queue()
        if hold_on_target and start_search_listener(loop, states):
            print("  listening to the AI on /aware/search_state (hold on candidate/confirmed)")

        async def follow_ai():
            paused = False
            while True:
                state = await states.get()
                print(f"  AI state: {state}")
                if state in ("candidate", "confirmed") and not paused:
                    await mission.pause_mission()
                    paused = True
                    print(f"  AI: {state.upper()} -> HOLDING POSITION over the person")
                elif state == "confirmed":
                    print("  AI: CONFIRMED -> keep holding (Ctrl+C = return and land)")
                elif state in ("rejected", "lost", "searching") and paused:
                    await mission.start_mission()
                    paused = False
                    print(f"  AI: {state.upper()} -> RESUMING the patrol")

        async def progress():
            async for mp in mission.subscribe_mission_progress():
                print(f"  waypoint {mp.current}/{mp.total}")
                if mp.total and mp.current >= mp.total:
                    return
        ai_task = asyncio.ensure_future(follow_ai())
        await until_stopped(progress(), stop)
        ai_task.cancel()

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
