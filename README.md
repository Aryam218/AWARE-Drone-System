# AWARE – Drone System

AI-powered autonomous drone for **crowd analytics** and **missing-person search**
at large events, demonstrated in a simulated Expo venue in Riyadh.

| Folder | What it is | Guide |
|---|---|---|
| [`simulation/`](simulation/) | The simulated world: venue, 248 animated people, PX4 drone with camera, autonomous patrol | **[simulation/README.md](simulation/README.md)** — start here |
| [`rasid_video_pipeline/`](rasid_video_pipeline/) | The AI search: Grounding DINO + GPT verification + tracking, and the dashboard backend | |

## Quick start (Ubuntu 24.04)

```bash
git clone https://github.com/Aryam218/AWARE-Drone-System.git
cd AWARE-Drone-System/simulation
bash setup/install.sh      # once: installs everything (30–90 min), then restart the computer
bash setup/check.sh        # everything PASS?
aware_run                  # starts the whole system
```

Full step-by-step instructions, for people new to Linux, Gazebo and ROS:
**[simulation/README.md](simulation/README.md)**.

## How it works

```
Gazebo world (248 people) ──► drone camera ──► ROS 2 /aware/camera/image
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
  Grounding DINO (find people) ──► color filter (cheap) ──► GPT check (expensive)
        │                                                        │
        │                                            MATCH ──► operator confirms (c / r, or dashboard)
        ▼                                                        │
  /aware/search_state ──► patrol HOLDS over the person ◄─────────┘ ──► tracking
```
