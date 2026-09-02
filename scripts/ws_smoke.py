"""
Quick manual check of the inference WebSocket.

Usage (with the API running on :8000):

    .venv\\Scripts\\python scripts\\ws_smoke.py
    .venv\\Scripts\\python scripts\\ws_smoke.py --url ws://127.0.0.1:8000/ws/inference --stride 3 --max-frames 300
"""

import argparse
import asyncio
import json
import time

import websockets


async def run(url: str, max_frames: int) -> None:
    t0 = time.time()
    frames = 0
    incidents = 0
    first_tracks = None

    async with websockets.connect(url, max_size=None) as ws:
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            kind = msg.get("type")

            if kind == "meta":
                print("META", msg)
            elif kind == "frame":
                frames += 1
                if msg["tracks"] and first_tracks is None:
                    first_tracks = msg["frame_id"]
                    print(
                        f"first tracks @ frame {msg['frame_id']}: "
                        f"{len(msg['tracks'])} -> {msg['tracks'][0]}"
                    )
                if frames % 100 == 0:
                    rate = frames / (time.time() - t0)
                    print(f"...{frames} frames  t={msg['t']}s  {rate:.1f} fps")
            elif kind == "incident":
                incidents += 1
                print(
                    f"INCIDENT {msg['incident_type']} "
                    f"conf={msg['confidence']} ids={msg['track_ids']} t={msg['t']}"
                )
            elif kind == "done":
                print("DONE", msg)
                break
            elif kind == "error":
                print("ERROR", msg)
                break

            if max_frames and frames >= max_frames:
                print(f"stopping after {max_frames} frames")
                break

    print(
        f"frames={frames} incidents={incidents} "
        f"elapsed={time.time() - t0:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/inference",
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    url = f"{args.url}?stride={args.stride}"
    asyncio.run(run(url, args.max_frames))


if __name__ == "__main__":
    main()
