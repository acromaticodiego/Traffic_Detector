from services.vision_service.app.motion.motion_analyzer import (
    MotionAnalyzer,
)
from services.vision_service.app.tracking.track_state import (
    TrackState,
)


def main():

    print("=" * 60)
    print("MOTION ANALYZER TEST")
    print("=" * 60)

    analyzer = MotionAnalyzer()

    # ==========================================
    # CREATE TEST TRACK
    # ==========================================

    track = TrackState(
        track_id=1,
        class_id=1,
        class_name="car",
        confidence=0.95,
        x1=100,
        y1=100,
        x2=140,
        y2=140,
    )

    # ==========================================
    # SIMULATE MOVEMENT
    # ==========================================

    positions = [
        (100, 100),
        (102, 102),
        (105, 105),
        (110, 110),
        (120, 120),
        (135, 135),
        (136, 136),
    ]

    for index, (x, y) in enumerate(
        positions,
        start=1,
    ):

        track.update(
            confidence=0.95,
            x1=x,
            y1=y,
            x2=x + 40,
            y2=y + 40,
        )

        analysis = analyzer.analyze(
            track
        )

        print()
        print(f"Frame {index}")

        print(
            f"  Track ID: "
            f"{analysis.track_id}"
        )

        print(
            f"  Movement: "
            f"dx={analysis.dx:.2f} "
            f"dy={analysis.dy:.2f}"
        )

        print(
            f"  Speed: "
            f"{analysis.speed:.2f}"
        )

        print(
            f"  Direction: "
            f"{analysis.direction:.2f}°"
        )

        print(
            f"  Moving: "
            f"{analysis.moving}"
        )

        print(
            f"  Acceleration: "
            f"{analysis.acceleration:.2f}"
        )

        print(
            f"  Abrupt change: "
            f"{analysis.abrupt_change}"
        )

    print()
    print("=" * 60)
    print("MOTION ANALYZER TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()