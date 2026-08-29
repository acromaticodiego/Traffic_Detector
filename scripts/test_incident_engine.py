from services.vision_service.app.tracking.track_state import (
    TrackState,
)

from services.vision_service.app.motion.motion_analyzer import (
    MotionAnalyzer,
)

from services.vision_service.app.incidents.incident_engine import (
    IncidentEngine,
)


# ==========================================
# CREATE TRACKS
# ==========================================

track_a = TrackState(
    track_id=1,
    class_id=1,
    class_name="car",
    confidence=0.95,
    x1=100,
    y1=100,
    x2=140,
    y2=140,
)

track_b = TrackState(
    track_id=2,
    class_id=3,
    class_name="motorcycle",
    confidence=0.90,
    x1=145,
    y1=100,
    x2=185,
    y2=140,
)


# ==========================================
# MOTION ANALYZER
# ==========================================

motion_analyzer = MotionAnalyzer()


# ==========================================
# INITIAL FRAME
# ==========================================

motion_analyzer.analyze(
    track_a
)

motion_analyzer.analyze(
    track_b
)


# ==========================================
# NORMAL MOVEMENT
# ==========================================

track_a.update(
    confidence=0.95,
    x1=102,
    y1=102,
    x2=142,
    y2=142,
)

track_b.update(
    confidence=0.90,
    x1=143,
    y1=102,
    x2=183,
    y2=142,
)


motion_analyzer.analyze(
    track_a
)

motion_analyzer.analyze(
    track_b
)


# ==========================================
# SIMULATE SUDDEN MOVEMENT
# ==========================================

track_a.update(
    confidence=0.95,
    x1=120,
    y1=120,
    x2=160,
    y2=160,
)

track_b.update(
    confidence=0.90,
    x1=125,
    y1=120,
    x2=165,
    y2=160,
)


motion_a = motion_analyzer.analyze(
    track_a
)

motion_b = motion_analyzer.analyze(
    track_b
)


# ==========================================
# INCIDENT ENGINE
# ==========================================

incident_engine = IncidentEngine()


incidents = incident_engine.process(
    tracks=[
        track_a,
        track_b,
    ],
    motion_analysis=[
        motion_a,
        motion_b,
    ],
)


# ==========================================
# OUTPUT
# ==========================================

print("=" * 60)
print("INCIDENT ENGINE TEST")
print("=" * 60)

print()

print(
    f"Track A: "
    f"{track_a.class_name} "
    f"ID={track_a.track_id}"
)

print(
    f"Track B: "
    f"{track_b.class_name} "
    f"ID={track_b.track_id}"
)

print()

print("Motion A:")
print(
    f"  Speed: "
    f"{motion_a.speed:.2f}"
)

print(
    f"  Acceleration: "
    f"{motion_a.acceleration:.2f}"
)

print(
    f"  Moving: "
    f"{motion_a.moving}"
)

print(
    f"  Abrupt change: "
    f"{motion_a.abrupt_change}"
)

print()

print("Motion B:")
print(
    f"  Speed: "
    f"{motion_b.speed:.2f}"
)

print(
    f"  Acceleration: "
    f"{motion_b.acceleration:.2f}"
)

print(
    f"  Moving: "
    f"{motion_b.moving}"
)

print(
    f"  Abrupt change: "
    f"{motion_b.abrupt_change}"
)

print()

print(
    f"Incidents detected: "
    f"{len(incidents)}"
)

print()

for incident in incidents:

    print("[INCIDENT]")

    print(
        f"  Type: "
        f"{incident.incident_type}"
    )

    print(
        f"  Track IDs: "
        f"{incident.track_ids}"
    )

    print(
        f"  Confidence: "
        f"{incident.confidence:.2f}"
    )

    print(
        f"  Data: "
        f"{incident.data}"
    )

print()

print("=" * 60)
print("INCIDENT ENGINE TEST FINISHED")
print("=" * 60)