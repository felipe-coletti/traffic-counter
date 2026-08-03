import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture('video.mp4')

lines = [
    ('Street 1', (655, 445), (1100, 440)),
    ('Street 2', (655, 295), (830, 305)),
    ('Street 3', (350, 305), (550, 310)),
    ('Street 4', (125, 485), (470, 505)),
    ('Street 5', (950, 335), (1100, 415)),
    ('Street 6', (275, 300), (100, 425)),
]

vehicle_tracks = {}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker='bytetrack.yaml',
        classes=[2, 3, 5, 7],
        conf=0.35,
        verbose=False
    )
    annotated_frame = results[0].plot()

    for name, p1, p2 in lines:
        cv2.line(
            annotated_frame,
            p1,
            p2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            annotated_frame,
            name,
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    boxes = results[0].boxes

    if boxes.id is not None:
        for box in boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            vehicle_tracks[track_id] = (cx, cy)

            cv2.circle(
                annotated_frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

    cv2.imshow('Traffic Counter', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()