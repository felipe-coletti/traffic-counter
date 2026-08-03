import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture('video.mp4')

line1 = ((655, 445), (1100, 440))
line2 = ((655, 295), (830, 305))
line3 = ((350, 305), (550, 310))
line4 = ((125, 485), (470, 505))
line5 = ((950, 335), (1100, 415))
line6 = ((275, 300), (100, 425))

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

    cv2.line(
        annotated_frame,
        line1[0],
        line1[1],
        (0, 255, 255),
        3
    )

    cv2.line(
        annotated_frame,
        line2[0],
        line2[1],
        (0, 255, 255),
        3
    )

    cv2.line(
        annotated_frame,
        line3[0],
        line3[1],
        (0, 255, 255),
        3
    )

    cv2.line(
        annotated_frame,
        line4[0],
        line4[1],
        (0, 255, 255),
        3
    )

    cv2.line(
        annotated_frame,
        line5[0],
        line5[1],
        (0, 255, 255),
        3
    )

    cv2.line(
        annotated_frame,
        line6[0],
        line6[1],
        (0, 255, 255),
        3
    )

    cv2.putText(
        annotated_frame,
        'Street 1',
        (line1[0][0], line1[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        'Street 2',
        (line2[0][0], line2[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        'Street 3',
        (line3[0][0], line3[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        'Street 4',
        (line4[0][0], line4[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        'Street 5',
        (line5[0][0], line5[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        'Street 6',
        (line6[0][0], line6[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow('Traffic Counter', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()