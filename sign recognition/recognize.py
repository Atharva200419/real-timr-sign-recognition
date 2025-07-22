import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/asl_model.h5")
classes = [chr(i) for i in range(65, 91)]  # A-Z

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.reshape(1, 64, 64, 1) / 255.0

    pred = model.predict(normalized)
    label = classes[np.argmax(pred)]

    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
