from ultralytics import YOLO
import cv2

# Tải mô hình đã huấn luyện
model = YOLO("runs/classify/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    probs = results[0].probs 
    class_id = probs.top1
    class_name = results[0].names[class_id]
    conf = probs.data[class_id].item()

    if class_name != "Me" and conf >= 0.3:
        label = f"{class_name} ({conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        label = "Khong Tim Thay Vat The"
        color = (0, 0, 255)

    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow("Mango Classification (Webcam)", frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
