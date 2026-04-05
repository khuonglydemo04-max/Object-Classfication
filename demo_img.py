from ultralytics import YOLO
import cv2

# Tải mô hình classification đã huấn luyện
model = YOLO('runs/classify/train/weights/best.pt')
image_path = "D:/AGU/Năm3_24-25/Object Classfication/mango_dataset/train/bad mangoes/-222-_jpg.rf.9b8dd7f6beb69e0f68959e594c647905.jpg"
img = cv2.imread(image_path)

results = model(img)
probs = results[0].probs
class_id = probs.top1
class_name = results[0].names[class_id]
conf = probs.data[class_id].item()

label = f"{class_name} ({conf*100:.2f}%)"
cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

cv2.imshow("Kết quả phân loại", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
