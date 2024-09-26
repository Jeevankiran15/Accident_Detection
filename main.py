import cv2
from ultralytics import YOLO


model = YOLO('runs/detect/train4/weights/best.pt')  

cap = cv2.VideoCapture("test3.mp4") 


while True:
  
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model(frame,conf = 0.6)
   
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLOv8 Webcam Inference', annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

