from ultralytics import YOLO
import cv2 as cv
import cvzone

# vid = cv.VideoCapture("Video1.mp4")
img = cv.imread("Img1.jpg")
model = YOLO("best.pt")

while True:
    # _ , img = vid.read()
    img = cv.resize(img,(1080,720))
    results = model(img , stream=True , verbose=False)
    # print(results)
    person = 0
    for result in results:
        # print(result)
        boxes = result.boxes
        for box in boxes:
            # print(box)
            cls = int(box.cls)
            conf = int(box.conf[0]*100)/100
            if cls == 5:
                person+=1
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1
                cvzone.cornerRect(img, (x1,y1,w,h), t=2, colorC=(255,0,255), colorR=(255,0,255))
                cvzone.putTextRect(img, f'{conf}', (x1,y1), scale=1, thickness=1, offset=5)
    cv.imshow("Video",img)
    print(person)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break


