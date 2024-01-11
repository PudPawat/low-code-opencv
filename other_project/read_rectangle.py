import cv2
import json
from pathlib import Path

with Path("config/rectangles.json").open("r") as f:
    config = json.load(f)

print(config)
print(len(config['main']))
print(len(config['sub']['0']))

img = cv2.imread("data/1.bmp")

for i in range(len(config['main'])):
    main = config['main'][str(i)]
    # imCrop = im[int(main[1]):int(main[1] + main[3]), int(main[0]):int(main[0] + main[2])]
    y0 = int(main[1])
    y1 = int(main[1] + main[3])
    x0 = int(main[0])
    x1 = int(main[0] + main[2])
    cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),3)


    for j in range(len(config['sub'][str(i)])):
        sub = config['sub'][str(i)][str(j)]
        sub_y0 = int(sub[1])
        sub_y1 = int(sub[1] + sub[3])
        sub_x0 = int(sub[0])
        sub_x1 = int(sub[0] + sub[2])
        cv2.rectangle(img, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255), 3)

cv2.imshow("asd",img)
cv2.waitKey(0)
