import cv2
import numpy as np

img = cv2.imread("messi5.jpg")
template = cv2.imread("template.jpg")
height, width = template.shape[:2]

res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
_, _, top_left, _ = cv2.minMaxLoc(res)
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(img, top_left, bottom_right, (255, 139, 0), 2)
cv2.imshow("TemplateMatching", img)
while cv2.waitKey(1000) != 27:
    pass

cv2.destroyAllWindows()
