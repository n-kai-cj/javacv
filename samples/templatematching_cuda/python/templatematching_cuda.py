import cv2
import numpy as np

def getType(img):
    if img.ndim == 2:
        return cv2.CV_8UC1
    elif img.ndim == 3:
        return cv2.CV_8UC3

img = cv2.imread("messi5.jpg")
template = cv2.imread("template.jpg")
height, width= template.shape[:2]

templatematching_cuda = cv2.cuda.createTemplateMatching(getType(template), cv2.TM_SQDIFF)

img_cuda = cv2.cuda_GpuMat(img)
template_cuda = cv2.cuda_GpuMat(template)
res_cuda = templatematching_cuda.match(img_cuda, template_cuda)
res = res_cuda.download()

_, _, top_left, _ = cv2.minMaxLoc(res)
bottom_right = (top_left[0] + width, top_left[1] + height)
cv2.rectangle(img, top_left, bottom_right, (255, 139, 0), 2)
cv2.imshow("TemplateMatchingCuda", img)
while cv2.waitKey(1000) != 27:
    pass

cv2.destroyAllWindows()
