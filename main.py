import cv2
import numpy as np

# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('images/scan_1.jpeg')

scale_percent = 30 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image = resized


original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 2)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cv2.imshow('tresh', thresh)

# Find contours and filter using contour area filtering to remove noise
cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
AREA_THRESHOLD = 10
for c in cnts:
    area = cv2.contourArea(c)
    if area < AREA_THRESHOLD:
        cv2.drawContours(thresh, [c], -1, 0, -1)

# Repair checkbox horizontal and vertical walls
repair_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
repair = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, repair_kernel1, iterations=1)
repair_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, repair_kernel2, iterations=1)

# Detect checkboxes using shape approximation and aspect ratio filtering
checkbox_contours = []
cnts, _ = cv2.findContours(repair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.035 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    
    if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2) and (w > 40 and w < 50):
        imgCrop = thresh[y:(y+h),x:(x+w)]
        n_white_pix = np.sum(imgCrop == 255)
        if n_white_pix > 700:
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 3)
            checkbox_contours.append(1)
        else:
            cv2.rectangle(original, (x, y), (x + w, y + h), (12,12,255), 3)    
            checkbox_contours.append(0)
        
res = checkbox_contours[::-1]
votecount = sum(res)

if votecount < 1 or votecount > 1:
    print('invalid')
else:
    for num,value in enumerate(res,start=1):
        print(str(num)+": "+str(value))

#cv2.imshow('tresh', thresh)
#cv2.imshow('repair', repair)
#cv2.imshow('original', original)
#cv2.waitKey()