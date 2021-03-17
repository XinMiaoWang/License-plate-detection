import numpy as np
import cv2


def brightness(gray_img):
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size

    # 灰階圖的直方圖
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 計算灰階圖像素點偏離均值(128)的流程
    a , ma = 0, 0
    # np.full 建立一個陣列，使用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 計算偏離128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)
    # 亮度係數
    k = abs(da) / m
    print('亮度係數',k)
    if k[0] > 1:
        if da > 0:
            print("過亮")
        else:
            print("過暗")
        return 1
    else:
        print("亮度正常")
        return 0

# 車牌傾斜校正
def rotate(image,angle,center=None,scale=1.0):
    (w,h) = image.shape[0:2]
    if center is None:
        center = (w//2,h//2) # 計算中心點
    wrapMat = cv2.getRotationMatrix2D(center,angle,scale) # 旋轉矩陣
    return cv2.warpAffine(image,wrapMat,(h,w)) #仿射變換

def detectLicense():
    img = cv2.imread('test1.jpeg')
    img = cv2.resize(img, (620,480) )
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉成灰階圖
    val = brightness(gray) # 檢查是否過暗、過亮


    if val == 1:
        clahe = cv2.createCLAHE()
        gray = clahe.apply(gray)
        cv2.imshow('gray', gray)
        img_Denoising = cv2.medianBlur(gray, 5)
    else:
        img_Denoising = cv2.bilateralFilter(gray, 13, 15, 15)

    cv2.imshow('img_Denoising', img_Denoising)

    # img_GaussianBlur = cv2.GaussianBlur(gray, (5, 5), 0)
    # img_bilateralFilter = cv2.bilateralFilter(gray, 13, 15, 15)
    # img_medianBlur = cv2.medianBlur(gray, 5)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img_Denoising, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edged = cv2.Canny(dilation, 30, 200) #Perform Edge detection
    # cv2.imshow('edged', edged)

    image, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours,key=cv2.contourArea, reverse = True)[:10] # 只取前10大輪廓
    screenCnt = None # 車牌位置

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4: # 若輪廓有四個點，即判斷為車牌
            screenCnt = approx
            break


    point = []
    for i in range(4):
        point.append(screenCnt[i][:][0])
    point = np.array(point)
    print(point)

    rect = cv2.minAreaRect(point)
    print(rect)
    angle = rect[2]
    print('angle : ',angle)

    if angle < 0:
        angle = angle
    else:
        angle = -angle


    result = cv2.polylines(img, pts=[screenCnt], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imshow("result", result)

    # Masking the part other than the number plate
    mask = np.zeros(img_Denoising.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    # 定位出車牌
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = img_Denoising[topx:bottomx+1, topy:bottomy+1]

    height, width = Cropped.shape[:2]
    (_, thresh) = cv2.threshold(Cropped, 150, 255, cv2.THRESH_BINARY)

    dst = rotate(thresh, angle) # 車牌傾斜校正

    cv2.imshow("dst", dst)
    cv2.imwrite('result1.jpg', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    detectLicense()


