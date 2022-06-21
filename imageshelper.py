import cv2
import numpy as np

def resizeImage(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def stackImages(imgArray,scale,lables=[]):
    imgArray= np.float32(imgArray)
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list) or isinstance(imgArray[0],np.ndarray)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    imageArrBlank = np.zeros((imgArray.shape[0],imgArray.shape[1],int(height*scale), int(width*scale)), np.float32)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                resized = cv2.resize(imgArray[x][y], (0,0), fx=scale, fy=scale)
                imageArrBlank[x][y] = resized*255
        imageBlank = np.zeros((height*scale, width*scale), np.float32)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imageArrBlank[x])
            hor_con[x] = np.concatenate(imageArrBlank[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imageArrBlank[x] = cv2.resize(imageArrBlank[x], (0,0), fx=scale, fy=scale)
        hor= np.hstack(imageArrBlank)
        hor_con= np.concatenate(imageArrBlank)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def truncate_to_one(inputImg):
    img = np.float32(inputImg)
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            img[y, x] = img[y, x] / 255.0

    return img


