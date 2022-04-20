import numpy as np
import cv2
from queue import Queue

distorted = []

for i in range(20):
    image= cv2.imread("./camera_cal/calibration"+str(i+1)+".jpg")
    distorted.append(image)
    
row, nx, ny= 0, 9, 6

object_points=[]
image_points=[]

objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2]= np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for image in distorted:
    
    gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, corners= cv2.findChessboardCorners(gray, (nx, ny), None)
    
    if(ret):
        object_points.append(objp)
        image_points.append(corners)
        if row>3:
            continue
        cv2.drawChessboardCorners(image,(nx,ny), corners, ret)
        row += 1
        
        
image_test= cv2.cvtColor(cv2.imread("./camera_cal/calibration1.jpg"),cv2.COLOR_BGR2RGB)

y=image_test.shape[0]
x=image_test.shape[1]

_ ,mtx ,dist ,_ ,_ = cv2.calibrateCamera(object_points, image_points,(y,x),None,None)


def undistort(img):
    return cv2.undistort(img,mtx,dist, None, mtx)
    
offset=200
image_test1= cv2.imread("./test_images/straight_lines1.jpg")
height, width= image_test1.shape[0], image_test1.shape[1]
src=np.float32([(593,450),(700,450),(1200,700),(200,700)])
dst=np.float32([(offset,0),(width-offset,0),(width-offset,height),(offset,height)])

def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    M= cv2.getPerspectiveTransform(src, dst) 
    inv= cv2.getPerspectiveTransform(dst, src)
    warped= cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,inv

def reverse_warping(img,M):
    img_size = (img.shape[1], img.shape[0])
    unwarped= cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return unwarped  

def channel_wise_thresholding(image,thresh):
    
    image = image*(255/np.max(image))

    binary_output = np.zeros_like(image)
    binary_output[(image > thresh[0]) & (image <= thresh[1])] = 1
    
    return binary_output
    
def custom_channel_converter(img):
    
    img1=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)[:,:,0] # Y channel
    img2=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)[:,:,1] #Cr channel
    img3=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)[:,:,1] #L channel
    img4=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)[:,:,2] #S channel
    return img1, img2, img3, img4