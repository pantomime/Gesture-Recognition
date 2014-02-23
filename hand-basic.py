import cv2
import numpy as np

# Starts Webcam
cap = cv2.VideoCapture(0)

# Performs Blurring,Thresholding and Edge detection
while( cap.isOpened() ) :
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)
       
# Finds the object with the maximum area in the frame   
    max_area=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]
    
# Calculates Convex hull    
    hull = cv2.convexHull(cnt)
    
# Calculates the center of the object using moments
    moments = cv2.moments(cnt)
    
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
              
    centr=(cx,cy)       
    cv2.circle(img,centr,5,[0,0,255],2)       
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
    cv2.drawContours(drawing,[hull],0,(0,0,255),2) 

# Finds Approximate contours and convex hull.
# This time convex hull returns indices of cnt rather than explicit co-ordinates.            
    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)
    
# Finds convexity defects in the current frame
# cv2.pointPolygonTest determines whether the convexity defect is inside or outside the object
    try:
        defects = cv2.convexityDefects(cnt,hull)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt,centr,True)
            cv2.line(img,start,end,[0,255,0],2)
            cv2.circle(drawing,far,5,[0,0,255],-1)
 
# Prints the command in the upper-left pane of the frame
                  
            if (defects.shape[0]==4):
                cv2.putText(drawing,'Stable',(30,50), font, 1,(255,255,255),2)
            elif(defects.shape[0]==3):
                cv2.putText(drawing,'Forward',(30,50), font, 1,(255,255,255),2)
	    elif(defects.shape[0]==2):
                cv2.putText(drawing,'Backward',(30,50), font, 1,(255,255,255),2)
    except AttributeError:
        pass
               
               
    cv2.imshow('output',drawing)
    cv2.imshow('input',img)
                
    k = cv2.waitKey(10)
    
    if k == 27:
        break
