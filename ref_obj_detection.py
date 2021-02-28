import numpy as np 
import cv2
import matplotlib.pyplot as plt
from imutils import perspective
from scipy.spatial.distance import euclidean

cap = cv2.VideoCapture() 
cap.open("http://192.168.43.1:8080/video")

while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()
    original=frame.copy()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#BGR to gray conversion
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)#gray to binary conversion for better contour detection
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#finding contours
    cs = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)#sorting w.r.t. area in descending order
    cs1=cs[0:1]
    cs3=cs[2:3]
    image = cv2.drawContours(frame, cs1, -1, (0, 0, 255), 2)
    cs1=np.squeeze(cs1)

    image = cv2.drawContours(image, cs3, -1, (0, 0, 255), 2)
    cs3=np.squeeze(cs3)
    
    centre_w_h_angle=cv2.minAreaRect(cs1)
    vertices=cv2.boxPoints(centre_w_h_angle)

    centre_w_h_anglex=cv2.minAreaRect(cs3)
    verticesx=cv2.boxPoints(centre_w_h_anglex)

    vertic_array = np.array(vertices, dtype="int")
    # vertic_array
    vertic_arrayx = np.array(verticesx, dtype="int")

    clk_pts=perspective.order_points(vertic_array)
    clk_ptsx=perspective.order_points(vertic_arrayx)
    tl,tr,br,bl=clk_pts
    tlx,trx,brx,blx=clk_ptsx

    cv2.drawContours(image,[vertic_array],0,(0,255,0),2)
    cv2.drawContours(image,[vertic_arrayx],0,(0,255,0),2)

    w1=euclidean(tl,tr)
    w2=euclidean(bl,br)
    h1=euclidean(tl,bl)
    h2=euclidean(tr,br)
    pixel_per_mm=(w1+w2+h1+h2)/120
    w=euclidean(tl,tr)/pixel_per_mm
    h=euclidean(tl,bl)/pixel_per_mm
    # Display the resulting frame
    # w1x=euclidean(tlx,trx)
    # w2x=euclidean(blx,brx)
    # h1x=euclidean(tlx,blx)
    # h2x=euclidean(trx,brx)

    wnew=euclidean(tlx,trx)/pixel_per_mm
    hnew=euclidean(tlx,blx)/pixel_per_mm



    wx,wy= (tl[0]+tr[0])*0.5,(tl[1]+tr[1])*0.5
    hx,hy= (tr[0]+br[0])*0.5,(tr[1]+br[1])*0.5
    cv2.putText(image, "{: .2f} mm".format(w),(int(wx - 20), int(wy - 5)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
    cv2.putText(image, "{: .2f} mm".format(h),(int(hx), int(hy)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)


    wxnew,wynew= (tlx[0]+trx[0])*0.5,(tlx[1]+trx[1])*0.5
    hxnew,hynew= (trx[0]+brx[0])*0.5,(trx[1]+brx[1])*0.5
    cv2.putText(image, "{: .2f} mm".format(wnew),(int(wxnew - 20), int(wynew - 5)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
    cv2.putText(image, "{: .2f} mm".format(hnew),(int(hxnew), int(hynew)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)


    cv2.imshow('real',original)
    cv2.imshow('contours',image)
    # plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows()