import numpy as np
import cv2
from matplotlib import pyplot as plt


paste_img = cv2.imread('Pics/sunny.jpg',cv2.IMREAD_COLOR)


cap = cv2.VideoCapture(1)

def nothing(x):
    pass

def imshow_fullscreen(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(winname, img)

try:
    s_time = 1
    # make window with variable window size
    cv2.namedWindow("Adjustment bars", cv2.WINDOW_NORMAL)
    # make track bar
    cv2.createTrackbar("MIN_R", "Adjustment bars", 0, 255, nothing)
    cv2.createTrackbar("MIN_G", "Adjustment bars", 0, 255, nothing)
    cv2.createTrackbar("MIN_B", "Adjustment bars", 0, 255, nothing)
    cv2.createTrackbar("MAX_R", "Adjustment bars", 0, 255, nothing)
    cv2.createTrackbar("MAX_G", "Adjustment bars", 0, 255, nothing)
    cv2.createTrackbar("MAX_B", "Adjustment bars", 0, 255, nothing)
    # cv2.createTrackbar("Canny_min",'Adjustment bars', 1, 2000, nothing)
    # cv2.createTrackbar("Canny_max",'Adjustment bars', 2, 2000, nothing)
    cv2.createTrackbar("gamma",'Adjustment bars', 1, 300, nothing)
    cv2.createTrackbar("turnR",'Adjustment bars', 0, 1, nothing)
    cv2.createTrackbar("turnL",'Adjustment bars', 0, 1, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar("switch", 'image',0,1,nothing)

    cv2.namedWindow('end', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('end', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('end', end)

    gamma = 0.5
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')

    while(True):
        s_time = 1
         # Capture frame-by-frame
        min_r = cv2.getTrackbarPos('MIN_R','Adjustment bars')
        min_g = cv2.getTrackbarPos('MIN_G','Adjustment bars')
        min_b = cv2.getTrackbarPos('MIN_B','Adjustment bars')
        max_r = cv2.getTrackbarPos('MAX_R','Adjustment bars')
        max_g = cv2.getTrackbarPos('MAX_G','Adjustment bars')
        max_b = cv2.getTrackbarPos('MAX_B','Adjustment bars')
        # canny_min = cv2.getTrackbarPos("Canny_min",'Adjustment bars')
        # canny_max = cv2.getTrackbarPos("Canny_max",'Adjustment bars')
        gamma = cv2.getTrackbarPos("gamma",'Adjustment bars')
        s_L = cv2.getTrackbarPos("turnL",'Adjustment bars',)
        s_R = cv2.getTrackbarPos("turnR",'Adjustment bars',)
        cut_clr_image = np.array([[[min_b, min_g, min_r],[min_b, min_g, min_r],[min_b, min_g, min_r]],[[max_b, max_g, max_r],[max_b, max_g, max_r],[max_b, max_g, max_r]]], dtype = np.uint8)
        cv2.imshow('Adjustment bars', cut_clr_image)

        # min_r = 120
        # min_g = 170
        # min_b = 178

        # max_r = 255
        # max_g = 255
        # max_b = 255

        for i in range(256):
            gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/float(gamma/100.0))

        CLR_CUT_MIN = np.array([min_b, min_g, min_r], np.uint8) # color range min gbr
        CLR_CUT_MAX = np.array([max_b, max_g, max_r], np.uint8)
    
        ret, frame = cap.read()

        # print((cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN))

        # frame = cv2.resize(frame,(cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN))

        h=len(frame)
        w=len(frame[0])
        paste_img = cv2.resize(paste_img,(w,h))

        cv2.imshow('frame',frame)

    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    # Display the resulting frame
        th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        # cv2.imshow('th3',th3)

        th3_c = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('th3_c',th3_c)

        dst = cv2.addWeighted(frame,0.8,th3_c,0.2,0)

        # cv2.imshow('dst',dst)

        mask = cv2.inRange(dst, CLR_CUT_MIN, CLR_CUT_MAX)
        cv2.imshow('mask',mask)
        BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #cv2.imshow('2BGR',BGR)
        bit = cv2.bitwise_and(frame, BGR)
        #cv2.imshow('bit',bit)
        gaus = cv2.GaussianBlur(BGR,  (3,3), 1.3)
        bitnot = cv2.bitwise_not(mask)
        BGR2 = cv2.cvtColor(bitnot, cv2.COLOR_GRAY2BGR)
        gaus2 = cv2.GaussianBlur(BGR2,  (3,3), 1.3)

        bit2 = cv2.bitwise_and(frame, gaus2)
        cv2.imshow('bit2__',bit2)
        maskwhite = cv2.addWeighted(bit2, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 1, 0)
        # cv2.imshow('white',maskwhite)
        paste = cv2.bitwise_and(paste_img, gaus)

        gamma_bit2 = cv2.LUT(bit2,gamma_cvt)
        cv2.imshow('gamma_bit2__',gamma_bit2)

        end = cv2.add(gamma_bit2, paste)
        end = cv2.GaussianBlur(end, (3,3), 2.6)


        if s_L == 1:
            end = end.transpose(1,0,2)[::1]
            # end = end.transpose(1,0,2)[::1]
            cv2.setWindowProperty('end', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif s_R == 1:
            end = end.transpose(1,0,2)[::-1]
            # end = end.transpose(1,0,2)[::-1]
            cv2.setWindowProperty('end', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # elif s_L == 1 and s_R == 1:
        else:
            cv2.setWindowProperty('end', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('end', end)

    
        key = cv2.waitKey(1)
        if  key& 0xFF == ord('q'):
            break
        elif key == 27: #pushing ESC key, then finish
            break
        elif key&0xff == ord('s'):
            ### save images ###
            cv2.imwrite('./Pics_result/org_'+str(s_time)+'.png',frame)
            cv2.imwrite('./Pics_result/processed_'+str(s_time)+'.png',end)
            # cv2.imwrite('./Pics/messicolor'+str(s_time)+'.png',im)
            s_time += 1
            




finally:
    cap.release()
    cv2.destroyAllwindows()