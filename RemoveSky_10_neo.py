#coding:utf-8

# --------------------------
#   module import
# --------------------------
import cv2
import numpy as np
import sys
import os

cap = cv2.VideoCapture(0)
# org_img = cv2.imread('./Pics/kumo.jpg',cv2.IMREAD_COLOR)
# paste_img = cv2.imread('./Pics/universe.jpg',cv2.IMREAD_COLOR)
paste_img_org = cv2.imread('./Pics/universe.jpg',cv2.IMREAD_COLOR)
for rotate_num in range(2):
	paste_img_org = paste_img_org.transpose(1,0,2)[::-1]

# --------------------------
#   Macro
# --------------------------
GAUSIAN_PARA = 1.3


# --------------------------
#   Function
# --------------------------
def nothing(x):
    pass

def fld_check_make(path):
	if not os.path.exists(path):
		os.mkdir(path)
	else:
		print("already existed:  "+path)

def make_cmd_panel_window():
	cv2.namedWindow("Adjustment bars", cv2.WINDOW_NORMAL)
	cv2.createTrackbar("MIN_R", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("MIN_G", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("MIN_B", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("MAX_R", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("MAX_G", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("MAX_B", "Adjustment bars", 0, 255, nothing)
	cv2.createTrackbar("gamma_org",'Adjustment bars', 1, 150, nothing)
	cv2.createTrackbar("gamma_pst",'Adjustment bars', 1, 150, nothing)

def make_cmd_panel_window_2():
	cv2.namedWindow("Adjustment bars_2", cv2.WINDOW_NORMAL)
	cv2.createTrackbar("gaus org",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("gaus pst",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("gaus canny",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("gaus f",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("ric f",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("median f",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("byiteral f",'Adjustment bars_2',0,1,nothing)
	cv2.createTrackbar("Processor OFF/ON",'Adjustment bars_2',0,1,nothing)


def make_output_window_1():
	cv2.namedWindow('capture 8:Final putput', cv2.WINDOW_NORMAL)
	cv2.moveWindow('capture 8:Final putput', 1920, 0)
	cv2.setWindowProperty('capture 8:Final putput', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def make_output_window_2():
	cv2.namedWindow('capture 9:Final putput2', cv2.WINDOW_NORMAL)
	cv2.moveWindow('capture 9:Final putput2', 1920, 0)
	cv2.setWindowProperty('capture 9:Final putput2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
	s_time = 1
	make_cmd_panel_window()
	make_cmd_panel_window_2()
	# make_output_window_1()
	make_output_window_2()

	gamma_org = 0.5
	gamma_cvt_org = np.zeros((256,1),dtype = 'uint8')
	gamma_cvt_pst = np.zeros((256,1),dtype = 'uint8')

	# gausian_1 = input("Need **gausian filter** to orginal image?--any key:no, 1:yes>> ")
	# gausian_2 = input("Need **gausian filter** to paste image?  --any key:no, 1:yes>> ")
	# gausian_4 = input("Need **gausian filter** to canny image?  --any key:no, 1:yes>> ")
	# gausian_3 = input("Need **gausian filter** to final image?  --any key:no, 1:yes>> ")
	# ric 	  = input("Need **ricursive filter** to final image?--any key:no, 1:yes>> ")
	# med 	  = input("Need **median filter** to final image?   --any key:no, 1:yes>> ")
	# by_lite	  = input("Need **by_lite filter** to final image?  --any key:no, 1:yes>> ")
	print("****************************************************************************")
	save_times= input("Save folder number   -- press any key or number         >> ")
	print("****************************************************************************")

	while True:
		min_r = cv2.getTrackbarPos('MIN_R','Adjustment bars')
		min_g = cv2.getTrackbarPos('MIN_G','Adjustment bars')
		min_b = cv2.getTrackbarPos('MIN_B','Adjustment bars')
		max_r = cv2.getTrackbarPos('MAX_R','Adjustment bars')
		max_g = cv2.getTrackbarPos('MAX_G','Adjustment bars')
		max_b = cv2.getTrackbarPos('MAX_B','Adjustment bars')
		gamma_org = cv2.getTrackbarPos("gamma_org",'Adjustment bars')
		gamma_pst = cv2.getTrackbarPos("gamma_pst",'Adjustment bars')
		
		gausian_1 = cv2.getTrackbarPos("gaus org",'Adjustment bars_2')
		gausian_2 = cv2.getTrackbarPos("gaus pst",'Adjustment bars_2')
		gausian_4 = cv2.getTrackbarPos("gaus canny",'Adjustment bars_2')
		gausian_3 = cv2.getTrackbarPos("gaus f",'Adjustment bars_2')
		ric 	  = cv2.getTrackbarPos("ric f",'Adjustment bars_2')
		med 	  = cv2.getTrackbarPos("median f",'Adjustment bars_2')
		by_lite	  = cv2.getTrackbarPos("byiteral f",'Adjustment bars_2')
		processor = cv2.getTrackbarPos("Processor OFF/ON",'Adjustment bars_2')


		# min_r = 0
		# min_g = 119
		# min_b = 120
		# max_r = 255 #255
		# max_g = 255 #255
		# max_b = 255	#255
		# # gamma_org = 100
		# gamma_pst = 100

		cut_clr_image = np.array([[[min_b, min_g, min_r],[min_b, min_g, min_r]],[[max_b, max_g, max_r],[max_b, max_g, max_r]]], dtype = np.uint8)
		cv2.imshow('Adjustment bars', cut_clr_image)		
		
		CLR_CUT_MIN = np.array([min_b, min_g, min_r], np.uint8) # color range min gbr
		CLR_CUT_MAX = np.array([max_b, max_g, max_r], np.uint8) # color range max

		ret, org_img = cap.read() # cam cap color
		# cv2.imshow('01', cut_clr_image)

		if gamma_org == 0:
			gamma_org = 1
		if gamma_pst == 0:
			gamma_pst = 1

		for i in range(256):
			gamma_cvt_org[i][0] = 255 * (float(i)/255) ** (1.0/float(gamma_org/100.0))
			gamma_cvt_pst[i][0] = 255 * (float(i)/255) ** (1.0/float(gamma_pst/100.0))

		# resize paste image
		h=len(org_img)
		w=len(org_img[0])
		paste_img = cv2.resize(paste_img_org,(w,h))

		# cutting area
		gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
		th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		if gausian_4 == 1:
			th3 = cv2.GaussianBlur(th3, (3,3), GAUSIAN_PARA)
		th3_c = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)
		dst = cv2.addWeighted(org_img,0.8,th3_c,0.2,0)


		# trim original image
		cut_img = cv2.inRange(dst,CLR_CUT_MIN, CLR_CUT_MAX)
		sig_area_img = cv2.cvtColor(cut_img, cv2.COLOR_GRAY2BGR)
		if gausian_1 == 1:
			sig_area_img = cv2.GaussianBlur(sig_area_img, (5,5), GAUSIAN_PARA)#1.3
		masked_sig_clr_img = cv2.bitwise_and(org_img, sig_area_img)

		inv_cut_img = cv2.bitwise_not(cut_img)
		inv_sig_area_img = cv2.cvtColor(inv_cut_img, cv2.COLOR_GRAY2BGR)
		if gausian_2 == 1:
			inv_sig_area_img = cv2.GaussianBlur(inv_sig_area_img, (3,3), GAUSIAN_PARA)#1.3
		masked_inv_cut_img = cv2.bitwise_and(org_img, inv_sig_area_img) # porpose image
		gamma_bit2 = cv2.LUT(masked_inv_cut_img,gamma_cvt_org)
		masked_inv_cut_img = gamma_bit2
		
		# cv2.imshow("test",inv_sig_area_img)

		# trim paste image
		masked_replace_white = cv2.addWeighted(masked_inv_cut_img, 1, cv2.cvtColor(cut_img, cv2.COLOR_GRAY2RGB), 1, 0)
		masked_paste_img = cv2.bitwise_and(paste_img, sig_area_img)
		gamma_bit2_pst = cv2.LUT(masked_paste_img,gamma_cvt_pst)
		masked_paste_img = gamma_bit2_pst

		final_img = cv2.add(masked_inv_cut_img,masked_paste_img)
		final_img_t = final_img.transpose(1,0,2)[::-1]
		if gausian_3 == 1:
			final_img_t = cv2.GaussianBlur(final_img_t, (3,3), GAUSIAN_PARA)#1.3
		if  ric == 1:
			final_img_t = cv2.blur(final_img_t,(5,5))
		if  med == 1:
			final_img_t = cv2.medianBlur(final_img_t, ksize=5)
		if by_lite == 1:
			final_img_t = cv2.bilateralFilter(final_img_t,9,75,75)
		# cv2.imshow('capture 8:Final putput', final_img_t)
		
		if processor == 0:
			org_img_t = org_img.transpose(1,0,2)[::-1]
			cv2.imshow('capture 9:Final putput2', org_img_t)
		elif processor == 1:
			cv2.imshow('capture 9:Final putput2', final_img_t)


		key = cv2.waitKey(1)
		if key == 27: #pushing ESC key, then finish
			break
		elif key&0xff == ord('s'):
			fld_name = "./Doc_pic_"+save_times
			fld_check_make(fld_name)
			### save images ###
			# cv2.imwrite('./Pics_result_3/org_'+str(s_time)+'.png',org_img)
			# cv2.imwrite('./Pics_result_3/processed_'+str(s_time)+'.png',final_img)
			cv2.imwrite(fld_name+'/00_origin_image.png',org_img)
			cv2.imwrite(fld_name+'/01_paste_org_image.png',paste_img_org)
			cv2.imwrite(fld_name+'/02_paste_resize_image.png',paste_img)
			cv2.imwrite(fld_name+'/03_countours_accent.png',th3)
			cv2.imwrite(fld_name+'/04_cloudy_parts_gray.png',sig_area_img)
			cv2.imwrite(fld_name+'/05_cloudy_parts_color.png',masked_sig_clr_img)
			cv2.imwrite(fld_name+'/06_.png',inv_sig_area_img)
			cv2.imwrite(fld_name+'/07_.png',masked_inv_cut_img)
			cv2.imwrite(fld_name+'/08_.png',masked_paste_img)
			cv2.imwrite(fld_name+'/09_.png',final_img)
			cv2.imwrite(fld_name+'/10_.png',final_img_t)
			# cv2.imwrite('./Pics/messicolor'+str(s_time)+'.png',im)
			s_time += 1

finally:
	cap.release()
	cv2.destroyAllwindows()