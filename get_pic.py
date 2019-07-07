#coding:utf-8

# --------------------------
#   module import
# --------------------------
import cv2
import numpy as np
import time
from datetime import datetime # get current time
from datetime import date



# --------------------------
#   Macro
# --------------------------
WHITE_MIN = np.array([150, 150, 150], np.uint8) # color range min gbr
WHITE_MAX = np.array([255, 255, 255], np.uint8) # color range max
BLUE_MIN = np.array([  0,   0,   0], np.uint8) # color range min gbr
BLUE_MAX = np.array([255,   0,   0], np.uint8) # color range max
BLACK_MIN = np.array([0, 0, 0], np.uint8) # color range min gbr
BLACK_MAX = np.array([50, 50, 50], np.uint8) # color range max
GET_DATA_CYCLE = 5 #[min]

# cap = cv2.VideoCapture(0)
# org_img = cv2.imread('./Pics/kumo.jpg',cv2.IMREAD_COLOR)
paste_img = cv2.imread('./Pics/universe.jpg',cv2.IMREAD_COLOR)

cap = cv2.VideoCapture(1) # CAMERA asign

# --------------------------
#   Function
# --------------------------
def nothing(x):
    pass

try:
	s_time = 1
	# make window with variable window size
	now = datetime.now()
	now = datetime(now.year, now.month, now.day,now.hour,now.minute,now.second)

	date_time_1 = datetime(now.year, now.month, now.day,now.hour,now.minute+5,0)
	
	while True:
		rep, org_img = cap.read() # cam cap color
		now = datetime.now()
		now = datetime(now.year, now.month, now.day,now.hour,now.minute,now.second)

		if now == date_time_1:
			cv2.imwrite('./Pics_result_1027/org_'+str(now.month)+str(now.day)+str(now.hour)+"_"+str(now.minute)+"_"+str(s_time)+'.png',org_img)
			print("get pic No."+str(s_time))

			s_time += 1
			
			if now.minute+GET_DATA_CYCLE < 59:
				date_time_1 = datetime(now.year, now.month, now.day,now.hour,(now.minute+GET_DATA_CYCLE)%60,now.second)
			elif now.minute+GET_DATA_CYCLE >= 59:
				if now.hour >= 23:
					date_time_1 = datetime(now.year, now.month, now.day+1,(now.hour+1)%24,(now.minute+GET_DATA_CYCLE)%60,now.second)
				else:
					date_time_1 = datetime(now.year, now.month, now.day,(now.hour+1)%24,(now.minute+GET_DATA_CYCLE)%60,now.second)

			time.sleep(1)
					 
		cv2.imshow('capture 8:Final putput', org_img)



		key = cv2.waitKey(1)
		if key == 27: #pushing ESC key, then finish
			break
		elif key&0xff == ord('s'):
			### save images ###
			cv2.imwrite('./Pics_result_1027/org_v'+str(s_time)+'.png',org_img)
			cv2.imwrite('./Pics_result_1027/processed_'+str(s_time)+'.png',final_img)
			# cv2.imwrite('./Pics/messicolor'+str(s_time)+'.png',im)
			s_time += 1




finally:
	cap.release()
	cv2.destroyAllwindows()
