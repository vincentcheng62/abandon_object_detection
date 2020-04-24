import numpy as np
import cv2
import time, datetime
import math
import os

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x,y,w,h]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[0]+bb_test[2], bb_gt[0]+bb_gt[2])
  yy2 = np.minimum(bb_test[1]+bb_test[3], bb_gt[1]+bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2])*(bb_test[3]) + (bb_gt[2])*(bb_gt[3]) - wh)
  return(o)

def IsInside(bb_inner, bb_outer):
    if bb_outer[0] <= bb_inner[0] and bb_outer[1] <= bb_inner[1] and bb_outer[0]+bb_outer[2]>=bb_inner[0]+bb_inner[2] and bb_outer[1]+bb_outer[3]>=bb_inner[1]+bb_inner[3]:
        return True
    else:
        return False

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
#cap = cv2.VideoCapture('video1.avi')
#cap = cv2.VideoCapture('4k.mp4')
cap = cv2.VideoCapture('test2.MOV')

# define ROI
ZID_x = 1298
ZID_y = 912
ZID_w = 390
ZID_h = 233

#trackerType = "MOSSE"  #fastest
trackerType = "CSRT"  #best tracking
multiTracker = cv2.MultiTracker_create()


kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

#default is 0.003, the smaller the harder to relearn
fgbg_long = cv2.bgsegm.createBackgroundSubtractorGSOC(replaceRate=0.0003)
fgbg_short = cv2.bgsegm.createBackgroundSubtractorGSOC(replaceRate=0.012)

#fgbg = cv2.cuda.createBackgroundSubtractorMOG2()

#cuda_stream = cv2.cuda_Stream()
skip_sec = 154
current_fps_for_skip_sec = 30
learning_frame_num=500

skip_frame=skip_sec*current_fps_for_skip_sec
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame) # skip frame

resize_scale=2
counter=1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

output_dir = "./gsoc_debug_"+str(st) +"/"
os.mkdir(output_dir)

name = os.path.join(output_dir, 'display.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(name, fourcc, 30, (int(ZID_w*resize_scale*3), int(ZID_h*resize_scale*2)))

# The value between 0 and 1 that indicates how fast the background model is learnt. 
# Negative parameter value makes the algorithm to use some automatically chosen learning rate. 
# 0 means that the background model is not updated at all, 1 means that the background model is
# completely reinitialized from the last frame.
learninglong = 0.4
learningshort = 0.4
iou_threshold = 0.5
iou_threshold_for_long_overlap = 0.2
prev_frame_this_frame_iou_thrd = 0.3
obj_min_px_side = 5

prev_frame_potential_obj=[]

while(1):
    start = time.time()
    ret, frame = cap.read()

    zoom_frame = frame[ZID_y:ZID_y+ZID_h, ZID_x:ZID_x+ZID_w]
    zoom_frame_tracked = zoom_frame.copy()

    fgmask_long = fgbg_long.apply(zoom_frame, learningRate=learninglong)
    fgmask_short = fgbg_short.apply(zoom_frame, learningRate=learningshort)
    #fgmask = fgbg.apply(frame, 0.1, cuda_stream)

    fgmask_long = cv2.morphologyEx(fgmask_long, cv2.MORPH_CLOSE, kernel_close)
    fgmask_short = cv2.morphologyEx(fgmask_short, cv2.MORPH_CLOSE, kernel_close)

    #fgmask_long = cv2.morphologyEx(fgmask_long, cv2.MORPH_OPEN, kernel)
    #fgmask_short = cv2.morphologyEx(fgmask_short, cv2.MORPH_OPEN, kernel)



    image, contours_long, hierarchy =cv2.findContours(fgmask_long, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours_long : 
        cv2.fillPoly(fgmask_long, pts =[cnt], color=(255,255,255))

    
    image, contours_long, hierarchy =cv2.findContours(fgmask_long, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image, contours_short, hierarchy =cv2.findContours(fgmask_short, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours_short : 
        cv2.fillPoly(fgmask_short, pts =[cnt], color=(255,255,255))

    
    image, contours_short, hierarchy =cv2.findContours(fgmask_short, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    


    fgmask_color_long = cv2.cvtColor(fgmask_long, cv2.COLOR_GRAY2BGR)
    fgmask_color_short = cv2.cvtColor(fgmask_short, cv2.COLOR_GRAY2BGR)

    fgmask_long_minus_short = fgmask_long - fgmask_short
    fgmask_long_minus_short = cv2.morphologyEx(fgmask_long_minus_short, cv2.MORPH_OPEN, kernel_open)

    image, contours_long_minus_short, hierarchy =cv2.findContours(fgmask_long_minus_short, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours_long_minus_short : 
        cv2.fillPoly(fgmask_long_minus_short, pts =[cnt], color=(255,255,255))

    
    image, contours_long_minus_short, hierarchy =cv2.findContours(fgmask_long_minus_short, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    


    fgmask_color_long_minus_short = cv2.cvtColor(fgmask_long_minus_short, cv2.COLOR_GRAY2BGR)
    fgmask_color_long_minus_short_filtered = fgmask_color_long_minus_short.copy()
    fgmask_color_long_minus_short_filtered.fill(0)


    tracking_result, tracked_boxes = multiTracker.update(zoom_frame)
    print("tracking_result ", tracking_result)

    for i, newbox in enumerate(tracked_boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #cv2.rectangle(zoom_frame_tracked, p1, p2, colors[i% len(colors)], 2, 1)
        cv2.rectangle(zoom_frame_tracked, p1, p2, (0, 255, 0), 3)

    if counter > learning_frame_num:
        obj_id=0
        for cnt in contours_long : 
            x,y,w,h = cv2.boundingRect(cnt)
            if w > obj_min_px_side and h > obj_min_px_side:
                color = colors[int(obj_id) % len(colors)]
                cv2.drawContours(fgmask_color_long, cnt, -1, color, 1)
                obj_id=obj_id+1
                cv2.rectangle(fgmask_color_long, (x, y), (x+w, y+h), color, 1)

        obj_id=0
        for cnt in contours_short : 
            x,y,w,h = cv2.boundingRect(cnt)
            if w > obj_min_px_side and h > obj_min_px_side:
                color = colors[int(obj_id) % len(colors)]
                cv2.drawContours(fgmask_color_short, cnt, -1, color, 1)
                obj_id=obj_id+1
                cv2.rectangle(fgmask_color_short, (x, y), (x+w, y+h), color, 1)

        obj_id=0
        for cnt in contours_long_minus_short : 
            x,y,w,h = cv2.boundingRect(cnt)
            if w > obj_min_px_side and h > obj_min_px_side:
                color = colors[int(obj_id) % len(colors)]
                cv2.drawContours(fgmask_color_long_minus_short, cnt, -1, color, 1)
                obj_id=obj_id+1
                cv2.rectangle(fgmask_color_long_minus_short, (x, y), (x+w, y+h), color, 1)

        filtered_list_long = []
        for cnt in contours_long_minus_short : 
            x,y,w,h = cv2.boundingRect(cnt)
            if w > obj_min_px_side and h > obj_min_px_side:

                # only draw if the same object exist in the long bg and of similar size
                #max_iou_score = 0.0
                long_ans = None
                short_ans = None

                for long_cnt in contours_long:
                    xl,yl,wl,hl = cv2.boundingRect(long_cnt)

                    if wl > obj_min_px_side and hl > obj_min_px_side:
                        #print("long: ", str((xl,yl,wl,hl)), " short: ", str((x,y,w,h)))
                        #iou_score = iou((xl,yl,wl,hl), (x,y,w,h))
                        isinside = IsInside((x,y,w,h), (xl,yl,wl,hl))
                        #max_iou_score = max(iou_score, max_iou_score)
                        #if max_iou_score > iou_threshold_for_long_overlap:
                            #break
                        if isinside:
                            long_ans = (long_cnt, (xl,yl,wl,hl) )
                            break

                if not long_ans is None:
                    max_iou_score = 0.0
                    for short_cnt in contours_short:
                        xl,yl,wl,hl = cv2.boundingRect(short_cnt)

                        if wl > obj_min_px_side and hl > obj_min_px_side:
                            #print("long: ", str((xl,yl,wl,hl)), " short: ", str((x,y,w,h)))
                            #iou_score = iou((xl,yl,wl,hl), (x,y,w,h))
                            #isinside = IsInside((x,y,w,h), (xl,yl,wl,hl))
                            #max_iou_score = max(iou_score, max_iou_score)
                            #if max_iou_score > iou_threshold_for_long_overlap:
                                #break
                            # if isinside:
                            #     short_ans = (short_cnt, (xl,yl,wl,hl) )

                                # Final judge
                            iou_score = iou((xl,yl,wl,hl), long_ans[1])
                            max_iou_score = max(iou_score, max_iou_score)

                    if max_iou_score < iou_threshold_for_long_overlap:
                        filtered_list_long.append(long_ans[0])

                    # if short_ans is None and not long_ans is None:
                    #     filtered_list_long.append(long_cnt)           


                #if max_iou_score > iou_threshold_for_long_overlap:

        obj_id=0
        for filtered_cnt in filtered_list_long:
            x,y,w,h = cv2.boundingRect(filtered_cnt)
            color = colors[int(obj_id) % len(colors)]
            cv2.drawContours(fgmask_color_long_minus_short_filtered, filtered_cnt, -1, color, 1)
            obj_id=obj_id+1
            cv2.rectangle(fgmask_color_long_minus_short_filtered, (x, y), (x+w, y+h), color, 1)

            isPrevFrameAlsoExist=False
            if len(prev_frame_potential_obj) > 0:
                for obj in prev_frame_potential_obj:
                    xp,yp,wp,hp = cv2.boundingRect(obj)
                    iou_score = iou((xp,yp,wp,hp), (x,y,w,h))
                    if iou_score > prev_frame_this_frame_iou_thrd:
                        isPrevFrameAlsoExist = True
                        break

                if isPrevFrameAlsoExist:
                    #cv2.rectangle(zoom_frame_tracked, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    # isTracked=False
                    # for box in tracked_boxes:
                    #     iou_score = iou(box, (x,y,w,h))
                    #     if iou_score > iou_threshold:
                    #         isTracked = True
                    #         print("Object is already tracked at " + str((x,y,w,h)))
                    #         break


                    # if not isTracked:
                    #     print("Add one object to track at " + str((x,y,w,h)))
                    #     multiTracker.add(createTrackerByName(trackerType), zoom_frame, (x,y,w,h))

        
        prev_frame_potential_obj = filtered_list_long

            # Add to track if it is not yet tracked
            # if w > obj_min_px_side and h > obj_min_px_side:
            #     isTracked=False
            #     for box in boxes:
            #         iou_score = iou(box, (x,y,w,h))
            #         if iou_score> 0:
            #             print("iou_score: ", iou_score)
            #         if iou_score > iou_threshold:
            #             isTracked = True
            #             print("Object is already tracked at " + str((x,y,w,h)))
            #             break


            #     if not isTracked:
            #         print("Add one object to track at " + str((x,y,w,h)))
            #         multiTracker.add(createTrackerByName(trackerType), zoom_frame, (x,y,w,h))
            
            # else:
            #     print("object too small to be tracked: " + str((x,y,w,h)))



    ####################################################################
    ################### Display and fps countering #####################
    ####################################################################

    resized_long = cv2.resize(fgmask_color_long, None, fx=resize_scale, fy=resize_scale)
    resized_short = cv2.resize(fgmask_color_short, None, fx=resize_scale, fy=resize_scale)
    resized_long_minus_short = cv2.resize(fgmask_color_long_minus_short, None, fx=resize_scale, fy=resize_scale)
    resized_long_minus_short_filtered = cv2.resize(fgmask_color_long_minus_short_filtered, None, fx=resize_scale, fy=resize_scale)
    resized_zoom_frame = cv2.resize(zoom_frame, None, fx=resize_scale, fy=resize_scale)
    resized_zoom_frame_tracked = cv2.resize(zoom_frame_tracked, None, fx=resize_scale, fy=resize_scale)


    cv2.rectangle(resized_zoom_frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.rectangle(resized_zoom_frame_tracked, (0, 0), (400, 60), (0, 0, 0), -1)

    cv2.putText(resized_zoom_frame, "Raw color video (ROI)" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(resized_zoom_frame_tracked, "Detection result" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_long, "Long bg subtraction" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(resized_short, "Short bg subtraction" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(resized_long_minus_short, "Long minus short" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(resized_long_minus_short_filtered, "Long minus short filtered" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    time_needed = time.time() - start
    fps = 1.0/time_needed
    cv2.putText(resized_long, "fps: " + str(fps)[:5] + "  (raw video is 30fps)" , (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    timenow_min = math.floor((counter/30)/60)
    timenow_sec = math.floor((counter/30)%60)

    cv2.putText(resized_long, "Time: " + str(timenow_min) + "mins:" + str(timenow_sec) + "secs" , (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    if counter <= learning_frame_num:
        cv2.putText(resized_long, "(Learning bg.....)" , (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)    
    else:
        cv2.putText(resized_long, "(Detect and tracking.....)" , (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)    

    merged1 = np.vstack((resized_long, resized_zoom_frame))    
    merged2 = np.vstack((resized_short, resized_zoom_frame_tracked))    
    merged3 = np.vstack((resized_long_minus_short, resized_long_minus_short_filtered))    
    merged = np.hstack((merged1, merged2, merged3))    

    cv2.imshow('frame',merged)
    video_writer.write(merged)
    print("Now at frame: ", counter)
    counter=counter+1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()