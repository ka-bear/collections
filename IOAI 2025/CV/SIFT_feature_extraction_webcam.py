'''
This notebook is me trying to cook some object tracking method using SIFT features. It doesn't reallly perform really well, but i blame the detector 
(yet i'm not gonna improve the detector because it's probably just going to prove that the real issue is my idea is bad)

still, this definitely showcases how to use SIFT, a feature extraction method. You can ctrl+F it to find it. 

Also, it's not a .ipynb bcs i can't figure out how to use webcam on ipynb 


As long as the HOG detector detects a "person" below the line, it'll keep track of SIFT features. 
but the HOG detector is q bad (it runs realtime on my computer without GPU but it's not a good model)
bounding boxes in blue, SIFT features are the green dots 
'''


import cv2 
import torch 
import torch.nn as nn 

import numpy as np 

import copy 

import matplotlib.pyplot as plt 

def age_range_and_gender_from_past_dets(past_dets): 
    return ('0-4', 'Male')



# dealing with BBOX 
def rtlb_IOU(rtlb1, rtlb2): 
    i_r = min(rtlb1[0], rtlb2[0]) 
    i_t = max(rtlb1[1], rtlb2[1]) 
    i_l = max(rtlb1[2], rtlb2[2]) 
    i_b = min(rtlb1[3], rtlb2[3]) 

    intersection = (i_r-i_l) * (i_b-i_t) 
    area1 = (rtlb1[0] - rtlb1[2]) * (rtlb1[3] - rtlb1[1]) 
    area2 = (rtlb2[0] - rtlb2[2]) * (rtlb2[3] - rtlb2[1]) 

    return intersection / (area1 + area2 - intersection)

def rtlb_to_centroid(rtlb): 
    return ( int((rtlb[0] + rtlb[2])//2), int((rtlb[1] + rtlb[3])//2) ) 

def pt_within_bbox(pt, rtlb): 
    return ( (rtlb[2] <= pt[0]) and (pt[0] <= rtlb[0]) ) and ( (rtlb[1] <= pt[1]) and (pt[1] <= rtlb[3]) )




# entrance 

def pts_to_mc(pt1, pt2): 
    dy = pt2[1] - pt1[1] 
    dx = pt2[0] - pt1[0]
    m = dy/dx 
    c = pt1[1] - m*pt1[0] # c = y - mx 
    return m, c 

class EntranceLine: 
    def __init__(self, a, b): 
        if isinstance(a, (float, int)): # m and c 
            self.m = a 
            self.c = b 
        else: # pt1 and pt2 
            m, c = pts_to_mc(a, b) 
            self.m = m 
            self.c = c 
    
    def above(self, x, y): # above means y < line_y 
        line_y = self.m*x + self.c 
        return (y < line_y) 

    def below(self, x, y): # below means y > line_y 
        line_y = self.m*x + self.c 
        return (y > line_y) 

    def entered(self, entrance_condition, x, y): 
        #print("IN:", x, y) 
        #print("SELF: y={:.3f}x+{:.3f}".format(self.m, self.c)) 
        #print("Y:", self.m*x + self.c)
        if entrance_condition == EntranceCondition.BELOW: 
            return self.below(x, y) 
        elif entrance_condition == EntranceCondition.ABOVE: 
            return self.above(x, y) 
        raise ValueError("entrance_condition must be either EntranceCondition.BELOW or EntranceCondition.ABOVE") 
        

''' # attempt at making EntranceCondition.ABOVE's type EntranceCondition: 
class EntranceCondition: 
    def __init__(self, above): 
        self.above = above 

    BELOW = exec('EntranceCondition(0)') 
    ABOVE = exec('EntranceCondition(1)') 
'''

class EntranceCondition: 
    BELOW = 0 
    ABOVE = 1 



# gaussian kernel 
# taken from https://stackoverflow.com/a/43346070 
def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


# this is the part where i was cooking, this part doens't use SIFT though 

# try to fit a rectangle to some of the points, while minimizing its size 
# to do this, we have 4 parameters - xmin xmax ymin yax 
# we want to get dL/d[param], and it ideally is continuous so that dL/d[param] can be gotten 
# --> let's give full reward for all points fully within, then exponentially decreasing reward 
# for those outside (clipped 2d gaussian!! or maybe inverse square decay) 
# + we additionally add a factor of the box size to the loss function, so that it'll want to be small. 



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

class BBOX(): 
    def __init__(self, W, H): 
        self.W = W 
        self.H = H 
        xl, xh, yl, yh = W//3, 2*W//3, H//3, 2*H//3 # initialize bbox 
        self.xl = nn.Parameter(torch.tensor(xl, dtype=torch.float32, device=device)) 
        self.xh = nn.Parameter(torch.tensor(xh, dtype=torch.float32, device=device)) 
        self.yl = nn.Parameter(torch.tensor(yl, dtype=torch.float32, device=device)) 
        self.yh = nn.Parameter(torch.tensor(yh, dtype=torch.float32, device=device)) 

        all_gkern = gkern(2*W)[:, W-H:-(W-H)] # make gaussian loss kernel 
        all_gkern /= all_gkern.max() # so that the max is 1 
        all_gkern = 1 - all_gkern # so the center is now 0 
        self.all_gkern = torch.tensor(all_gkern, device=device) # to use torch's autograd 
    
    def parameters(self): 
        params = [self.xl, self.xh, self.yl, self.yh] 
        return iter(params) 

    def loss_kernel(self): 
        # gets the loss kernel used in loss calculation 
        cx = round((self.xl.item()+self.xh.item())/2) 
        cy = round((self.yl.item()+self.yh.item())/2) 
        loss_kernel = self.all_gkern[self.W-cx:(2*self.W)-cx+1, self.H-cy:(2*self.H)-cy+1] # get section of all_gkern for loss kernel 
        loss_kernel[round(self.xl.item()):round(self.xh.item())+1, round(self.yl.item()):round(self.yh.item())+1] = 0 # if it's within, then it's fine don't care 
        
        return loss_kernel 

    def size(self): 
        return (self.xh - self.xl) * (self.yh - self.yl) 

    def rtlb(self): 
        return self.xh.item(), self.yl.item(), self.xl.item(), self.yh.item() 


def coords_to_bbox(coords, WH=(800,600), in_frac=0.95, size_penalty=1, lr=0.0001, 
                   max_iters=1000, loss_cutoff=200, out_format='rtlb', return_conf=True): 
    assert out_format=='rtlb', "Out format of "+out_format+" has not been implemented yet. "
    #assert return_conf==False, "Confidence measure not implemented yet, can't return_conf" 

    W, H = WH 
    assert W>H, "W must be more than H for the current implementation. "

    #print(coords) 


    bbox_model = BBOX(W, H) 
    optimizer = torch.optim.Adam(bbox_model.parameters(), lr=lr)
    

    for iter in range(max_iters): 
        optimizer.zero_grad() 

        # "train" bbox to minimize loss 
        loss_kernel = bbox_model.loss_kernel() 

        # first part of loss term: trying to include points. 
        # 1/in_frac is inspired by 1/beta in eq (3) of https://arxiv.org/abs/2002.09594v2 
        L = 0 
        for pt in coords: 
            L += (1/in_frac)*( loss_kernel[round(pt[0]), round(pt[1])].item() ) 
        L /= len(coords) 

        # second part of loss term: penalizing size 
        L += size_penalty * bbox_model.size() 


        if (L <= loss_cutoff): break 


        L.backward() 
        optimizer.step() 



    bbox = bbox_model.rtlb() 

    if return_conf: 
        return bbox, 1.0 # confidence 
    return bbox




# image camera size: 640 x 480 

# detection & tracking init 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

sift = cv2.SIFT_create()
bf_matcher = cv2.BFMatcher() 

det_context = (15, 15, 15, 15) # in rtlb 
# at every detection, tries to get more pixels around it 


# params 
sift_to_bbox_thres = 0.9 
kp_match_in_frac = 0.95
kp_match_bbox_size_penalty = 1 
kp_match_lr=0.0001
kp_match_max_iters=1000
kp_match_loss_cutoff=200
repeat_tracked_IOU_thres = 0.5 
num_unfound_to_del = 5 

# define entrance line 
entranceline_poss = [(0,240) , (640, 240) ] 
entrance_line = EntranceLine( *entranceline_poss ) 
# y = mx+c; this is m and c 
entrance_condition = EntranceCondition.ABOVE # lower y than line 



# visualization settings 
path_colour = (255,0,0) 
path_thickness = 1 
bbox_colour = (255,0,0) 
bbox_thickness = 2 
show_pred_age_gender = True 


entered_cnt = 0 
trackeds = {}
'''
FORMAT of trackeds: 
{
    {'bbox': (r,t,l,b), 'kp': kp, 'des': des, 'past_bboxes': past_bboxes, 
    'past_dets': past_detections, 'unfound': 0}
}

Each element has bbox, keypoints, descriptors, past bboxes, and past detections (to use for detecting)
unfound represents how many times it wasn't found, **in a row**. If it isn't found many times, then it's deleted. 
'''
dones = {} # dones has additional fields: 'age_range' and 'gender' 
gones = {} # same format as trackeds 


# video init 
cap = cv2.VideoCapture(0) 



while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width, channels = frame.shape 


    # match with existing objects being tracked 
    frame_kp, frame_des = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None) 
    for tidx in trackeds.keys(): 
        #print(trackeds[tidx]['des']) 
        #print(frame_des)
        matches = bf_matcher.match(trackeds[tidx]['des'], frame_des) 

        matched_kps = [ frame_kp[m.trainIdx] for m in matches ] 
        matched_dess = [ frame_des[m.trainIdx] for m in matches ] 

        coords = [ frame_kp[m.trainIdx].pt for m in matches ] 
        new_bbox, conf = coords_to_bbox(coords, (img_width, img_height), kp_match_in_frac, 
                                        kp_match_bbox_size_penalty, kp_match_lr, 
                                        kp_match_max_iters, kp_match_loss_cutoff) 

        if conf > sift_to_bbox_thres : # it seems to be found & matched 
            trackeds[tidx]['bbox'] = new_bbox 
            trackeds[tidx]['kp'] = matched_kps 
            trackeds[tidx]['des'] = np.stack(matched_dess) 
            trackeds[tidx]['past_bboxes'].append(new_bbox) 
            trackeds[tidx]['past_dets'].append(copy.deepcopy(frame[y_min:y_max, x_min:x_max])) 
            trackeds[tidx]['unfound'] = 0 
        
        else: 
            # another time, it wasn't matched 
            new_unfound = trackeds[tidx]['unfound'] + 1 
            if new_unfound >= num_unfound_to_del: # NOTE: if need to save time, can remove the > 
                gones[len(gones.keys())] = trackeds.pop(tidx) # move to gones 
    


    



    # add more detections to tracker 
    detections, _ = hog.detectMultiScale(frame) 

    for detection in detections: 
        #print("HOG DET:", detection)
        x, y, w, h = detection 

        # get min/maxes, and a bit of context around the detection 
        x_min = max(x-det_context[2], 0) 
        x_max = min(x+w+det_context[0], img_width-1) 
        y_min = max(y-det_context[1], 0) 
        y_max = min(y+h+det_context[3], img_height-1)
        rtlb = (x_max, y_min, x_min, y_max) # rtlb bounding box 

        # check if it's considered already entered 
        #print("BEFORE ETNERED")
        if entrance_line.entered(entrance_condition, *rtlb_to_centroid(rtlb)): continue # skip this 
        #print("AFER ENTERED")

        # find out if it's already tracked; if so, ignore 
        # if IOU with a tracked is above 0.5, then it's considered already tracked 
        already_tracked = False 
        for k, tracked in trackeds.items(): 
            if rtlb_IOU( rtlb , tracked['bbox']) > repeat_tracked_IOU_thres : 
                already_tracked = True 
                break 
        if already_tracked: continue 
        #print("AFTER IOU CHECK")

        # get SIFT keypoints and descriptors on Region of Interest 
        detection_roi = frame[y_min:y_max, x_min:x_max] 
        gray_detection_roi = cv2.cvtColor(detection_roi, cv2.COLOR_BGR2GRAY) 
        det_kp, det_des = sift.detectAndCompute(gray_detection_roi, None)

        # add to trackeds 
        trackeds[len(trackeds)] = {
            'bbox': rtlb, 
            'kp': det_kp, 
            'des': det_des, 
            'past_bboxes': [rtlb], 
            'past_dets': [copy.deepcopy(detection_roi)], 
            'unfound': 0, 
        } 




    # check if any tracked has entered 
    for tidx in trackeds.keys(): 
        centroid = rtlb_to_centroid(trackeds[tidx]['bbox']) 
        if entrance_line.entered(entrance_condition, *centroid): 
            # move to dones 
            didx = len(dones.keys()) 
            dones[didx] = trackeds.pop(tidx) 
            age_range, gender = age_range_and_gender_from_past_dets(dones[didx]['past_dets']) 

            # get age and gender 
            dones[didx]['age_range'] = age_range 
            dones[didx]['gender'] = gender 

            # update entered count 
            entered_cnt += 1 

    


    for tidx in trackeds.keys(): 
        tracked = trackeds[tidx] 

        bbox = tracked['bbox'] 
        centroids = [ rtlb_to_centroid(rtlb) for rtlb in tracked['past_bboxes']] # x, y 
        
        # draw path - line connecting centroids 
        for i in range(len(centroids)-1): 
            #print(centroids[i], centroids[i+1])
            cv2.line(frame, centroids[i], centroids[i+1], path_colour, thickness=path_thickness)
        
        # draw bbox 
        r, t, l, b = bbox 
        cv2.rectangle(frame, (round(l), round(t)), (round(r), round(b)), bbox_colour, bbox_thickness)

        label = str(tidx) 
        if show_pred_age_gender: 
            age_range, gender = age_range_and_gender_from_past_dets(tracked['past_dets']) 
            label = label + ": Age "+age_range+", "+gender 

        # DEBUG sift_matches_to_bbox: draw keypoints 
        for kp in tracked['kp']: 
            cv2.circle(frame, (int(kp.pt[0]), int(kp.pt[1])), 2, (0,255,0), 2)

        # draw text label 
        cv2.putText(frame, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colour, 1)
    

    # show done count 
    cv2.putText(frame, f"Entered: {entered_cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # show enter line 
    cv2.line(frame, *entranceline_poss, (255,0,0))


    # display frame 
    cv2.imshow("Tracking...", frame) 

    if cv2.waitKey(1) == ord('q'):
        break








