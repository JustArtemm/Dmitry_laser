import cv2
import numpy as np
import matplotlib.pyplot as plt

# init_pts = np.array([
#     [331,187],
#     [ 922, 196],
#     [220, 570],
#     [1032, 571]
# ]).astype(np.float32)
init_pts = np.array([
    [402, 210],
    [1002, 223],
    [318, 615 ],
    [1150, 613]
]).astype(np.float32)
h = 31
w = 37
ppu = 10


def process(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    red_mask = image[:,:,0] #== image[:,:,0].max()
    # red_mask = red_mask.astype(np.float32)*255

    # red_mask = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)

    # red_mask = cv2.GaussianBlur(red_mask, (31,31),0)
    # attention = np.zeros_like(red_mask)
    # attention[150:550,300:1100] = 1
    # red_mask = red_mask * attention
    ret, red_mask = cv2.threshold(red_mask,254,255,cv2.THRESH_BINARY)
    Contours = cv2.findContours(red_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = Contours[0][0].reshape(-1,2)
    center = contour.max(axis=0)/2 + contour.min(axis = 0)/2
    return red_mask, center.astype(int)

def make_transform(image, init_pts, h,w, ppu):
    
    dst_pts = np.array([
        [0,0],
        [0,w*ppu],
        [h*ppu, 0],
        [h*ppu, w*ppu]
        

    ]).astype(np.float32)
    init_pts = init_pts.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(init_pts, dst_pts)
    dist = cv2.warpPerspective(image, matrix, dsize=(ppu*h,ppu*w))
    return dist.transpose(1,0,2)

# cap = cv2.VideoCapture('/Users/artem/Desktop/Dmitry_laser/data/real_cam_ruler')
cap = cv2.VideoCapture(0)

# fig, ax = plt.subplots(1,1)

points = []


while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # recolored = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite( 'data/image.png',frame)
    # Display the resulting frame
    # cv2.imshow('image', frame)
    frame = make_transform(frame, init_pts, h, w,ppu)
    cv2.imshow('image', frame)
    try:
        mask, center = process(frame)
    except:
       mask, center = frame, None
    #print(center)
    if not center is None:
        mask = cv2.circle(mask, (center[0],center[1]), radius=10, color=(255, 0, 0), thickness=-1)
        points.append(center)
        points_narr = np.array(points)
        # plt.clean()
        plt.clf()
        plt.plot(points_narr[-100:,0],points_narr[-100:,1])
        plt.xlim(0,ppu*w)
        plt.ylim(0,ppu*h)
        # plt.scatter(center[0],center[1], s=1)
        
        
        plt.pause(0.01)
        
      
        # plt.show()
        
    cv2.imshow('mask',mask)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
plt.show()
 
# When everything done, release the video capture object
cap.release()