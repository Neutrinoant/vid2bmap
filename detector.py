from collections.abc import Iterable
import numpy as np
import cv2

from cluster import make_cluster_indices

def FLD(img, roistart=None, roiend=None, **kwargs):
    '''
    Return numpy array of line segments, each consists of two end points of increasing order of x

    roistart, roiend: 
        leftmost-top and rightmost-bottom point of region-of-interest rectangle on image.
        if given, line detection only requested inside the rectangle (inclusive).
    '''
    # Create default Fast Line Detector class
    fld = cv2.ximgproc.createFastLineDetector(**kwargs)
    if isinstance(roistart, Iterable) and isinstance(roiend, Iterable):
        ay,ax = roistart
        by,bx = roiend
    else:
        ay,ax = (0,0)
        by,bx = img.shape - [1,1]

    cropped = img[ay:by+1, ax:bx+1, :].copy()
    grayimg = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    lines = fld.detect(grayimg)
    
    # debug: Draw lines on the image
    # line_on_image = fld.drawSegments(grayimg, lines)
    # cv2.imshow('lines', line_on_image)
    # cv2.waitKey(0)
    
    # cv2.imwrite("resline.jpg", line_on_image)

    if isinstance(lines, np.ndarray):
        # translate
        lines = np.reshape(lines, (-1,4))
        lines[:,0] += ax
        lines[:,1] += ay
        lines[:,2] += ax
        lines[:,3] += ay

        # sort each line points by x
        indices = lines[:,0] < lines[:,2]
        L1 = lines[indices,:]
        L2 = lines[np.invert(indices),:]
        L2 = np.hstack([L2[:,2:], L2[:,:2]])
        lines = np.vstack([L1, L2])

        # rearrange best grid (to [y1,x1,y2,x2] order)
        lines = lines[:,[1,0,3,2]]
    else:
        lines = np.empty((0,4), dtype=np.float32)

    return lines

def bar_segment_detection_general(img, roistart=None, roiend=None, f_minlen=0.3, thr=10):
    """bar 노트의 '일부'를 찾아줌. (이미지의 roi를 지원)

    Args:
        img (_type_): _description_
        roistart (_type_, optional): _description_. Defaults to None.
        roiend (_type_, optional): _description_. Defaults to None.
        f_minlen (float, optional): factor of minimum length of bar divided by image size. Defaults to 0.3.
        thr (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    bars = []
    min_length = int(round(f_minlen * img.shape[1]))

    # bar segment detection
    lines = FLD(img, roistart, roiend,
                    length_threshold=min_length, 
                    canny_aperture_size=7,
                    do_merge=True)
    
    # bar 노트의 특징: 정확히 가로선임
    lines = lines[np.abs(lines[:,0]-lines[:,2]) < 2]   # 좌우 높이차가 최대 2픽셀까지 허용
    mean_y = np.mean(np.array([lines[:,0], lines[:,2]]), axis=0)
    lines[:,0] = mean_y
    lines[:,2] = mean_y     # 좌우 높이차를 0으로 만듦 (좌우 y값의 평균으로)
    
    if lines.size > 0:
        
        Y = lines[:,0]
        X = np.zeros(Y.size)    # 상수좌표로 아무거나 해도 됨
        XY = np.vstack([X,Y]).T
        
        # cluster and merge neighbor lines (더 긴 쪽으로)
        indices = make_cluster_indices(XY, thr)
        for idx in np.unique(indices):
            cluster = lines[indices==idx,:]
            yd, yu = np.max(cluster[:,0]), np.min(cluster[:,0])

            if cluster.shape[0] >= 2 and (yd - yu) >= 2:
                # y좌표: 2픽셀 이상의 두꺼운 bar의 상하 y좌표의 중간
                y_mid = np.mean([yu, yd])
                
                # x좌표: 가장 오른쪽으로 먼 점과, 가장 왼쪽으로 먼 점
                xl, xr = cluster[:,1].min(), cluster[:,3].max()
                
                bar_mid = np.array([y_mid, xl, y_mid, xr], dtype=np.float32)
                bars.append(bar_mid)
            # else: # cluster.shape[0] == 1:
            #     bars.append(cluster[0])
        
    if len(bars) > 0:
        bars = np.vstack(bars)
    else:
        bars = np.empty((0,4), dtype=np.float32)
        
    return bars


if __name__ == "__main__":
    pass