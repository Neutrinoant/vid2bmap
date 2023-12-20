import os
from pathlib import Path 
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def get_text_size(
    label: str,
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1
):
    # multi-line 지원하도록 확장한 cv2.getTextSize
    
    OUTLINE_FONT_THICKNESS = 3 * font_thickness
    line_heights = []
    line_widths = []

    for line in label.splitlines():
        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline
        line_heights.append(line_height)
        line_widths.append(line_width)
        
    total_line_height = sum(int(lh * line_spacing) for lh in line_heights[:-1]) + int(line_heights[-1])
    total_line_width = int(max(line_widths))
        
    return total_line_width, total_line_height


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb


def cv2_draw_bboxes(image, bboxes, color=(0,0,255), thickness=3, inplace=False):
    # bboxes: array of shpae (N,4), 
    #   N - number of bounding boxes
    #   4 - (y1,x1,y2,x2), where (x1,y1) is top-left, (x2,y2) is bottom-right 
    if inplace:
        imgcopy = image
    else:
        imgcopy = image.copy()
    for (y1,x1,y2,x2) in bboxes:
        cv2.rectangle(imgcopy, (x1,y1), (x2,y2), color=color, thickness=thickness)
    return imgcopy

def integrate_bbox_rois(image, bboxes, pad=1):
    # 이미지에서 직사각형 grid를 따라 만든 모든 bounding box를 roi로 자른다음 한 화면에 출력
    # Args:
    #   bboxes: bounding box array of shape (N,M,4), build around grid points
    #       N - number of rows of grid points
    #       M - number of columns of grid points
    #       4 - (y1,x1,y2,x2), where (x1,y1) is top-left, (x2,y2) is bottom-right 
    #   pad: pad line length
    
    PLW = pad  # pad line width
    dtype = np.uint8
    
    # x축을 따라 bbox들을 연결, 사이에 black line (pad) 를 끼움
    concat_bboxes = []
    for row in bboxes:
        ys1, _, ys2, _ = row[0]
        roi_seq = [np.zeros((ys2+1-ys1, PLW, 3), dtype=dtype)]          # vertical black line between bboxes
        for y1,x1,y2,x2 in row:
            roi_seq.append(image[y1:y2+1, x1:x2+1])  # horizontally next bbox
            roi_seq.append(np.zeros((y2+1-y1, PLW, 3), dtype=dtype))    # vertical black line between bboxes
        hstacked_roi = np.hstack(roi_seq)
        concat_bboxes.append(hstacked_roi)

    # bbox들을 넓게 펼쳐 하나의 이미지로 만듦
    WB = concat_bboxes[-1].shape[1] # 가장 긴 bbox_hstacked 의 가로길이
    padded_vstacked_rois = [np.zeros((PLW, WB, 3), dtype=dtype)]            # horizontal black line
    for hstacked_roi in concat_bboxes:
        hb, wb = hstacked_roi.shape[:2]
        padded_hstacked_roi = np.zeros((hb, WB, 3), dtype=dtype)
        lpad = (WB - wb) // 2
        padded_hstacked_roi[:,lpad:lpad+wb,:] = hstacked_roi
        padded_vstacked_rois.append(padded_hstacked_roi)
        padded_vstacked_rois.append(np.zeros((PLW, WB, 3), dtype=dtype))    # horizontal black line
    padded_vstacked_rois = np.vstack(padded_vstacked_rois)
        
    return padded_vstacked_rois


def integrate_grid_dataset(imgdir, bbox, x_range, y_range, pad=1):
    # 특정 label의 "grid dataset"에 대해, 모든 이미지를 적절한 격자 위치에 두어 한 화면에 출력
    #   grid dataset := 격자 위의 모든 위치 (i,j)에 대해 "i_j.jpg" 이미지를 포함한 데이터셋
    # Args:
    #   imgdir: grid dataset folder
    #   bbox: 격자 위치별 bounding box (y1,x1,y2,x2)
    #   x_range: j값의 범위, 끝점 미포함
    #   y_range: i값의 범위, 끝점 미포함
    #   pad: pad line length
    
    PLW = pad  # pad line width
    dtype = np.uint8
    
    # check files
    for i in range(y_range[0], y_range[1]):
        for j in range(x_range[0], x_range[1]):
            p = Path(imgdir) / f"{i}_{j}.jpg"
            assert p.exists(), f"file not found: {str(p)}"
        
    # 각 라인에서 가장 많은 h값으로 통일
    
    # x축을 따라 bbox들을 연결, 사이에 black line (pad) 를 끼움
    concat_bboxes = []
    for i in range(y_range[0], y_range[1]):
        y1,x1,y2,x2 = bbox[i,0]
        h = y2-y1+1
        
        roi_seq = [np.zeros((h, PLW, 3), dtype=dtype)]          # vertical black line between bboxes
        for j in range(x_range[0], x_range[1]):
            y1,x1,y2,x2 = bbox[i,j]
            img = cv2.imread(os.path.join(imgdir, f"{i}_{j}.jpg"))
            # img = imutils.resize(img, height=h)
            img = cv2.resize(img, (h, y2-y1+1))
            roi_seq.append(img)  # horizontally next bbox
            roi_seq.append(np.zeros((h, PLW, 3), dtype=dtype))    # vertical black line between bboxes
        hstacked_roi = np.hstack(roi_seq)
        concat_bboxes.append(hstacked_roi)

    # bbox들을 넓게 펼쳐 하나의 이미지로 만듦
    WB = concat_bboxes[-1].shape[1] # 가장 긴 bbox_hstacked 의 가로길이
    padded_vstacked_rois = [np.zeros((PLW, WB, 3), dtype=dtype)]            # horizontal black line
    for hstacked_roi in concat_bboxes:
        hb, wb = hstacked_roi.shape[:2]
        padded_hstacked_roi = np.zeros((hb, WB, 3), dtype=dtype)
        lpad = (WB - wb) // 2
        padded_hstacked_roi[:,lpad:lpad+wb,:] = hstacked_roi
        padded_vstacked_rois.append(padded_hstacked_roi)
        padded_vstacked_rois.append(np.zeros((PLW, WB, 3), dtype=dtype))    # horizontal black line
    padded_vstacked_rois = np.vstack(padded_vstacked_rois)
        
    return padded_vstacked_rois


def imdraw(img: np.ndarray, gray=True, text=None, border=False):
    # img: array of shape (H,W,C)
    npimg = img
    w,h = npimg.shape[1], npimg.shape[0]
    npimg = cv2.resize(npimg, (5*w, 5*h))
    
    dpi = 100
    w,h = npimg.shape[1]/dpi, npimg.shape[0]/dpi
    
    fig = Figure(figsize=(w,h), dpi=dpi)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    
    if border:
        bdcolor = "red"
    else:
        bdcolor = "white"
    
    ax.axis("off")
    if text:
        ax.text(npimg.shape[1]//4, 5, text, style='italic',fontweight='bold',
            bbox={'facecolor':bdcolor, 'alpha':0.8, 'pad':5})
        
    if gray:
        ax.imshow(npimg, cmap="gray")
    else:
        ax.imshow(npimg)
        
    ax.margins(0,0)
    fig.tight_layout()
    
    canvas.draw()
    buf = canvas.buffer_rgba()
    new_img = np.asarray(buf)
    
    return new_img


def imshow(img, text=None, gray=True, outfile=""):
    # img: array of shape (H,W,C)
    npimg = img
    
    dpi = 100
    w,h = npimg.shape[1]/dpi, npimg.shape[0]/dpi
    
    fig = Figure(figsize=(w,h), dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.axis("off")
    if text:
        ax.text(npimg.shape[1]//2 - 9*len(text)//2, 0, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    
    if gray:
        ax.imshow(npimg, cmap="gray")
    else:
        ax.imshow(npimg)
        
    ax.margins(0)
    fig.tight_layout()
    
    if outfile:
        fig.savefig(outfile)
    else:
        raise NotImplementedError


def show_labels(labels, bbox, image, dists=None, outfile="", noiseid=None):
    
    import torchvision.datasets as dset
    from detectionAI.config import Config as CfgAI
    
    import json
    def load_json(file):
        with open(file, "rt") as f:
            return json.load(f)
    
    # get label
    folder_dataset_reference = dset.ImageFolder(root=CfgAI.ref_dir)
    reference_dataset = load_json(CfgAI.ref_path)
    pattern_classes = reference_dataset["names"]
    
    # mark each template locs on image
    for i,j in np.ndindex(bbox.shape[:2]):
        label_id = labels[i,j]
        y1,x1,y2,x2 = bbox[i,j]
        x,y = (x1, y1)
        if noiseid is not None and label_id == noiseid:
            label = 'X'
        elif label_id > 0:    # 0 is 'default' class
            label = pattern_classes[label_id - 1]   # -1 for fetch class label
        else:
            label = "BG"
        if label == 'X':
            cv2.putText(image, label, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
        elif label not in ["BG"]:
            cv2.putText(image, label, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,255,0), thickness=1)
        if dists is not None:
            d = dists[i,j]
            cv2.putText(image, f"{d:.2f}", (int(x1),int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,255,0), thickness=1)
    if outfile:
        cv2.imwrite(outfile, image)
    else:
        cv2.imshow("labels", image)
        cv2.waitKey(0)
        

def show_labels_one_hot(one_hot_labels, bbox, image, outfile=""):
    
    import torchvision.datasets as dset
    from detectionAI.config import Config as CfgAI
    
    one_hot_labels_inv = torch.flip(one_hot_labels, dims=(0,))
    
    # get label
    hEx = lambda x: hex(x)[2:].upper() if (x>=0 and x<16) else chr(x-16+ord('G'))
    
    # mark each template locs on image
    for i,j in np.ndindex(bbox.shape[:2]):
        one_hot = one_hot_labels_inv[i,j]
        y1,x1,y2,x2 = bbox[i,j]
        x,y = (x1, y1)
        
        if torch.sum(one_hot) == 0:
            # debug: 아무것도 없으면 X (outliers 표기)
            image = add_text_to_image(image, "X", (int(x),int(y)), 1, 1, cv2.FONT_HERSHEY_SIMPLEX, (0,0,255))
        elif (torch.count_nonzero(one_hot[1:]) == 0).all():
            continue        # 배경 외의 다른 패턴이 없으면 스킵
        else:
            label = ''.join([f"{hEx(l)}" if (k+1)%4>0 else f"{hEx(l)}\n" for k,l in enumerate(one_hot[1:])])    # 모든 값을 연결, (n,4) 행렬로 표현, 0(BG)과 1(BAR)는 제외
            image = add_text_to_image(image, label, (int(x),int(y)), 0.3, 1, cv2.FONT_HERSHEY_SIMPLEX, (0,255,0))
    if outfile:
        cv2.imwrite(outfile, image)
    else:
        cv2.imshow("labels", image)
        cv2.waitKey(0)
        

if __name__ == "__main__":
    
    image = cv2.imread('result.jpg')

    bboxes = np.array(
        [[10,10,50,50],
         [50,50,100,200],
         [100,200,500,400]]
    )
    image = cv2_draw_bboxes(image, bboxes)

    cv2.imshow("rect", image)
    cv2.waitKey(0)