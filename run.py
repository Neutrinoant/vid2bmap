
from tqdm import tqdm
import json
import os
from pathlib import Path
import shutil

import numpy as np
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import cv2

from video import Video
from config import get_cfg_defaults  # local variable usage pattern
from detector import bar_segment_detection_general
from cluster import make_cluster_indices
from detectionAI.dataset import VideoFrameDataset
from logger import logger, checkpoint
import visualize as vis

def load_json(file):
    with open(file, "rt") as f:
        return json.load(f)
    
def save_json(file, obj):
    with open(file, "wt") as f:
        json.dump(obj, f)

def load_config(path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()
    return cfg
    
@checkpoint
def save_frame(videofile, outdir, start_idx=0, end_idx=None):
    # video에 대한 frames를 load
    video = Video(videofile)
    return video.save_frames(outdir, start_idx=start_idx, end_idx=end_idx)
    
@checkpoint
def bar_fit(frame_paths, outfile, roi, display='splash', thr_minlen=0.20, thr_cluster=10):
    # roi: [x,y,w,h]
    # display: "splash", "approach", "slope", "vertical"
    assert len(frame_paths) > 0, "bar_fit: len(frame_paths) <= 0"
    
    x,y,w,h = roi
    roistart = (y, x)
    roiend = (y+h-1, x+w-1)
    
    # 각 frames에서 bar 노트를 매칭
    lines = []
    
    dataset = VideoFrameDataset(frame_paths, gray=False)
    frame_dataloader = DataLoader(dataset, batch_size=1, num_workers=6, shuffle=False)
    
    for img in tqdm(frame_dataloader):
        img = img.squeeze().numpy()
        segs = bar_segment_detection_general(img, roistart, roiend, 
                                     f_minlen=thr_minlen,
                                     thr=thr_cluster)
        lines.append(segs)

    lines = np.vstack(lines)
    
    # 모든 frames에 대해 매칭을 통합, 각 바의 양 끝점 좌표를 겟
    bars = []
    if lines.size > 0:
        
        Y = lines[:,0]
        X = np.zeros(Y.size)    # 상수좌표로 아무거나 해도 됨
        XY = np.vstack([X,Y]).T
        
        # cluster and merge neighbor lines (더 긴 쪽으로)
        indices = make_cluster_indices(XY, thr=thr_cluster)
        for idx in np.unique(indices):
            cluster = lines[indices==idx,:]
            yd, yu = np.max(cluster[:,0]), np.min(cluster[:,0])
            
            if cluster.shape[0] >= 2:
                # y좌표: 2픽셀 이상의 두꺼운 bar의 상하 y좌표의 중간
                y_mid = np.mean([yu, yd])
                
                # x좌표: 가장 오른쪽으로 먼 점과, 가장 왼쪽으로 먼 점
                xl, xr = cluster[:,1].min(), cluster[:,3].max()
                
                bar_mid = np.array([y_mid, xl, y_mid, xr], dtype=np.float32)
                bars.append(bar_mid)
            else: # cluster.shape[0] == 1:
                bars.append(cluster[0])
                
    if len(bars) > 0:
        bars = np.array(bars)
    else:
        bars = np.empty((0,4), dtype=np.float32)

    # 바를 y좌표 기준으로 정렬, 정수화
    indices = np.argsort(bars[:,0])
    bars = bars[indices]
    bars = np.round(bars).astype(int)
    
    # 끝점의 x,y를 각각 곡선에 근사 (하강 종류에 따라 달리 적용)
    
    def arbitrary_horizontal(x,a):
        return a

    def arbitrary_linear(x,a,b):
        return a*(x) + b

    def arbitrary_quadratic(x,a,b,c):
        return a*(x**2) + b*(x) + c

    XL = bars[:,1]
    XR = bars[:,3]
    Y = bars[:,0]
    
    if display in ["splash", "slope"]:
        arbitrary_curve = arbitrary_quadratic
    elif display == "approach":
        arbitrary_curve = arbitrary_linear
    elif display == "vertical":
        arbitrary_curve = arbitrary_horizontal
    else:
        raise NotImplementedError
        
    poptl, _ = curve_fit(arbitrary_curve, Y, XL)
    bars[:,1] = arbitrary_curve(bars[:,0], *poptl)
    poptr, _ = curve_fit(arbitrary_curve, Y, XR)
    bars[:,3] = arbitrary_curve(bars[:,0], *poptr)
    
    np.save(outfile, bars)
    
@checkpoint
def grid_from_bar(barfile, outfile, rm_factor=(6/1760/2), n_routes=28):
    
    bars = np.load(barfile)
    
    # bar의 끝부분 약간 제거
    #   x=80 ~ x=1839+1 위치의 bar 기준 양쪽 6씩 == 12/1760
    #   즉, 양쪽 각각 전체길이의 6/1760 정도 제거 필요
    bar_lens = bars[:,3] - bars[:,1]
    madi_end_lens = np.round(bar_lens * rm_factor).astype(int)
    bars[:,1] += madi_end_lens
    bars[:,3] -= madi_end_lens
    
    # 끝점 좌표들을 28개로 나누고 노트가 위치할 수 있는 좌표들을 겟 (정수좌표)
    grid = np.linspace(bars[:,:2], bars[:,2:], 2*n_routes+1)[1::2]    # 점갯수 28+29(건반중심), 홀수번 인덱스만 추출
    grid = np.transpose(grid, [1,0,2])
    np.save(outfile, np.round(grid).astype(np.uint32))
    
@checkpoint
def save_grid_bbox(gridfile, outfile, n_routes=28, xh_factor=1/2, yh_factor=1/3):
    
    grid = np.load(gridfile)
    
    # bounding box 만들기 (var: bbox)
    H,W = grid.shape[:2]
    bbox = np.zeros((H,W,4), dtype=np.uint32)
    for i,row in enumerate(grid):
        xl = np.min(row[:,1])
        xr = np.max(row[:,1])
        unit_x = (xr - xl) / n_routes
        # bbox_h = unit_x // 6           # 노스텔 일반노트 패턴 픽셀수 분석해서 적당히 근사한 결과
        bbox_xh = np.round(unit_x * xh_factor).astype(int)           # 노스텔 일반노트 패턴 픽셀수 분석해서 적당히 근사한 결과
        bbox_yh = np.round(unit_x * yh_factor).astype(int)           # 노스텔 일반노트 패턴 픽셀수 분석해서 적당히 근사한 결과
        for j,(y,x) in enumerate(row):
            bbox[i,j,:] = (y - bbox_yh, x - bbox_xh, y + bbox_yh, x + bbox_xh)
    
    np.save(outfile, bbox)  # int인 bbox를 저장함

@checkpoint
def label_bbox_frames_fast(frame_paths, bboxfile, outfile):
    
    import torch
    import torchvision.transforms as transforms
    from torch.autograd import Variable
    import torchvision.datasets as dset
    from torch.utils.data import DataLoader
    
    from detectionAI.model import SiameseNetwork
    from detectionAI.dataset import VideoBBoxDataset, LabeledDataset
    
    from detectionAI.config import Config as CfgAI
    CfgAI.seed_everything(CfgAI.seed)
    
    logger.info(f"run on device {CfgAI.device}")
    
    bbox = np.load(bboxfile)
    H_G, W_G = bbox.shape[:2]

    # model setting 
    checkpoint = torch.load(CfgAI.checkpoint_path, map_location=CfgAI.device)
    model_state_dict = checkpoint["model_state_dict"]
    
    net = SiameseNetwork(imgsize=CfgAI.imgsize)
    net.load_state_dict(model_state_dict)
    net = net.to(CfgAI.device)
    
    net.eval()  # BatchNorm() 레이어를 끔
    
    # ready for video dataset
    rois_dataset = VideoBBoxDataset(frame_paths, bbox, out_shape=CfgAI.imgsize, should_invert=False, gray=True)
    rois_dataloader = DataLoader(rois_dataset, num_workers=CfgAI.num_workers, batch_size=1, shuffle=False)
    
    # ready for labeling dataset
    # folder_dataset_reference = dset.ImageFolder(root=CfgAI.ref_dir)
    # reference_dataset = LabeledDataset(imageFolderDataset=folder_dataset_reference,
    #                                         transform=transforms.Compose([transforms.Resize(CfgAI.imgsize), transforms.ToTensor()]),
    #                                         should_invert=False,
    #                                         gray=True)
    # reference_dataloader = DataLoader(reference_dataset, num_workers=0, batch_size=CfgAI.train_data_ref_batch_size, shuffle=False)
    reference_dataset = load_json(CfgAI.ref_path)
        
    with torch.no_grad():   # gradient 계산 메모리 사용을 꺼버림
        
        # # ref dataset forward
        # Y_ref_arr = []
        # L_ref_arr = []
    
        # for reference_data in reference_dataloader:
        #     X_reference, L_reference = reference_data
        #     X_reference = Variable(X_reference).to(CfgAI.device)
        #     Y_reference = net.forward_once(X_reference)
        #     Y_ref_arr.append(Y_reference)
        #     L_ref_arr.append(L_reference)
        
        Y_ref_arr = torch.tensor(reference_dataset["centroids_y"]).to(CfgAI.device)
        L_ref_arr = torch.tensor(reference_dataset["ids"]).to(CfgAI.device)
        
        # Y_ref_arr = torch.concat(Y_ref_arr, dim=0).to(CfgAI.device)
        # L_ref_arr = torch.concat(L_ref_arr, dim=0).to(CfgAI.device)
        increase_idx = torch.argsort(L_ref_arr)
        Y_label = Y_ref_arr[increase_idx]   # shape: (L,10), L: number of labels
        
        assert Y_label.shape == Y_ref_arr.shape, "label_bbox_frames_fast: all labels are unique"     # 모든 label은 1개씩만 존재해야함
        
        label_arr = []
        
        for img_tensors in tqdm(rois_dataloader):
            # test dataset forward
            B,G,H,W = img_tensors.shape[:4]
            img_tensors = img_tensors.view(-1, *img_tensors.shape[2:])  # (B,G,H,W) -> (B*G,H,W)
            img_tensors = Variable(img_tensors).to(CfgAI.device)
            Y_test_arr = net.forward_once(img_tensors)
            
            # choose label
            Y_test_arr = Y_test_arr.view(-1,1,*Y_test_arr.shape[1:])     # shape: (B*G,1,10)
            D = torch.sum((Y_test_arr - Y_label)**2, dim=2)**0.5        # shape: (B*G,L)
                
            Id_min = torch.argmin(D, dim=1)
            D_min = D[np.arange(D.shape[0]),Id_min]
            Id_filtered = torch.where(D_min < CfgAI.thr_max_dist, Id_min+1, 0)    # add 1 to keep 0 as 'default' class
            
            label_ids = Id_filtered.view(B, H_G, W_G)
            label_arr.append(label_ids)
            
        label_arr = torch.concat(label_arr, dim=0)
        label_arr = label_arr.detach().cpu().numpy()
    
    np.save(outfile, label_arr)
    
@checkpoint
def graph_partition(labelfile, outfile, thr_dist=1):
    
    import torch
    from torch.nn.functional import one_hot
    import torchvision.datasets as dset
    from detectionAI.config import Config as CfgAI
    from sklearn.cluster import DBSCAN
    
    # clustering 준비
    db = DBSCAN(eps=thr_dist, min_samples=1, metric='euclidean')
    
    labels = np.load(labelfile)   # (F, G_H, G_W)

    F, G_H, G_W = labels.shape[:3]
    
    # L = len(dset.ImageFolder(root=CfgAI.ref_dir).classes)   # 배경 제외
    L = len(load_json(CfgAI.ref_path)["ids"])   # 배경 제외
    
    # 각 프레임별 라벨행렬을 one-hot으로 변경
    labels_inv = np.flip(labels, 1).copy()       # 행을 뒤집어 맨 아래가 0번째가 되도록 함
    labels_inv = torch.from_numpy(labels_inv)
    onehots = one_hot(labels_inv, num_classes=L+1)    # (F, G_H, G_W, L+1)
    
    # 각 행이 [frame, grid_h, route, label] 인 (P,4) 행렬 추출 (배경제외 라벨이 발견된 위치)
    table_fhrl = torch.argwhere(onehots[:,:,:,1:] > 0)
    
    P = table_fhrl.shape[0]
    
    # 각 행이 [route, time, label, frame] 인 (P,4) 행렬 추출
    table_rtlf = torch.zeros((P, 4), dtype=table_fhrl.dtype)
    table_rtlf[:,[0,2,3]] = table_fhrl[:,[2,3,0]]
    table_rtlf[:,1] = table_fhrl[:,0] + table_fhrl[:,1]
    
    # 각 행이 cluster 번호인 (P,) 행렬 추출 (클러스터는 인덱스 0~(C-1) 까지)
    l_scale = 5
    r_scale = L * l_scale
    db_input_Y = table_rtlf[:,1]
    db_input_X = table_rtlf[:,0] * r_scale + table_rtlf[:,2] * l_scale
    db_input = torch.hstack([db_input_Y[:,None], db_input_X[:,None]])
    table_c = torch.tensor(db.fit_predict(db_input))
    
    clusters = torch.unique(table_c)    # sorted increasing order
    C = clusters.shape[0]
    assert (clusters == torch.arange(0,C)).all(), "frame_displacement: missing cluster if occur"
    
    # 클러스터와 프레임을 연결해 connected graph(s)를 형성,
    # 각 행이 graph 번호인 (P,) 행렬 추출
    map_cluster_frame = []
    table_cluster_num_frames = torch.zeros((C,), dtype=torch.int64)     # 클러스터당 프레임 개수
    for i in range(C):
        mask = (table_c == i)
        frame_ids = torch.unique(table_rtlf[mask, 3])
        frames = set(frame_ids.detach().cpu().tolist())
        map_cluster_frame.append(frames)
        table_cluster_num_frames[i] = len(frames)
        
    map_frame_cluster = []
    for i in range(F):
        mask = (table_rtlf[:,3] == i)
        cluster_ids = torch.unique(table_c[mask])
        map_frame_cluster.append(set(cluster_ids.detach().cpu().tolist()))

    table_g = torch.zeros((P,), dtype=torch.int64)
    cluster_set_pool = set(range(C))
    graph_id = 0
    
    while True:

        if not cluster_set_pool:
            break
        
        # 초기 클러스터 번호 선택
        cluster_set_queue = set([list(cluster_set_pool)[0]])
        cluster_set_used = set()
        frame_set_queue = set()
        frame_set_used = set()
        mask = torch.zeros((P,), dtype=torch.bool)
        
        # subgraph 찾기
        while True:
            
            if not cluster_set_queue:
                break
            
            for cid in cluster_set_queue:
                frame_set_queue = frame_set_queue | map_cluster_frame[cid]
                mask = mask | (table_c == cid)
                
            cluster_set_used = cluster_set_used | cluster_set_queue
            frame_set_queue = frame_set_queue - frame_set_used
            
            if not frame_set_queue:
                break
            
            for fid in frame_set_queue:
                cluster_set_queue = cluster_set_queue | map_frame_cluster[fid]
                
            frame_set_used = frame_set_used | frame_set_queue
            cluster_set_queue = cluster_set_queue - cluster_set_used
            
        cluster_set_pool = cluster_set_pool - cluster_set_used
            
        # table에 subgraph 번호 부여
        table_g[mask] = graph_id
        graph_id += 1
        
    # 그래프 및 주요정보 저장
    table_cftg = torch.hstack((table_c[:,None], table_rtlf[:,[3,1]], table_g[:,None]))
    np.save(outfile, table_cftg.detach().cpu().numpy())

@checkpoint
def tag_graph_cluster_outliers(cftgfile, outfile, n_workers):
    """그래프에서 outlier 후보 클러스터를 추출,
        후보들을 다시 connected subgraphs 로 partition,
        각 subgraph 에서 vertex covering 으로 outlier 클러스터를 확정
    """
    from typing import List
    from copy import deepcopy
    from itertools import combinations
    
    import torch
    
    table_cftg = torch.from_numpy(np.load(cftgfile))
    table_c = table_cftg[:,0]
    table_f = table_cftg[:,1]
    table_t = table_cftg[:,2]
    table_g = table_cftg[:,3]
    
    P = table_cftg.shape[0]
    C = table_cftg[:,0].max() + 1
    G = table_cftg[:,3].max() + 1
    
    map_cluster_frame = [set() for _ in range(C)]   # 클러스터별 포함된 프레임들
    map_cluster_fo = [list() for _ in range(C)] # 클러스터별 포함된 프레임들과 그 상대적위치
    
    table_o = torch.zeros((P,), dtype=torch.int64)  # 클러스터내 프레임별 상대적 거리

    for i in range(C):
        indices = torch.argwhere(table_c == i).squeeze()
        
        # 클러스터별 프레임 set을 매핑
        frame_ids = torch.unique(table_f[indices])
        frames = set(frame_ids.detach().cpu().tolist())
        map_cluster_frame[i] = frames
        
        # 각 행이 rel_loc 인 (P,) 행렬 추출
        min_time = torch.min(table_t[indices])
        table_o[indices] = table_t[indices] - min_time
        
        # 클러스터별 (프레임,상대적위치) 쌍을 매핑 (동일프레임 중복포함도 가능)
        #   이때 행은 프레임기준 오름차순 정렬된 순서임
        fo = torch.vstack([table_f[indices], table_o[indices]]).T
        sorted_indices = torch.argsort(fo[:,0])
        map_cluster_fo[i] = fo[sorted_indices]
        
    all_outlier_clusters = []
        
    # subtable에 대해 (클러스터 개수 C')
    #   클러스터 내 프레임번호가 동일선상에 있도록 클러스터를 배열

    logger.debug(f"  n_graphs: {G}")
    
    for g in range(G):
        # subtable 추출
        mask = (table_g == g)
        table_f_sub = table_f[mask]
        table_c_sub = table_c[mask]
        
        clusters_sub: torch.Tensor = torch.unique(table_c_sub)
        
        ### 1. naive outlier: 클러스터 내 동일 프레임이 2개 이상이면 outlier 목록에 추가
        cfs, counts = table_cftg[mask][:,[0,1]].unique(return_counts=True, dim=0)
        if (counts >= 2).any():
            clusters_sub_bad = cfs[counts >= 2][:,0].unique()
            uniques, counts = torch.hstack([clusters_sub, clusters_sub_bad]).unique(return_counts=True)
        else:
            clusters_sub_bad = torch.empty((0,))
            
        all_outlier_clusters += clusters_sub_bad.tolist()
        
        if clusters_sub.shape[0] <= 1:
            continue
        
        ### 2. contextual outlier: 다른 클러스터와의 연결관계가 나쁘면 outlier clusters 목록에 추가
        
        indices_pair = []
        frame_loader = DataLoader(table_f_sub.unique(), num_workers=n_workers, batch_size=1)
        
        # 프레임별 클러스터를 전부 모으고 pair로 짝지음 (bad cluster 포함)
        for f in frame_loader:
            clusters_sub_f = set(table_c_sub[table_f_sub == f.flatten()].tolist())
            pairs = [[c1,c2] for c1,c2 in combinations(clusters_sub_f, r=2)]
            indices_pair += pairs
        
        # 두번 이상 겹치는 클러스터 조합만 추출
        uniques, counts = torch.tensor(indices_pair).unique(dim=0, return_counts=True)
        cluster_pairs_raw = uniques[counts >= 2]
        
        # bad cluster 를 제거
        mask_c_good = torch.ones(cluster_pairs_raw.shape[0], dtype=torch.bool)
        for c in clusters_sub_bad:
            mask_c_good &= ~torch.any(cluster_pairs_raw == c, dim=1)
        cluster_pairs_good = cluster_pairs_raw[mask_c_good]
        
        if cluster_pairs_good.shape[0] == 0:
            continue
        
        edges = []
        cluster_pair_loader = DataLoader(cluster_pairs_good, num_workers=n_workers, batch_size=1)
        for pair in cluster_pair_loader:
            i,j = pair.flatten().tolist()
            
            # 겹치는 프레임(개수>=2) 추출
            intersection = torch.tensor(list(map_cluster_frame[i] & map_cluster_frame[j]))
            # assert intersection.numel() >= 2
            
            # 프레임간의 순서를 프레임오름차순으로 각각 추출
            sorted_fo1 = map_cluster_fo[i]
            sorted_fo2 = map_cluster_fo[j]
            indices1 = torch.searchsorted(sorted_fo1[:,0].contiguous(), intersection)
            indices2 = torch.searchsorted(sorted_fo2[:,0].contiguous(), intersection)
            inter_loc1 = sorted_fo1[indices1,1]
            inter_loc2 = sorted_fo2[indices2,1]
            
            # 상대순서로 변환
            inter_loc1 -= inter_loc1.min()
            inter_loc2 -= inter_loc2.min()
            
            # 상대순서가 서로 다를 경우 둘다 outlier 후보가 됨
            if (inter_loc1 != inter_loc2).any():
                edges.append([i, j])
        
        if len(edges) == 0:
            continue    # 발견된 outlier 없음
        
        table_edges = torch.tensor(edges)
        outlier_cluster_candits = set(table_edges.flatten().tolist())
            
        # 모아둔 edge 의 연결관계를 통해 subgraph로 분리
        cur_g = 1
        table_cluster_subgraph = torch.tensor([[c,0] for c in outlier_cluster_candits]) # (C_outlier_candits,2)
        
        for c1,c2 in table_edges:
            indices_c1 = torch.argwhere(table_cluster_subgraph[:,0] == c1).squeeze()
            indices_c2 = torch.argwhere(table_cluster_subgraph[:,0] == c2).squeeze()
            g1 = table_cluster_subgraph[indices_c1,1]
            g2 = table_cluster_subgraph[indices_c2,1]
            if g1 == 0 and g2 == 0:
                table_cluster_subgraph[indices_c1,1] = cur_g
                table_cluster_subgraph[indices_c2,1] = cur_g
                cur_g += 1
            elif g1 == 0:
                table_cluster_subgraph[torch.argwhere(table_cluster_subgraph[:,1] == g1).squeeze(), 1] = g2 # 모든 g1을 g2로 변경
            else:   # g2 == 0 or 둘다 != 0
                table_cluster_subgraph[torch.argwhere(table_cluster_subgraph[:,1] == g2).squeeze(), 1] = g1 # 모든 g2를 g1으로 변경
                
        subgraph_ids = table_cluster_subgraph[:,1].unique()
        
        outlier_clusters = []
        
        # 각 subgraph에 대해
        for sgid in subgraph_ids:
            
            clusters_sg = table_cluster_subgraph[table_cluster_subgraph[:,1] == sgid, 0]
            mask_sg = torch.zeros(table_edges.shape, dtype=torch.bool)
            for c in clusters_sg:
                mask_sg |= (table_edges == c)
            assert (torch.sum(mask_sg, dim=1) != 1).all(), f"tag_graph_cluster_outliers: {table_edges.tolist()}"
            mask_sg = torch.all(mask_sg, dim=1)
            table_edges_sg = table_edges[mask_sg]
            
            V = clusters_sg.shape[0]
            E = table_edges_sg.shape[0]
        
            ### minimum vertex covering 을 찾음 (백트래킹)
            # node = [vertices, mask]
            #   vertices - 포함시킨 vertex set
            #   mask - 남은 edge를 추출하는 마스크 of shape (E,)
            
            solution_node = []  # solution은 최소 covering중 가장 먼저 발견한 1개만 허용
            
            ### method 1: brute force
            # global_min_cover_len = clusters_sg.shape[0]
            # stack: List[List[List[int],torch.Tensor]] = [[list(),torch.ones(E, dtype=torch.bool)]]     # initial node
            # while stack:
            #     node = stack.pop()
            #     prev_vertices, prev_mask = node
                
            #     if table_edges_sg[prev_mask].numel() == 0:
            #         # 남은 edge가 없으면 종료
            #         if len(prev_vertices) < global_min_cover_len:
            #             global_min_cover_len = len(prev_vertices)
            #             solution_node = node    # 솔루션을 업데이트
            #         # continue
            #         break   # greedy: 가장 많은 불일치를 일으키는 순으로 cluster를 최소한으로 꺼내어 불일치를 해소시킬 수 있으면, 그게 outliers
                
            #     elif len(prev_vertices)+1 >= global_min_cover_len:
            #         # 이전까지의 solution보다 더 작은 covering을 만들 수 없다면 가지치기
            #         continue
                
            #     # 남은 edge와 vertex를 겟
            #     rem_edges = table_edges_sg[prev_mask]
            #     uniques, counts = rem_edges.flatten().unique(return_counts=True)
            #     rem_vertices = uniques[torch.argsort(counts, descending=True)]  # 가장 많은 vertex순으로 정렬 (속도향상)
                
            #     # 남은 vertex에 대해
            #     candits = []
            #     for v in rem_vertices:
            #         new_vertices, new_mask = deepcopy(node)
                    
            #         # v를 포함하는 edge를 전부 제거하도록 mask 추가
            #         new_mask &= ~torch.any(table_edges_sg == v, dim=1)
            #         new_vertices.append(v.item())
                    
            #         candits.append([new_vertices, new_mask])
                
            #     stack += candits[::-1]  # 가장 먼저 찾은것부터 추가
                    
            # solution_vertices, _ = solution_node
            
            
            ### method 2: greedy
            prev_vertices = []
            prev_mask = torch.ones(E, dtype=torch.bool)
            while True:
                
                rem_edges = table_edges_sg[prev_mask]
                
                # 남은 edge가 없으면 종료
                if rem_edges.numel() == 0:
                    solution_node = [prev_vertices, prev_mask]    # 솔루션을 업데이트
                    break   # greedy: 가장 많은 불일치를 일으키는 순으로 cluster를 최소한으로 꺼내어 불일치를 해소시킬 수 있으면, 그게 outliers
                
                # 남은 vertices를 높은 degree 순으로 정렬
                uniques, counts = rem_edges.flatten().unique(return_counts=True)
                rem_vertices = uniques[torch.argsort(counts, descending=True)]
                
                # 다음 v를 포함하는 edge를 전부 제거하도록 mask 추가
                v = rem_vertices[0]
                prev_mask &= ~torch.any(table_edges_sg == v, dim=1)
                prev_vertices.append(v.item())
                    
            solution_vertices, _ = solution_node
            
            # 그걸 outlier clusters 목록에 추가
            outlier_clusters += solution_vertices
            
        all_outlier_clusters += outlier_clusters
    
    # outlier 에 tag 붙임
    all_outlier_clusters = torch.tensor(all_outlier_clusters)
    table_cftg_tag = table_cftg.clone()
    for c in all_outlier_clusters:
        mask = (table_c == c)
        table_cftg_tag[mask,0] = -table_cftg_tag[mask,0] - 1    # outlier는 부호를 반대로 - 1 (0때문에)
         
    # 새 그래프 정보를 저장
    np.save(outfile, table_cftg_tag.numpy())

@checkpoint
def graph_partition_with_tag(labelfile, cftgfile, outfile, thr_dist=1):
    
    import torch
    from torch.nn.functional import one_hot
    import torchvision.datasets as dset
    from detectionAI.config import Config as CfgAI
    from sklearn.cluster import DBSCAN
    
    # clustering 준비
    db = DBSCAN(eps=thr_dist, min_samples=1, metric='euclidean')
    
    labels = np.load(labelfile)   # (F, G_H, G_W)

    F, G_H, G_W = labels.shape[:3]
    
    # L = len(dset.ImageFolder(root=CfgAI.ref_dir).classes)   # 배경 제외
    L = len(load_json(CfgAI.ref_path)["ids"])   # 배경 제외
    
    # 각 프레임별 라벨행렬을 one-hot으로 변경
    labels_inv = np.flip(labels, 1).copy()       # 행을 뒤집어 맨 아래가 0번째가 되도록 함
    labels_inv = torch.from_numpy(labels_inv)
    onehots = one_hot(labels_inv, num_classes=L+1)    # (F, G_H, G_W, L+1)
    
    # 각 행이 [frame, grid_h, route, label] 인 (P_raw,4) 행렬 추출 (배경제외 라벨이 발견된 위치)
    table_fhrl = torch.argwhere(onehots[:,:,:,1:] > 0)
    
    # 각 행이 [route, time, label, frame] 인 (P_raw,4) 행렬 추출
    table_rtlf = table_fhrl[:,[2,1,3,0]]
    table_rtlf[:,1] += table_fhrl[:,0]
    
    # outlier 를 전부 제거
    table_cftg_tag = torch.from_numpy(np.load(cftgfile))
    outlier_mask = (table_cftg_tag[:,0] < 0)
    table_rtlf_good = table_rtlf[~outlier_mask]
    
    P = table_rtlf_good.shape[0]     # outlier 가 없는 테이블의 크기
    
    # 각 행이 cluster 번호인 (P,) 행렬 추출 (클러스터는 인덱스 0~(C-1) 까지)
    l_scale = 5
    r_scale = L * l_scale
    db_input_Y = table_rtlf_good[:,1]
    db_input_X = table_rtlf_good[:,0] * r_scale + table_rtlf_good[:,2] * l_scale
    db_input = torch.hstack([db_input_Y[:,None], db_input_X[:,None]])
    table_c = torch.tensor(db.fit_predict(db_input))
    
    clusters = torch.unique(table_c)    # sorted increasing order
    C = clusters.shape[0]
    assert (clusters == torch.arange(0,C)).all(), "frame_displacement: missing cluster if occur"
    
    # 클러스터와 프레임을 연결해 connected graph(s)를 형성,
    # 각 행이 graph 번호인 (P,) 행렬 추출
    map_cluster_frame = []
    for i in range(C):
        mask = (table_c == i)
        frame_ids = torch.unique(table_rtlf_good[mask, 3])
        frames = set(frame_ids.detach().cpu().tolist())
        map_cluster_frame.append(frames)
        
    map_frame_cluster = []
    for i in range(F):
        mask = (table_rtlf_good[:,3] == i)
        cluster_ids = torch.unique(table_c[mask])
        map_frame_cluster.append(set(cluster_ids.detach().cpu().tolist()))

    table_g = torch.zeros((P,), dtype=torch.int64)
    cluster_set_pool = set(range(C))
    graph_id = 0
    
    while True:

        if not cluster_set_pool:
            break
        
        # 초기 클러스터 번호 선택
        cluster_set_queue = set([list(cluster_set_pool)[0]])
        cluster_set_used = set()
        frame_set_queue = set()
        frame_set_used = set()
        mask = torch.zeros((P,), dtype=torch.bool)
        
        # subgraph 찾기
        while True:
            
            if not cluster_set_queue:
                break
            
            for cid in cluster_set_queue:
                frame_set_queue = frame_set_queue | map_cluster_frame[cid]
                mask = mask | (table_c == cid)
                
            cluster_set_used = cluster_set_used | cluster_set_queue
            frame_set_queue = frame_set_queue - frame_set_used
            
            if not frame_set_queue:
                break
            
            for fid in frame_set_queue:
                cluster_set_queue = cluster_set_queue | map_frame_cluster[fid]
                
            frame_set_used = frame_set_used | frame_set_queue
            cluster_set_queue = cluster_set_queue - cluster_set_used
            
        cluster_set_pool = cluster_set_pool - cluster_set_used
            
        # table에 subgraph 번호 부여
        table_g[mask] = graph_id
        graph_id += 1
        
    G = graph_id
    
    # 그래프 및 주요정보 저장
    table_cftg_good = torch.hstack((table_c[:,None], table_rtlf_good[:,[3,1]], table_g[:,None]))
    np.save(outfile, table_cftg_good.detach().cpu().numpy())

@checkpoint
def frame_displacement(labelfile, cftgfile, outfile):
    # outlier 없는 테이블을 받아 간소화된 frame displacement
        
    import torch
    
    table_cftg = torch.from_numpy(np.load(cftgfile))
    table_c = table_cftg[:,0]
    table_f = table_cftg[:,1]
    table_t = table_cftg[:,2]
    table_g = table_cftg[:,3]
    
    P = table_cftg.shape[0]
    C = table_c.max() + 1
    G = table_g.max() + 1
    
    labels = np.load(labelfile)   # (F, G_H, G_W)
    F, G_H, G_W = labels.shape[:3]
    
    table_cluster_mm = torch.zeros((C,2), dtype=torch.int64)        # 클러스터별 min/max time
    table_cluster_mm_frame = torch.zeros((C,2), dtype=torch.int64)  # 클러스터별 min/max frame
    table_cluster_num_frames = torch.zeros((C,), dtype=torch.int64) # 클러스터별 프레임 개수
    table_o = torch.zeros((P,), dtype=torch.int64)                  # 클러스터내 프레임별 상대적 거리
    
    for i in range(C):
        mask = (table_c == i)
        
        min_time = torch.min(table_t[mask])
        max_time = torch.max(table_t[mask])
        table_cluster_mm[i] = torch.tensor([min_time, max_time])
        
        min_frame = torch.min(table_f[mask])
        max_frame = torch.max(table_f[mask])
        table_cluster_mm_frame[i] = torch.tensor([min_frame, max_frame])
        
        frames = table_f[mask].unique()
        table_cluster_num_frames[i] = frames.shape[0]
        
        table_o[mask] = table_t[mask] - min_time
        
    # 클러스터별 길이 추출
    table_cluster_len = table_cluster_mm[:,1] - table_cluster_mm[:,0] + 1
    
    # 클러스터별 변위(initial loc)값을 위한 (C,) 행렬 준비
    table_cluster_d = torch.zeros((C,), dtype=torch.int64)
    
    # 프레임별 변위값을 위한 (F,) 행렬 준비
    table_frame_d = torch.zeros((F,), dtype=torch.int64)
    
    is_used_frame = torch.zeros((F,), dtype=bool)
    
    # subtable에 대해 (클러스터 개수 C')
    #   클러스터 내 프레임번호가 동일선상에 있도록 클러스터를 배열
    for g in range(G):
        # subtable 추출
        mask = (table_g == g)
        table_f_sub = table_f[mask]
        table_c_sub = table_c[mask]
        table_o_sub = table_o[mask]
        clusters_sub = torch.unique(table_c_sub)
        table_cluster_mm_frame_sub = table_cluster_mm_frame[clusters_sub]
        table_cluster_num_frames_sub = table_cluster_num_frames[clusters_sub]
        
        # (min_frame, num_frames) 기준으로 클러스터를 정렬
        indices_num_frames = torch.argsort(table_cluster_num_frames_sub, descending=True)
        indices_mm_frame = torch.argsort(table_cluster_mm_frame_sub[indices_num_frames,0], stable=True)
        clusters_sub_sorted = clusters_sub[indices_num_frames][indices_mm_frame]
        
        # 첫 클러스터에 대해 작업:
        #   첫 클러스터의 인덱스릂 0으로
        #   프레임 변위 처리
        cid_0 = clusters_sub_sorted[0]
        table_cluster_d[cid_0] = 0
        
        indices_c_raw = torch.argwhere(table_c_sub == cid_0).flatten()
        
        # indices_c = indices_c_raw[mask]
        # frames = table_f_sub[indices_c]
        # rellocs = table_o_sub[indices_c]
        frames = table_f_sub[indices_c_raw]
        rellocs = table_o_sub[indices_c_raw]
        
        table_frame_d[frames] = rellocs
        
        is_used_frame[frames] = True
        cluster_used = [cid_0]
        
        # 두번째 클러스터부터의 작업
        irregular_clusters = []
        for cid in clusters_sub_sorted[1:]:
            indices_c = torch.argwhere(table_c_sub == cid).flatten()
            
            frames = table_f_sub[indices_c]
            rellocs = table_o_sub[indices_c]
            
            # 이전에 처리한 프레임과 겹치는 프레임을 추출
            intersection_indices = is_used_frame[frames]
            intersection = frames[intersection_indices]
            
            # 세로 클러스터 조건2: 현재 프레임중 이미 처리했던 프레임이 1개이상 존재해야함
            #   나중에 다시 처리 필요
            if intersection.numel() == 0:
                irregular_clusters.append(cid)
                continue
            
            # 겹치는 프레임들의 d 값을 구해둠
            dominant_d = table_frame_d[intersection]
            rellocs_sub = rellocs[intersection_indices]
            
            # 겹치는 프레임 각각에 대해,
            #   해당 프레임의 이전 d값에 매칭 후
            #   이전 d값들과 차이를 계산
            O_target = rellocs_sub
            D_target = dominant_d
            Orig_d = dominant_d.repeat(intersection.shape[0],1)
            New_d = D_target[:,None] - O_target[:,None] + rellocs_sub.repeat(intersection.shape[0],1)
            losses = torch.mean((Orig_d - New_d)**2, dim=1, dtype=torch.float32)
                
            # 최소 loss을 만드는 프레임에 대한 d값을 추출
            idx_min_loss = losses.argmin()                  # TODO: 최소 loss 0일텐데 필요한가?
            d_target_min = dominant_d[idx_min_loss]
            o_target_min = rellocs_sub[idx_min_loss]
            
            # 클러스터의 위치 지정
            table_cluster_d[cid] = d_target_min - o_target_min
            
            # 프레임의 새 d를 카운터에 누적
            table_frame_d[frames] = d_target_min - o_target_min + rellocs
            
            is_used_frame[frames] = True
            cluster_used += [cid]
            
        # 정상 클러스터만 포함 (개수: C'')
        cluster_used = torch.tensor(cluster_used)
            
        # 거대한 클러스터 다리 (C'',M) 행렬을 준비 (M: 클러스터를 조건에 맞게 넓은면끼리 이어붙일때 그 폭)
        C_sub = cluster_used.shape[0]
        min_cluster_d_sub = table_cluster_d[cluster_used].min()
        M = (table_cluster_d[cluster_used] + table_cluster_len[cluster_used]).max() - min_cluster_d_sub
        array_cluster_bridge = torch.zeros((C_sub,M), dtype=torch.int64)
    
        # 클러스터 다리에 클러스터를 나열
        for i, cid in enumerate(cluster_used):
            j_l = table_cluster_d[cid] - min_cluster_d_sub
            j_r = j_l + table_cluster_len[cid]
            
            indices_c = (table_c_sub == cid)
            frames = table_f_sub[indices_c]
            rellocs = table_o_sub[indices_c]
            
            uniques, counts = rellocs.unique(return_counts=True)    # unique는 상대위치값이 오름차순인 배열
            counts_extended = torch.zeros((j_r-j_l), dtype=counts.dtype)
            counts_extended[uniques] = counts
            
            array_cluster_bridge[i, j_l:j_r] += counts_extended
        
        # 클러스터 중심을 계산
        j_center = array_cluster_bridge.sum(dim=0).argmax() + min_cluster_d_sub
        
        # 프레임의 변위값을 업데이트
        frames_used = table_f_sub.unique()
        table_frame_d[frames_used] -= j_center
    
    # 프레임 변위값 저장
    table_frame_d = -table_frame_d  # 실제 프레임 변위는 위 계산법과 부호가 반대임
    np.save(outfile, table_frame_d.detach().cpu().numpy())
    
@checkpoint
def labels_to_falling_space_with_displacements(labelfile, dfile, outfile):
    
    import torch
    import torchvision.datasets as dset
    from torch.nn.functional import one_hot
    
    from detectionAI.config import Config as CfgAI
    
    # 라벨 로드
    labels = np.load(labelfile)     # (F, G_H, G_W)
    # L = len(dset.ImageFolder(root=CfgAI.ref_dir).classes)   # 배경제외 라벨개수
    L = len(load_json(CfgAI.ref_path)["ids"])   # 배경제외 라벨개수
    
    # 프레임드랍에 의한 위치조정값을 로드
    disp_arr = np.load(dfile)
    
    # 프레임 개수 세어서 거대한 one-hot 초기행렬 생성
    F, G_H, G_W = labels.shape[:3]
    T = (torch.arange(0, F) + disp_arr + G_H).max()
    fs_arr_cal = torch.zeros((T, G_W, L+1), dtype=torch.uint8)  # (T, G_W, L+1)
    
    # 각 프레임별 라벨행렬을 one-hot으로 변경
    labels_inv = np.flip(labels, 1).copy()       # 행을 뒤집어 맨 아래가 0번째가 되도록 함
    labels_inv = torch.from_numpy(labels_inv)
    onehots = one_hot(labels_inv, num_classes=L+1)    # (F, G_H, G_W, L+1)
    
    # 모든 라벨행렬들을 프레임드랍 반영하여 swift하면서 초기행렬에 합산
    for i, onehot in enumerate(onehots):
        i_cal = i + disp_arr[i]
        fs_arr_cal[i_cal:i_cal+G_H] += onehot
        
    # 결과: array of shape (T, G_W, L+1)
    np.save(outfile, fs_arr_cal.detach().cpu().numpy())

@checkpoint
def tag_outliers_fs(fsfile, outfile, thr_min_labels=0, thr_min_bars=0):
    
    import torch
    
    fs_arr = torch.from_numpy(np.load(fsfile))
    fs_arr_tag = fs_arr.clone()
    
    # 조건1: 각 칸엔 최소라벨수 이상 라벨이 발견되어야 함
    if thr_min_labels > 0:
        wrong_indices = torch.argwhere(
            (fs_arr[:,:,1:].sum(dim=2) < thr_min_labels) \
            & (fs_arr[:,:,1:].sum(dim=2) > 0)
        )
        fs_arr_tag[wrong_indices[:,0], wrong_indices[:,1], :] = 0  # 0으로 tag
    
    # 조건2: 첫번째 bar 이전, 마지막 bar 이후의 노트는 전부 오류임
    if thr_min_bars > 0:
        indices_bar = torch.argwhere(fs_arr[:,:,1].sum(dim=1) > thr_min_bars).squeeze()
        index_first_bar = indices_bar.min()
        index_last_bar = indices_bar.max()
        fs_arr_tag[:index_first_bar, :, :] = 0      # 0으로 태그
        fs_arr_tag[index_last_bar+1:, :, :] = 0
    
    # 태그된 fs_arr를 저장
    np.save(outfile, fs_arr_tag)

@checkpoint
def filter_BAR(fsfile, labelfile, outfile, thr_min_label_factor=1/3, thr_min_count=4):
    # fs_arr 에서 bar 노트를 추출
    # bar의 차트 구조:
    #   array element: [bar_exist], bar_exist of type Bool
    #   array shape: (T,)
    import torch
    
    labels = np.load(labelfile)     # (F, G_H, G_W)
    F, G_H, G_W = labels.shape
    
    fs_arr = torch.from_numpy(np.load(fsfile))    # (F+G_H-1, G_W, L+1)
    
    thr_min_label_bar = int(G_H * thr_min_label_factor)    # 한 위치에서 thr이상 탐지되어야 bar패턴으로 간주
    thr_min_count_bar = thr_min_count           # 라인에서 thr이상 bar패턴 감지시 bar노트로 간주
    fs_arr_bar = torch.where(fs_arr[:,:,1] >= thr_min_label_bar, fs_arr[:,:,1], 0)
    fs_bar_count = fs_arr_bar.sum(dim=1)
    fs_bar_exist = torch.where(fs_bar_count >= thr_min_count_bar, True, False)
    
    # 차트를 저장
    np.save(outfile, fs_bar_exist)

@checkpoint
def filter_cluster_wo_bar(fsfile, outfile):
    # bar가 제거된 fs_arr에서 가로축으로 라벨을 클러스터링
    
    import torch
    from sklearn.cluster import DBSCAN
    
    fs_arr = torch.from_numpy(np.load(fsfile))
    
    # 배경과 bar를 제외한 라벨의 좌표를 다 모아둠
    fs_label = fs_arr[:,:,2:].sum(dim=2)
    table_yx = torch.argwhere(fs_label > 0)
    
    # clustering 준비
    thr_dist = 1        # 이웃한 두 점간의 최대거리
    db = DBSCAN(eps=thr_dist, min_samples=1, metric='euclidean')
    
    # 각 행이 cluster 번호인 (N,) 행렬 추출 (클러스터는 인덱스 0~(C-1) 까지)
    t_scale = 5
    r_scale = 1
    db_input_Y = table_yx[:,0] * t_scale
    db_input_X = table_yx[:,1] * r_scale
    db_input = torch.hstack([db_input_Y[:,None], db_input_X[:,None]])
    table_c = torch.tensor(db.fit_predict(db_input))
    
    table_yxc = torch.hstack([table_yx, table_c[:,None]])
    np.save(outfile, table_yxc)

@checkpoint
def inspect_cluster(yxcfile, labelfile, fsfile, outfile, thr_kernel=0.1, thr_min_label=3):
    # DFS방식의 백트래킹으로 클러스터를 스캔,
    #   클러스터를 list of (note_type, width, score) 형태의 솔루션으로 변환
    # 솔루션 여러개일시 가장 높은 mean score를 갖는 솔루션을 택
    # 솔루션에서 각 노트를 (y,x,w,note_type) 로 변환, 모아서 table_chart를 생성
    #
    # 백트래킹 노드 템플릿: [(note_type, width, score), ...]
    #   note_type - 0(noise), 1(end), 2(glissando), 3(simple), 4(trill)
    #   width - 0(init), 1~28(노트의 폭)
    #   score - 일치하는 정도 (최대 G_H)
    from typing import List, Tuple
    import torch
    
    F, G_H, G_W = torch.from_numpy(np.load(labelfile)).shape
    
    fs_arr = torch.from_numpy(np.load(fsfile))
    
    table_yxc = torch.from_numpy(np.load(yxcfile))
    table_yx = table_yxc[:,[0,1]]
    table_c = table_yxc[:,2]
    C = table_c.unique().shape[0]
    
    # kernel 함수 정의 (n: note_type, w: width)
    def kernel(n, w):
        assert n in [1,2,3,4] and w in range(2,27+1)
        
        # label 순서에 따른 시작위치. 
        #   0(배경),1(bar),2~4(end),5~7(glissando),8~10(simple),11~14(trill)
        if n in [1,2,3]:
            s = 2+3*(n-1)
        else:   # n == 4
            s = 2+3*2   # trill은 기본 simple로 해야함
        
        # kernel의 정의
        ker = torch.zeros((14,w), dtype=torch.int64)
        ker[s,0] = ker[s+2,-1] = 1
        for i in range(1,w-1):
            ker[s+1,i] = 1
        if n == 4:
            if w % 2 == 0:
                ker[s+1,[w//2-1,w//2]] = 0
                ker[2+3*3,w//2-1] = 1
                ker[2+3*3+2,w//2] = 1
            else:
                ker[s+1,w//2] = 0
                ker[2+3*3+1,w//2] = 1
        ker = ker / w
            
        # kernel값의 통과한도 결정 (maximum G_H)
        # thr = torch.tensor([G_H*0.3])
        thr = torch.tensor([G_H*thr_kernel])
        
        return ker, thr
    
    table_chart = []
    
    for cid in range(C):
        mask = (table_c == cid)
        indices_inc_x = torch.argsort(table_yx[mask,1])
        YX = table_yx[mask][indices_inc_x]
        labels = fs_arr[YX[:,0], YX[:,1]]   # x축 오름차순 정렬된 클러스터 내 라벨. shape: (W, 14)
        W = labels.shape[0]
        
        stack: List[List[Tuple[int,int]]] = [[(0,0,0)]]     # initial node
        solutions = []
        
        while stack:
            node = stack.pop()
            ll = sum([nw[1] for nw in node])
            lr = W - ll
            lpos = ll
            
            if lr == 0:
                # solution!
                solutions.append(node[1:])  # initial node 제외하고 추가
                continue
            
            candits = []
            
            # noise 파악
            label = labels[lpos]
            n_label = label[2:].sum()
            if n_label > 0 and n_label < thr_min_label:
                score = G_H-(n_label**2-1)  # quadratic decrease
                next_node = node.copy() + [(0,1,score.item())]     # 폭이 1인 잡음 추가
                candits.append(next_node)
            
            # 노트인지 파악 (타입 4종, 최대폭=남은라벨수)
            for w in range(2,lr+1):
                labels_w = labels[lpos:lpos+w].T    # shape: (14,w)
                
                for node_type in [1,2,3,4]:
                    if node_type == 4 and w == 2:
                        continue
                    # end, glissando, simple, trill 순으로 조사
                    f_ker, thr_score = kernel(node_type,w)
                    score = torch.sum(f_ker * labels_w)
                    if score >= thr_score:
                        next_node = node.copy() + [(node_type,w,score.item())]
                        candits.append(next_node)
                
            # candidates를 먼저 발견한게 위에 오도록 추가
            stack += candits[::-1]
        
        # 솔루션이 없을 경우 - noise로 결정 (thr 설계미스일 가능성도 있지만)
        if len(solutions) == 0:
            # logger.warning(f"irregular cluster (type h1): cid {cid}, (y,x,w)=({YX[0,0].item()},{YX[0,1].item()},{labels.shape[0]})")
            n_labels = labels[:,2:].sum(dim=1)
            scores = (G_H-(n_labels**2-1)).clamp(min=0).tolist()    # quadratic decrease
            score = np.mean(scores)
            solution = [(0,W,score.item())]   # 클러스터 통째로 잡음으로 추가
            solutions.append(solution)
        
        # 찾은 솔루션 중 노드들의 (공평한)평균점수가 가장 높은 솔루션을 겟
        avg_scores = []
        for nodes in solutions:
            avg_score = sum([s*w for _,w,s in nodes]) / W
            avg_scores.append(avg_score)
        avg_scores = torch.tensor(avg_scores)
        id_max = avg_scores.argmax()
        answer = solutions[id_max]
    
        # 솔루션을 차트에 추가
        y,x = YX[0]
        d = 0
        for n,w,s in answer:
            table_chart.append([y,x+d,w,n])
            d += w
            
    table_chart = torch.tensor(table_chart)
        
    np.save(outfile, table_chart.numpy())

@checkpoint
def draw_fs_arr_fixed_note_type(fsfile, outdir, barfile="", chartfile="", max_height=300, max_hstack=10, font_scale=0.3, font_thickness=1, outline_rgb=(255,255,0)):
    
    fs_arr = np.load(fsfile)    # (T, G_W, L+1)
    
    T, G_W = fs_arr.shape[:2]
    L = fs_arr.shape[2] - 1
    
    # 라벨의 텍스트 크기 파악
    label_ex = "0000\n0000\n0000"
    lw, lh = vis.get_text_size(
        label_ex,
        font_scale=font_scale,
        font_thickness=font_thickness,
    )
    
    # # debug: 디자인 개편
    # lh = lh // 2
    
    # 텍스트 크기에 따라 거대한 검정 행렬 준비
    rh = (max_height - T % max_height) % max_height
    fs_img = np.zeros((lh*(T+rh),lw*28,3), dtype=np.uint8)
    H, W = fs_img.shape[:2]
    
    # 그릴 위치를 결정할 mask 행렬 준비
    fs_mask = np.zeros((T,G_W), dtype=bool)
    
    # bar 노트 처리
    if barfile:
        chart_bar = np.load(barfile)
        I_bar = np.argwhere(chart_bar == True).flatten()
        fs_arr[:,:,1] = 0   # bar 패턴을 제거
        for i in I_bar:
            yl = H - (i+1) * lh
            yr = H - (i) * lh
            fs_img[yl:yr] = np.array([64, 64, 64])   # 회색 마디선
            
    # chart의 노트타입별 처리
    colors = np.array(
        [[128,128,128],     # noise
         [0,255,0],         # end
         [255,255,0],       # glissando
         [255,255,255],     # simple
         [255,0,255]]       # trill
    )
    if chartfile:
        table_chart = np.load(chartfile)
        for y,x,w,n in table_chart:
            yl = int(H - (y+1) * lh)
            yr = int(H - (y) * lh)
            xl = int(x * lw)
            xr = int((x+w) * lw - 1)
            fs_img[yl:yr,xl:xr] = colors[n]
        
    # 나머지 라벨이 존재하는 곳에 표기
    label_mask = (fs_arr[:,:,1:].sum(axis=2) > 0)
    fs_mask |= label_mask
    
    hEx = lambda x: hex(x)[2:].upper() if (x>=0 and x<16) else chr(x-16+ord('G'))
    
    # 라벨을 채워둠
    for i,j in np.argwhere(fs_mask):
        # 텍스트 시작위치
        x = j * lw
        y = H - (i+1) * lh
        
        # 텍스트
        multi_hot = fs_arr[i,j]
        if multi_hot[1:].sum() == 0:
            continue
        pat_hot = multi_hot[2:]
        label_l = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 0])
        label_m = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 1])
        label_r = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 2])
        label = '\n'.join([label_l, label_m, label_r])
        
        # 그리기
        img = fs_img[y:y+lh, x:x+lw, :]
        if multi_hot[1] > 0:
            # bar노트는 외각선 긋기
            img[[0,lh-1],:,:] = np.array(outline_rgb)
            img[:,[0,lw-1],:] = np.array(outline_rgb)
        img = vis.add_text_to_image(
            img, 
            label,
            top_left_xy=(0,0),
            font_scale=font_scale,
            font_thickness=font_thickness,
            font_color_rgb=(0,255,0),
        )
        fs_img[y:y+lh, x:x+lw, :] = img

    # max_height기준으로 이미지 자르기
    fs_img_bgr = fs_img[:,:,[2,1,0]]
    fs_img_arr = fs_img_bgr.reshape(-1, max_height*lh, W, 3)
    fs_img_arr = np.flip(fs_img_arr, axis=0)
    
    # 이미지 양옆에 boundary 삽입 (1px 흰색선)
    s = fs_img_arr.shape
    fs_bd = np.ones((s[0],s[1],5,s[3]), dtype=np.uint8) * 255
    fs_img_arr = np.concatenate([fs_bd, fs_img_arr, fs_bd], axis=2)
    
    # max_hstack기준으로 이미지 나란히 엮을 준비
    N = fs_img_arr.shape[0]
    arange = list(range(0, N, max_hstack))
    if arange[-1] < N:
        arange.append(N)
    
    # 모두 파일로 저장
    os.makedirs(outdir, exist_ok=True)
    for i, j in zip(arange[:-1], arange[1:]):
        img = np.hstack(fs_img_arr[i:j])
        path = os.path.join(outdir, f"{i}.jpg")
        cv2.imwrite(path, img)

@checkpoint
def filter_tenuto_trill_chart(chartfile, out_tenuto, out_trill):
    # table_chart에서 테누토,트릴을 추출
    # 두 노트의 차트 구조:
    #   array element: [y1,y2,x1,x2],
    #       y1,y2: 시간축 시작과 끝, 정수값 (0, T-1)
    #       x1,x2: 건반축 시작과 끝, 정수값 (0~27)
    #   array shape: (None,4)
    
    import torch
    from sklearn.cluster import DBSCAN
    
    table_chart = torch.from_numpy(np.load(chartfile))
    
    # end노트를 다 모아둠
    chart_end = table_chart[table_chart[:,3] == 1]
    
    # end노트가 없으면 종료
    if chart_end.shape[0] == 0:
        np.save(out_tenuto, np.empty((0,4), dtype=np.int64))
        np.save(out_trill, np.empty((0,4), dtype=np.int64))
        return
    
    # 시작노트가 될 후보 라벨을 전부 모아둔 행렬 준비 (simple 또는 trill)
    chart_start = table_chart[(table_chart[:,3] == 3) | (table_chart[:,3] == 4)]
    
    # 테누토, 트릴 차트 준비, 둘다 (y1,y2,x1,x2) 형식
    chart_tenuto = []
    chart_trill = []
    
    # 분리된 클러스터 각각에 대해
    for note_end in chart_end:
        y,x,w,_ = note_end
        
        # 시작노트를 찾음 (end 노트보다 아래에서 가장 가까운 노트)
        start_candits = chart_start[(chart_start[:,1]==x) & (chart_start[:,2]==w)]
        if not (y > start_candits[:,0]).any():
            logger.warning(f"end note cannot match: {note_end.tolist()}")
            continue
        start_candits = start_candits[y - start_candits[:,0] > 0]
        note_start = start_candits[start_candits[:,0].argmax()]
        y_s, _, _, n_s = note_start
        
        # 시작노트가 테누토인지 트릴인지 구분
        if n_s == 3:
            chart_tenuto.append([y_s,y,x,x+w-1])
        elif n_s == 4:
            chart_trill.append([y_s,y,x,x+w-1])
        
    # 차트를 저장
    np.save(out_tenuto, np.array(chart_tenuto, dtype=np.int64))
    np.save(out_trill, np.array(chart_trill, dtype=np.int64))

@checkpoint
def filter_glissando_simple_chart(chartfile, tenutofile, out_glissando, out_simple):
    # table_chart에서 글리산도와 일반노트를 추출
    # 두 노트의 차트 구조:
    #   array element: [y,x1,x2],
    #       y: 시간축, 정수값 (0, T-1)
    #       x1,x2: 건반축 시작과 끝, 정수값 (0~27)
    #   array shape: (None,3)
    
    import torch
    from sklearn.cluster import DBSCAN
    
    table_chart = torch.from_numpy(np.load(chartfile))

    # chart에서 tenuto를 구성하는 start노트를 제거
    chart_tenuto = torch.from_numpy(np.load(tenutofile))
    mask = torch.ones(table_chart.shape[0], dtype=torch.bool)
    for y1,y2,x1,x2 in chart_tenuto:
        note_start = torch.tensor([y1,x1,(x2-x1+1),3], dtype=torch.int64)
        idx_start = torch.argwhere((table_chart == note_start).all(dim=1)).squeeze().item()
        mask[idx_start] = False
        
    table_chart = table_chart[mask]
    
    # glissando 노트 추출
    chart_glissando = table_chart[table_chart[:,3] == 2, :3]
    if chart_glissando.shape[0] == 0:
        chart_glissando = np.empty((0,3), dtype=np.int64)
    else:
        chart_glissando[:,2] = chart_glissando[:,1] + chart_glissando[:,2] - 1
    
    # simple 노트 추출
    chart_simple = table_chart[table_chart[:,3] == 3, :3]
    if chart_simple.shape[0] == 0:
        chart_simple = np.empty((0,3), dtype=np.int64)
    else:
        chart_simple[:,2] = chart_simple[:,1] + chart_simple[:,2] - 1
    
    # 차트를 저장
    np.save(out_glissando, np.array(chart_glissando, dtype=np.int64))
    np.save(out_simple, np.array(chart_simple, dtype=np.int64))


def run_until_label(videofile, metafile, outdir, cfg, keep=True):
    import shutil
    vp, mp = videofile, metafile
    
    # 파일경로 세팅
    outframedir = str(Path(outdir) / "frames")
    barfile = str(Path(outdir) / "bars.npy")
    gridfile = str(Path(outdir) / "grid.npy")
    bboxfile = str(Path(outdir) / "bbox.npy")
    labelfile = str(Path(outdir) / "labels.npy")
    
    if Path(labelfile).exists():
        return labelfile    # 이미 끝난 영상은 패스
    
    # parameters 를 로드
    meta = load_json(mp)
    start_frame_t, fps = meta["start"]
    start_frame_idx = int(start_frame_t * fps)
    end_frame_t, fps = meta["end"]
    end_frame_idx = int(end_frame_t * fps)
    roi = meta["roi"]
    display = meta["display"] if "display" in meta else "splash"
    
    s = cfg.BAR.FRAME_START
    n = cfg.BAR.NUM_FRAMES    # 시작프레임부터 특정 프레임개수 내에서만 bar찾기
    thr_minlen = cfg.BAR.THR_MIN_LEN_FACTOR
    thr_cluster = cfg.BAR.THR_CLUSTER
    rm_factor = cfg.GRID.BAR_REMOVAL_LEN_FACTOR
    n_routes = cfg.GRID.NUM_ROUTES
    xh_factor = cfg.GRID.FACTOR_OF_HALF_BBOX_X_TO_UNIT_X
    yh_factor = cfg.GRID.FACTOR_OF_HALF_BBOX_Y_TO_UNIT_X
    
    # 프레임 저장 (이미 저장시 path만 반환)
    frame_paths = save_frame(vp, outframedir, start_idx=start_frame_idx, end_idx=end_frame_idx)
    
    if not Path(barfile).exists():
        bar_fit(
            frame_paths=frame_paths[s:s+n], 
            outfile=barfile,
            roi=roi,
            display=display,
            thr_minlen=thr_minlen,
            thr_cluster=thr_cluster
        )
    
    if not Path(gridfile).exists():
        grid_from_bar(
            barfile=barfile,
            outfile=gridfile,
            rm_factor=rm_factor,
            n_routes=n_routes
        )
    
    if not Path(bboxfile).exists():
        save_grid_bbox(
            gridfile=gridfile,
            outfile=bboxfile,
            n_routes=n_routes,
            xh_factor=xh_factor,
            yh_factor=yh_factor
        )
    
    if not Path(labelfile).exists():
        label_bbox_frames_fast(
            frame_paths=frame_paths,
            bboxfile=bboxfile,
            outfile=labelfile
        )
    
    if not keep:
        shutil.rmtree(outframedir)
        os.remove(barfile)
        os.remove(gridfile)
        os.remove(bboxfile)
        
    return labelfile


def reset_until_label(outdir):
    import shutil
    
    # 파일경로 세팅
    outframedir = str(Path(outdir) / "frames")
    barfile = str(Path(outdir) / "bars.npy")
    gridfile = str(Path(outdir) / "grid.npy")
    bboxfile = str(Path(outdir) / "bbox.npy")
    labelfile = str(Path(outdir) / "labels.npy")
    
    if Path(labelfile).exists():
        os.remove(labelfile)
    if Path(bboxfile).exists():
        os.remove(bboxfile)
    if Path(gridfile).exists():
        os.remove(gridfile)
    if Path(barfile).exists():
        os.remove(barfile)
    if Path(outframedir).exists():
        shutil.rmtree(outframedir)
    

def run_until_fs_after_label(outdir, cfg, keep=True):
    # 파일경로 세팅
    labelfile = str(Path(outdir) / "labels.npy")
    cftgfile = str(Path(outdir) / "table_cftg.npy")
    cftg_tag_file = str(Path(outdir) / "table_cftg_tag.npy")
    cftg_good_file = str(Path(outdir) / "table_cftg_good.npy")
    dfile = str(Path(outdir) / "table_frame_d.npy")
    fsfile = str(Path(outdir) / "fs_arr.npy")
    tagfile = str(Path(outdir) / "fs_arr_tag.npy")
    
    if not Path(labelfile).exists():
        raise FileNotFoundError(labelfile)
    
    n_workers = cfg.DATALOADER.NUM_WORKERS
    thr_dist = cfg.CLUSTER.THR_MIN_DIST
    thr_min_labels = cfg.OUTLIER.THR_MIN_LABELS
    thr_min_bars = cfg.OUTLIER.THR_MIN_BARS
    
    if not Path(cftgfile).exists():
        graph_partition(
            labelfile=labelfile,
            outfile=cftgfile,
            thr_dist=thr_dist
        )
    
    if not Path(cftg_tag_file).exists():
        tag_graph_cluster_outliers(
            cftgfile=cftgfile,
            outfile=cftg_tag_file,
            n_workers=n_workers
        )
    
    if not Path(cftg_good_file).exists():
        graph_partition_with_tag(
            labelfile=labelfile,
            cftgfile=cftg_tag_file,
            outfile=cftg_good_file,
            thr_dist=thr_dist
        )
    
    if not Path(dfile).exists():
        frame_displacement(
            labelfile=labelfile,
            cftgfile=cftg_good_file,
            outfile=dfile
        )
    
    if not Path(fsfile).exists():
        labels_to_falling_space_with_displacements(
            labelfile=labelfile,
            dfile=dfile,
            outfile=fsfile
        )
    
    if not Path(tagfile).exists():
        tag_outliers_fs(
            fsfile=fsfile,
            outfile=tagfile,
            thr_min_labels=thr_min_labels,
            thr_min_bars=thr_min_bars
        )
    
    if not keep:
        os.remove(labelfile)
        os.remove(cftgfile)
        os.remove(dfile)
        os.remove(fsfile)
        
    return tagfile


def reset_until_fs_after_label(outdir):
    # 파일경로 세팅
    cftgfile = str(Path(outdir) / "table_cftg.npy")
    cftg_tag_file = str(Path(outdir) / "table_cftg_tag.npy")
    cftg_good_file = str(Path(outdir) / "table_cftg_good.npy")
    dfile = str(Path(outdir) / "table_frame_d.npy")
    fsfile = str(Path(outdir) / "fs_arr.npy")
    tagfile = str(Path(outdir) / "fs_arr_tag.npy")
    
    if Path(tagfile).exists():
        os.remove(tagfile)
    if Path(fsfile).exists():
        os.remove(fsfile)
    if Path(dfile).exists():
        os.remove(dfile)
    if Path(cftg_good_file).exists():
        os.remove(cftg_good_file)
    if Path(cftg_tag_file).exists():
        os.remove(cftg_tag_file)
    if Path(cftgfile).exists():
        os.remove(cftgfile)


def run_until_pattern_after_fs(outdir, cfg, debug=False):
    # 파일경로 세팅
    fsfile = str(Path(outdir) / "fs_arr_tag.npy")
    labelfile = str(Path(outdir) / "labels.npy")
    yxcfile = str(Path(outdir) / "table_yxc.npy")
    chartfile = str(Path(outdir) / "table_chart.npy")
    barfile = str(Path(outdir) / "chart_bar.npy")
    fsimgdir = str(Path(outdir) / "fs_arr_fixed_note_type_img")
    
    if not Path(fsfile).exists():
        raise FileNotFoundError(fsfile)
    elif not Path(labelfile).exists():
        raise FileNotFoundError(labelfile)
        
    thr_kernel = cfg.FILTER.THR_KERNEL
    thr_min_label = cfg.FILTER.THR_MIN_LABELS
    thr_min_label_factor = cfg.FILTER.BAR.THR_MIN_LABEL_FACTOR
    thr_min_count = cfg.FILTER.BAR.THR_MIN_COUNT
        
    if not Path(yxcfile).exists():
        filter_cluster_wo_bar(
            fsfile=fsfile,
            outfile=yxcfile
        )
    
    if not Path(chartfile).exists():
        inspect_cluster(
            yxcfile=yxcfile,
            labelfile=labelfile,
            fsfile=fsfile,
            outfile=chartfile,
            thr_kernel=thr_kernel,
            thr_min_label=thr_min_label
        )
    
    if not Path(barfile).exists():
        filter_BAR(
            fsfile=fsfile,
            labelfile=labelfile,
            outfile=barfile,
            thr_min_label_factor=thr_min_label_factor,
            thr_min_count=thr_min_count
        )
        
    if debug and not Path(fsimgdir).exists():
        draw_fs_arr_fixed_note_type(
            fsfile=fsfile,
            outdir=fsimgdir,
            barfile=barfile,
            chartfile=chartfile
        )
    
    return chartfile, barfile


def reset_until_pattern_after_fs(outdir):
    import shutil

    # 파일경로 세팅
    yxcfile = str(Path(outdir) / "table_yxc.npy")
    chartfile = str(Path(outdir) / "table_chart.npy")
    barfile = str(Path(outdir) / "chart_bar.npy")
    fsimgdir = str(Path(outdir) / "fs_arr_fixed_note_type_img")
    
    if Path(fsimgdir).exists():
        shutil.rmtree(fsimgdir)
    if Path(barfile).exists():
        os.remove(barfile)
    if Path(chartfile).exists():
        os.remove(chartfile)
    if Path(yxcfile).exists():
        os.remove(yxcfile)


def run_until_chart_after_pattern(outdir):
    # 파일경로 세팅
    chartfile = str(Path(outdir) / "table_chart.npy")
    tenutofile = str(Path(outdir) / "chart_tenuto.npy")
    trillfile = str(Path(outdir) / "chart_trill.npy")
    simplefile = str(Path(outdir) / "chart_simple.npy")
    glissandofile = str(Path(outdir) / "chart_glissando.npy")
    
    if not Path(chartfile).exists():
        raise FileNotFoundError(chartfile)
    
    if not Path(tenutofile).exists() or not Path(trillfile).exists():
        filter_tenuto_trill_chart(
            chartfile=chartfile,
            out_tenuto=tenutofile,
            out_trill=trillfile
        )
    
    if not Path(glissandofile).exists() or not Path(simplefile).exists():
        filter_glissando_simple_chart(
            chartfile=chartfile,
            tenutofile=tenutofile,
            out_glissando=glissandofile,
            out_simple=simplefile
        )
    
    return [simplefile, tenutofile, trillfile, glissandofile]


def reset_until_chart_after_pattern(outdir):
    # 파일경로 세팅
    tenutofile = str(Path(outdir) / "chart_tenuto.npy")
    trillfile = str(Path(outdir) / "chart_trill.npy")
    simplefile = str(Path(outdir) / "chart_simple.npy")
    glissandofile = str(Path(outdir) / "chart_glissando.npy")
    
    if Path(glissandofile).exists():
        os.remove(glissandofile)
    if Path(simplefile).exists():
        os.remove(simplefile)
    if Path(tenutofile).exists():
        os.remove(tenutofile)
    if Path(trillfile).exists():
        os.remove(trillfile)
    

def draw_fs_arr(fsfile, outdir, barfile="", tenutofile="", trillfile="", glissandofile="", simplefile="", max_height=300, max_hstack=10, font_scale=0.3, font_thickness=1, outline_rgb=(255,255,0)):
    
    fs_arr = np.load(fsfile)    # (F+G_H-1, G_W, L+1)
    
    T, G_W = fs_arr.shape[:2]
    L = fs_arr.shape[2] - 1
    
    # 라벨의 텍스트 크기 파악
    label_ex = "0000\n0000\n0000"
    lw, lh = vis.get_text_size(
        label_ex,
        font_scale=font_scale,
        font_thickness=font_thickness,
    )
    
    # debug: 디자인 개편
    # lh = lh // 2
    
    # 텍스트 크기에 따라 거대한 검정 행렬 준비
    rh = (max_height - T % max_height) % max_height
    fs_img = np.zeros((lh*(T+rh),lw*28,3), dtype=np.uint8)
    H, W = fs_img.shape[:2]
    
    # 그릴 위치를 결정할 mask 행렬 준비
    fs_mask = np.zeros((T,G_W), dtype=bool)
    
    # bar 노트 처리
    if barfile:
        chart_bar = np.load(barfile)
        I_bar = np.argwhere(chart_bar == True).flatten()
        fs_arr[:,:,1] = 0   # bar 패턴을 제거
        for i in I_bar:
            yl = H - (i+1) * lh
            yr = H - (i) * lh
            fs_img[yl:yr] = np.array([64, 64, 64])   # 회색 마디선
            
    # tenuto 노트 처리
    if tenutofile:
        chart_tenuto = np.load(tenutofile)
        for y1,y2,x1,x2 in chart_tenuto:
            fs_arr[y1:y2+1,x1:x2+1,:] = 0   # trill이 놓인 위치의 라벨을 전부 제거
            yl = int(H - (y2+1) * lh)
            yr = int(H - (y1) * lh)
            xl = int(x1 * lw + 2)
            xr = int((x2+1) * lw - 2)
            fs_img[yl:yr,xl:xr] = np.array([0,255,0])     # 초록색 사각형
    
    # trill 노트 처리
    if trillfile:
        chart_trill = np.load(trillfile)
        for y1,y2,x1,x2 in chart_trill:
            fs_arr[y1:y2+1,x1:x2+1,:] = 0   # trill이 놓인 위치의 라벨을 전부 제거
            yl = int(H - (y2+1) * lh)
            yr = int(H - (y1) * lh)
            xl = int(x1 * lw + 2)
            xr = int((x2+1) * lw - 2)
            fs_img[yl:yr,xl:xr] = np.array([255,0,255])     # 보라색 사각형
            
    # glissando 노트 처리
    if glissandofile:
        chart_glissando = np.load(glissandofile)
        for y,x1,x2 in chart_glissando:
            fs_arr[y,x1:x2+1,:] = 0   # glissando가 놓인 위치의 라벨을 전부 제거
            yl = int(H - (y+1) * lh)
            yr = int(H - (y) * lh)
            xl = int(x1 * lw + 2)
            xr = int((x2+1) * lw - 2)
            fs_img[yl:yr,xl:xr] = np.array([255,255,0])     # 노란색 사각형
    
    # simple 노트 처리
    if simplefile:
        chart_simple = np.load(simplefile)
        for y,x1,x2 in chart_simple:
            fs_arr[y,x1:x2+1,:] = 0   # simple가 놓인 위치의 라벨을 전부 제거
            yl = int(H - (y+1) * lh)
            yr = int(H - (y) * lh)
            xl = int(x1 * lw + 2)
            xr = int((x2+1) * lw - 2)
            fs_img[yl:yr,xl:xr] = np.array([255,255,255])     # 흰색 사각형
        
    # 나머지 라벨이 존재하는 곳에 표기
    label_mask = (fs_arr[:,:,1:].sum(axis=2) > 0)
    fs_mask |= label_mask
    
    hEx = lambda x: hex(x)[2:].upper() if (x>=0 and x<16) else chr(x-16+ord('G'))
    
    # 라벨을 채워둠
    for i,j in np.argwhere(fs_mask):
        # 텍스트 시작위치
        x = j * lw
        y = H - (i+1) * lh
        
        # 텍스트
        multi_hot = fs_arr[i,j]
        if multi_hot[1:].sum() == 0:
            continue
        pat_hot = multi_hot[2:]
        label_l = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 0])
        label_m = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 1])
        label_r = ''.join([f"{hEx(l)}" for k,l in enumerate(pat_hot) if k % 3 == 2])
        label = '\n'.join([label_l, label_m, label_r])
        
        # 그리기
        img = fs_img[y:y+lh, x:x+lw, :]
        if multi_hot[1] > 0:
            # bar노트는 외각선 긋기
            img[[0,lh-1],:,:] = np.array(outline_rgb)
            img[:,[0,lw-1],:] = np.array(outline_rgb)
        img = vis.add_text_to_image(
            img, 
            label,
            top_left_xy=(0,0),
            font_scale=font_scale,
            font_thickness=font_thickness,
            font_color_rgb=(0,255,0),
        )
        fs_img[y:y+lh, x:x+lw, :] = img

    # max_height기준으로 이미지 자르기
    fs_img_bgr = fs_img[:,:,[2,1,0]]
    fs_img_arr = fs_img_bgr.reshape(-1, max_height*lh, W, 3)
    fs_img_arr = np.flip(fs_img_arr, axis=0)
    
    # 이미지 양옆에 boundary 삽입 (1px 흰색선)
    s = fs_img_arr.shape
    fs_bd = np.ones((s[0],s[1],5,s[3]), dtype=np.uint8) * 255
    fs_img_arr = np.concatenate([fs_bd, fs_img_arr, fs_bd], axis=2)
    
    # max_hstack기준으로 이미지 나란히 엮을 준비
    N = fs_img_arr.shape[0]
    arange = list(range(0, N, max_hstack))
    if arange[-1] < N:
        arange.append(N)
    
    # 모두 파일로 저장
    os.makedirs(outdir, exist_ok=True)
    for i, j in zip(arange[:-1], arange[1:]):
        img = np.hstack(fs_img_arr[i:j])
        path = os.path.join(outdir, f"{i}.jpg")
        cv2.imwrite(path, img)
        

def draw_chart(outdir):
    # 파일경로 세팅
    fsfile = str(Path(outdir) / "fs_arr_tag.npy")
    barfile = str(Path(outdir) / "chart_bar.npy")
    tenutofile = str(Path(outdir) / "chart_tenuto.npy")
    trillfile = str(Path(outdir) / "chart_trill.npy")
    simplefile = str(Path(outdir) / "chart_simple.npy")
    glissandofile = str(Path(outdir) / "chart_glissando.npy")
    chartdir = str(Path(outdir) / "fs_arr_tag")

    if not Path(fsfile).exists():
        logger.warning("fsfile not found. make fsfile first")
        return

    if not Path(barfile).exists():
        barfile = ""
    if not Path(simplefile).exists():
        simplefile = ""
    if not Path(tenutofile).exists():
        tenutofile = ""
    if not Path(trillfile).exists():
        trillfile = ""
    if not Path(glissandofile).exists():
        glissandofile = ""
    
    if barfile + simplefile + tenutofile + trillfile + glissandofile == "":
        logger.warning("chart file not found. make chart first")
        return
    
    if not Path(chartdir).exists():
        draw_fs_arr(fsfile=fsfile, outdir=chartdir,
                    barfile=barfile, 
                    simplefile=simplefile, 
                    tenutofile=tenutofile, 
                    trillfile=trillfile, 
                    glissandofile=glissandofile)


def reset_draw_chart(outdir):
    import shutil
    chartdir = str(Path(outdir) / "fs_arr_tag")
    
    if Path(chartdir).exists():
        shutil.rmtree(chartdir)


def manual_update_chart(outdir):
    # 파일경로 세팅
    barfile = str(Path(outdir) / "chart_bar.npy")
    tenutofile = str(Path(outdir) / "chart_tenuto.npy")
    trillfile = str(Path(outdir) / "chart_trill.npy")
    simplefile = str(Path(outdir) / "chart_simple.npy")
    glissandofile = str(Path(outdir) / "chart_glissando.npy")
    
    if not (Path(barfile).exists() and Path(tenutofile).exists() and Path(trillfile).exists() and Path(simplefile).exists() and Path(glissandofile).exists()):
        logger.warning("chart file not found. make chart first")
        return
    
    chart_bar_path = barfile
    chart_simple_path = simplefile
    chart_tenuto_path = tenutofile
    chart_trill_path = trillfile
    chart_glissando_path = glissandofile
    
    chart_bar = np.load(chart_bar_path)
    chart_simple = np.load(chart_simple_path)
    chart_tenuto = np.load(chart_tenuto_path)
    chart_trill = np.load(chart_trill_path)
    chart_glissando = np.load(chart_glissando_path)
    
    txt_bar_path = Path(chart_bar_path).parent / f"{Path(chart_bar_path).stem}.txt"
    txt_simple_path = Path(chart_simple_path).parent / f"{Path(chart_simple_path).stem}.txt"
    txt_tenuto_path = Path(chart_tenuto_path).parent / f"{Path(chart_tenuto_path).stem}.txt"
    txt_trill_path = Path(chart_trill_path).parent / f"{Path(chart_trill_path).stem}.txt"
    txt_glissando_path = Path(chart_glissando_path).parent / f"{Path(chart_glissando_path).stem}.txt"
    
    np.savetxt(txt_bar_path, chart_bar, fmt="%d")
    np.savetxt(txt_simple_path, chart_simple, fmt="%d")
    np.savetxt(txt_tenuto_path, chart_tenuto, fmt="%d")
    np.savetxt(txt_trill_path, chart_trill, fmt="%d")
    np.savetxt(txt_glissando_path, chart_glissando, fmt="%d")
        
    input("Press Enter on terminal if finished ... ")
    
    chart_bar_new = np.loadtxt(txt_bar_path, dtype=chart_bar.dtype).reshape(-1)
    chart_simple_new = np.loadtxt(txt_simple_path, dtype=chart_simple.dtype).reshape(-1,3)
    chart_tenuto_new = np.loadtxt(txt_tenuto_path, dtype=chart_tenuto.dtype).reshape(-1,4)
    chart_trill_new = np.loadtxt(txt_trill_path, dtype=chart_trill.dtype).reshape(-1,4)
    chart_glissando_new = np.loadtxt(txt_glissando_path, dtype=chart_glissando.dtype).reshape(-1,3)

    np.save(chart_bar_path, chart_bar_new)
    np.save(chart_simple_path, chart_simple_new)
    np.save(chart_tenuto_path, chart_tenuto_new)
    np.save(chart_trill_path, chart_trill_new)
    np.save(chart_glissando_path, chart_glissando_new)
    
    os.remove(txt_bar_path)
    os.remove(txt_simple_path)
    os.remove(txt_tenuto_path)
    os.remove(txt_trill_path)
    os.remove(txt_glissando_path)
    

def run():
    # Notice:
    #   이 코드는 연구목적이므로 일반사용자를 위해 최적화되어있지 않습니다.
    #   Setting 0 을 준비된 데이터 경로로 설정하고
    #   Setting 1 은 많은 데이터 처리시 이어서 진행을 위해 사용 가능하며
    #   Setting 2 의 일부를 필요에 따라 주석 설정/해제 하여 beatmap 추출의 각 step을 이행하세요.
    #   Setting 3 은 특정 step의 undo를 위해 사용 가능합니다.
    
    ### Setting 0-1. data path
    datadir = "nosdb/video_expert"
    metadir = "nosdb/video_expert_meta"
    outdir = "nosdb/video_expert_output"
    cfgdir = "nosdb/video_expert_config"
    
    os.makedirs(outdir, exist_ok=True)

    remdir = os.listdir(datadir)
    
    # ### Setting 1. for continued extraction
    # for rdir in os.listdir(outdir):
    #     rp = [rp for rp in os.listdir(datadir) if Path(rp).stem == rdir][0]
    #     remdir.append(rp)
    
    vmc_triples = []
    for rp in remdir:
        mp = Path(metadir) / f"{Path(rp).stem}.json"
        assert mp.exists(), f"metafile not found: {mp}"
        cp = Path(cfgdir) / f"{Path(rp).stem}.yaml"
        vmc_triples.append((str(Path(datadir)/rp), str(mp), str(cp)))
        
    for vp, mp, cp in vmc_triples:
        rootdir = str(Path(outdir) / Path(vp).stem)
        os.makedirs(rootdir, exist_ok=True)
        
        logger.info(f"{rootdir}")
        try:
            cfg = get_cfg_defaults()
            if Path(cp).exists():
                cfg.merge_from_file(cp)
            cfg.freeze()
            
            ### Setting 2-1. step-by-step run
            run_until_label(vp, mp, rootdir, cfg=cfg, keep=True)
            run_until_fs_after_label(outdir=rootdir, cfg=cfg)
            run_until_pattern_after_fs(outdir=rootdir, cfg=cfg, debug=True)
            run_until_chart_after_pattern(outdir=rootdir)
            draw_chart(rootdir)
            
            # ### Setting 2-2. final revision
            # manual_update_chart(rootdir)
            
            # ### Setting 3. step-by-step reset
            # reset_draw_chart(outdir=rootdir)
            # reset_until_chart_after_pattern(outdir=rootdir)
            # reset_until_pattern_after_fs(outdir=rootdir)
            # reset_until_fs_after_label(outdir=rootdir)
            # reset_until_label(outdir=rootdir)
            
        except Exception as e:
            logger.error(e)
            pass


def run_demo():
    # Notice:
    #   이 코드는 연구목적이므로 일반사용자를 위해 최적화되어있지 않습니다.
    #   Setting 1 을 준비된 데이터 경로로 설정하고
    #   Setting 2 의 일부를 필요에 따라 주석 설정/해제 하여 beatmap 추출의 각 step을 이행하세요.
    #   Setting 3 은 특정 step의 undo를 위해 사용 가능합니다.
    
    ### Setting 1. data path
    videofile = "data/video.mp4"
    metafile = "data/meta.json"
    configfile = "data/config.yaml"     # optional
    outdir = "output"
    
    os.makedirs(outdir, exist_ok=True)
    
    rootdir = str(Path(outdir) / Path(videofile).stem)
    os.makedirs(rootdir, exist_ok=True)
        
    logger.info(f"{rootdir}")
    try:
        cfg = get_cfg_defaults()
        if Path(configfile).exists():
            cfg.merge_from_file(configfile)
        cfg.freeze()
        
        ### Setting 2-1. step-by-step run
        run_until_label(videofile, metafile, rootdir, cfg=cfg, keep=True)
        run_until_fs_after_label(outdir=rootdir, cfg=cfg)
        run_until_pattern_after_fs(outdir=rootdir, cfg=cfg, debug=True)
        run_until_chart_after_pattern(outdir=rootdir)
        draw_chart(rootdir)
        
        # ### Setting 2-2. final revision
        # manual_update_chart(rootdir)
        
        # ### Setting 3. step-by-step reset
        # reset_draw_chart(outdir=rootdir)
        # reset_until_chart_after_pattern(outdir=rootdir)
        # reset_until_pattern_after_fs(outdir=rootdir)
        # reset_until_fs_after_label(outdir=rootdir)
        # reset_until_label(outdir=rootdir)
        
    except Exception as e:
        logger.error(e)
        pass


if __name__ == "__main__":
    # run()
    run_demo()