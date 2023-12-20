import os
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import cv2

from logger import logger


class Video:
    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        gen = cv2.VideoCapture(path)
        w = int(gen.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(gen.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shape = (h, w)
    
    def nframes(self, dir):
        if Path(dir).exists():
            return len([p for p in os.listdir(dir)])
        else:
            raise Exception(f"no frames saved at {dir}. call save_frames() first.")

    def saved_paths(self, dir="."):
        if Path(dir).exists():
            paths = [str(Path(dir)/p) for p in os.listdir(dir)]
            return sorted(paths, key=lambda x:int(Path(x).stem))    # sort by frame id
        else:
            raise Exception(f"no frames saved at {dir}. call save_frames() first.")

    def _save_frame(self, id, frame, outdir):
        outpath = Path(outdir) / f"{id:05d}.jpg"
        cv2.imwrite(str(outpath), frame)

    def save_frames(self, dir=".", max_workers=8, start_idx=0, end_idx=None):
        # parallel save frames
        if not Path(dir).exists():
            Path(dir).mkdir(parents=True)
        elif any(Path(dir).iterdir()):
            logger.info(f"frames has already saved at {dir}")
            return self.saved_paths(dir)

        # store every frame in your storage
        gen = cv2.VideoCapture(self.path)
        gen.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        id = start_idx

        if end_idx is not None:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while id <= end_idx:
                    success, image = gen.read()
                    if not success:
                        break
                    executor.submit(self._save_frame, id, image, dir)
                    id += 1
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    success, image = gen.read()
                    if not success:
                        break
                    executor.submit(self._save_frame, id, image, dir)
                    id += 1
            
        logger.info(f"Total {id-start_idx} frames extracted.")
        return self.saved_paths(dir)


if __name__ == "__main__":
    video = Video(
        Path(__file__).parent.parent.parent
        / "dataset/test/preprocessing/2022-12-03 17-42-10.mp4"
    )
    video.save_frames(
        Path(__file__).parent.parent.parent / "output/2022-12-03 17-42-10"
    )
