from yacs.config import CfgNode as CN
from pathlib import Path


_C = CN()

_C.ROOT_DIR = str(Path(__file__).parent.parent.parent)

_C.SCREEN = CN()
_C.SCREEN.FALL = "" # splash, slope, approach, vertical

_C.BAR = CN()
_C.BAR.FRAME_START = 0
_C.BAR.NUM_FRAMES = 1000
_C.BAR.THR_MIN_LEN_FACTOR = 0.20
_C.BAR.THR_CLUSTER = 10

_C.GRID = CN()
_C.GRID.BAR_REMOVAL_LEN_FACTOR = (6 / 1760 / 2)
_C.GRID.NUM_ROUTES = 28
_C.GRID.FACTOR_OF_HALF_BBOX_X_TO_UNIT_X = 1/2   # unit_x(bar길이를 route개수로 나눈 값)에 대한 bbox의 x축길이 절반의 비율
_C.GRID.FACTOR_OF_HALF_BBOX_Y_TO_UNIT_X = 1/3   # unit_x(bar길이를 route개수로 나눈 값)에 대한 bbox의 y축길이 절반의 비율

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 6

_C.CLUSTER = CN()
_C.CLUSTER.THR_MIN_DIST = 1

_C.OUTLIER = CN()
_C.OUTLIER.THR_MIN_LABELS = 3
_C.OUTLIER.THR_MIN_BARS = 5

_C.FILTER = CN()
_C.FILTER.BAR = CN()
_C.FILTER.BAR.THR_MIN_LABEL_FACTOR = 0.3    # G_H 의 몇퍼센트 이상 탐지되어야하는지
_C.FILTER.BAR.THR_MIN_COUNT = 4             # 탐지되 bar가 G_W 중 몇개 이상이어야 하는지
_C.FILTER.THR_KERNEL = 0.1
_C.FILTER.THR_MIN_LABELS = 3

_C.VIDEO = CN()

_C.DATASET = CN()
_C.DATASET.VIDEO = CN()
_C.DATASET.VIDEO.PATH = ""
_C.DATASET.VIDEO.FRAME_SAVED_DIR = ""


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`