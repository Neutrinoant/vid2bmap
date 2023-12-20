# vid2bmap
For KSC2023, to be uploaded

### Notice
Pre-trained model and its meta data are given in this repo, but training/test dataset are not.

- pretrained model: `detectionAI/checkpoints/best.ckpt`
- meta data: `detectionAI/data/label.json`

### Run Demo

I. You need to prepare any `Nostalgia` screen-recorded video, its metadata (and config file if necessary).

- video
  - screen-recording required
  - [video sample link](https://youtu.be/RHS1z85FNxg)
- metadata
  - json-formatted
  - require 3 data: 
    - [starting_time, fps]. Its time unit is second.
    - [ending_time, fps]. Its time unit is second.
    - bounding box of cropped area [x_left, x_right, y_left, y_right]. Initial point (x,y)=(0,0) is located at leftmost-top. fps means the frame rate of your video.
  - ex: ```{"start": [10.499733333333367, 60.0], "end": [146.41826666666674, 60.0], "roi": [9, 121, 1687, 482]}```
- (optional) config file
  - check the `config.py` and usage of module `yacs`, and then make configuration file as `.yaml`, if needed.

Then configurate their paths at `run_demo()` in `run.py`.

II. Run the script:
```
python run.py
```