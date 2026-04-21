# T-DEED

## 1. Setup

### 1.1 Environment

First, go to the `T-DEED` directory
```bash
cd T-DEED
```

Create an environment for this project and activate it:
```bash
conda create -n tennis python=3.11
conda activate tennis
```

First, install the requirements:
```bash
pip install -r requirements.txt
```
Then, install the right version of torch:
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```
And finally
```bash
conda install ffmpeg
```

### 1.2 Videos

First, you have to download the videos from youtube. To do that, run the `dl_videos.sbatch` job:
```bash
sbatch jobs/dl_videos.sbatch
```

This will download all the videos from the `videos.csv` files and produce .mp4 and .ma4 files. You can get rid of the .ma4 files after the download is completed. 

> **NB**: When I ran this, one video didn't download correctly. I just downloaded it separately after and it worked.

Then, you have to process those videos into frames:
```bash
sbatch jobs/process_videos.sbatch
```

This will process all the frames for all the videos and add them into the `data/tennis/frames` folder.

### 1.3 Structure

After those steps, the T-DEED repo should look like this
```bash
T-DEED
├── checkpoints                 # contains the pretrained models
│   └── ...
├── config                      # contains the configuration files
│   └── ...
├── data
│   └── tennis
│       ├── class.txt
│       ├── frames              # processed frames
│       │   └── ...
│       ├── test.json           # test split
│       ├── train.json          # train split
│       ├── val.json            # val split
│       ├── videos              # mp4 videos for each match
│       │   ├── usopen...
│       │   ├── ...
│       │   └── wimbledon...
│       └── videos.csv          # csv file with info for each match
├── jobs                        # centralize all your .sbatch here
│   ├── dl_videos.sbatch
│   └── process_videos.sbatch
├── outputs                     # centralize all your .log here
│   ├── dl_videos.log
│   └── process-videos.log
└── # other files or folders
```

# 2. Perform inference

You can then test one of the two pretrained models (`Tennis_big` or `Tennis_small`) using
```bash
python inference.py --model <model_name> --video_path data/tennis/videos/<match_name>.mp4
```