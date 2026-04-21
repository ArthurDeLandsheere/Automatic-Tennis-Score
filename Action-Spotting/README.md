# Action Spotting

This folder is a modified copy of the [Spotting Temporally Precise, Fine-Grained Events in Video](https://github.com/jhong93/spot) Github repository that I adapted for our use.

## 1. Setup

### 1.1 Environment

First, go to the `Action-Spotting` directory
```bash
cd Action-Spotting
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

> Here I show how to download+process all the videos. It takes a lot of times so you can adapt this to just do it for a few of them.

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

### 1.3 Checkpoints

Create the `checkpoints` folder at the root of `Action-Spotting`. Go to `https://github.com/jhong93/e2e-spot-models/tree/main` and download the two models `tennis_rny002gsm_gru_rgb` and `tennis_rny008gsm_gru_rgb`. The `checkpoints` folder should now look like this:
```bash
checkpoints
├── tennis_rny002gsm_gru_rgb
│   ├── checkpoint_040.pt
│   └── config.json
└── tennis_rny008gsm_gru_rgb
    ├── checkpoint_041.pt
    └── config.json
```

### 1.4 Structure

After those steps, the Action-Spotting repo should look like this
```bash
Action-Spotting
├── checkpoints                 # contains the pretrained models
│   └── ...
├── data
│   └── tennis
│       ├── frames              # processed frames
│       │   └── ...
│       ├── videos              # mp4 videos for each match
│       │   └── ...
│       ├── class.txt           # events that are spotted in the videos
│       ├── test.json           # test split
│       ├── train.json          # train split
│       ├── val.json            # val split
│       └── videos.csv          # csv file with info for each match
├── jobs                        # centralize all your .sbatch files here
│   ├── dl_videos.sbatch
│   └── process_videos.sbatch
├── outputs                     # centralize all your .log files here
│   ├── dl_videos.log
│   └── process-videos.log
└── # other files or folders
```

## 2. Test

To test the pre-trained models, you can run
```bash
sbatch jobs/inference.sbatch
```
This will use one of the models on the `single.json` split, which only contains one small clip. To run on a full split, just change to `test`, `val` or `train`. 

This will produce predictions in the `checkpoints` file.

To visualize the predictions on the small clip, you can use the `visualize_predictions.ipynb` notebook. You have to **run it locally** so just download the notebook, the predictions, and the frames on your machine and run the notebook.