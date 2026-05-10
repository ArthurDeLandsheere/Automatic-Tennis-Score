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

Install the requirements:
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

Create the `checkpoints` folder at the root of `Action-Spotting`. Go to https://github.com/jhong93/e2e-spot-models/tree/main and download the two models `tennis_rny002gsm_gru_rgb` and `tennis_rny008gsm_gru_rgb`. The `checkpoints` folder should now look like this:
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
├── visualize_predictions.py


Tracking
├── checkpoints                 # contains the pretrained models
│   └── ...
├── data
│   └── tennis
│       ├── videos              # mp4 videos for each match
│       │   └── ...
├── jobs                        # centralize all your .sbatch files here
│   ├── ...
├── outputs                     # centralize all your .log files here
│   ├──...
├── scripts
│   ├── track.py
│   └── visualize.py
├── src
│   ├── ...


Score
├── data
│   └── tennis
│       ├── videos              # mp4 videos for each match
│       │   └── ...
│       │   frames
│       └── ...
├── jobs                        # centralize all your .sbatch files here
│   ├── ...
├── outputs                     # centralize all your .log files here
│   ├──...
├── scripts
│   └── visualize_full.py
├── io_utils.py
├── pipeline.py
├── score.py
```

## 2. Test

To test the pre-trained models, you can run
```bash
sbatch jobs/inference.sbatch
```
This will use one of the models on the `single.json` split, which only contains one small clip. To run on a full split, just change to `test`, `val` or `train`. 

This will produce predictions in the `checkpoints` file.

To visualize the predictions on the small clip, you can use the `visualize_predictions.ipynb` notebook. You have to **run it locally** so just download the notebook, the predictions, and the frames on your machine and run the notebook.

## 3. Evaluation

I ran inference on the whole test split with both pre-trained models (rny002 and rny008) and the results are quite good. You can see the detailed results in the `eval.log` file in the `outputs` folder but here is a summary.

So basically I ran the two models, plus an ensemble model that averages the predictions of both models. Concretely, for every frame in every video, each model outputs a confidence score for each of the 6 event classes. The ensemble simply takes the average of those two score vectors, then picks the class with the highest averaged score as the prediction.

I ran those tests with the following parameters:
- *tolerance = [0,1,2,4]*: maximum allowed frame offset between a predicted event and the ground truth for it to count as a correct detection
- *NMS = [0,1,2,3,5]*: window size (in frames) within which duplicate detections are suppressed, keeping only the most confident one (it's just like we saw in the theory)

Since the results with tol=0 are really bad (logical since a model would then only be right if it predicts events at the exact frame they are happening), I did two tables, one in which I average the precision over all tolerances including 0 and the second in which I only average over tol=1,2,4. So in the tables below the numbers represent the Avg mAP over all videos in the test set and over all classes.

### Including tol=0
| Model    | NMS=0     | NMS=1 | NMS=2 | NMS=3 | NMS=5 |
|----------|-----------|-------|-------|-------|-------|
| rny002   | **86.65** | 83.98 | 83.89 | 83.83 | 83.78 |
| rny008   | **86.55** | 84.02 | 83.90 | 83.85 | 83.80 |
| ensemble | **87.18** | 84.56 | 84.47 | 84.43 | 84.41 |

### Excluding tol=0
| Model    | NMS=0 | NMS=1     | NMS=2 | NMS=3 | NMS=5 |
|----------|-------|-----------|-------|-------|-------|
| rny002   | **96.29** | 95.99 | 95.90 | 95.83 | 95.76 |
| rny008   | 96.97 | **97.07** | 96.94 | 96.88 | 96.82 |
| ensemble | 96.63 | **97.11** | 97.02 | 96.98 | 96.95 |

So of course the results are better when we average over tol=1,2,4 and exclude tol=0 since the results with tol=0 are bad. The rny008 and ensemble perform the best when excluding tol=0 with NMS=1.