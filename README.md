This repository is an official PyTorch implementation of the paper **"Super-Resolution on Rotationally Scanned Photoacoustic Microscopy Images Incorporating Scanning Prior"**.

## Dependencies
* Python 3.8.5
* torch = 1.12.0
* numpy
* skimage
* matplotlib
* tqdm
* opencv-python

## Code
```bash
git clone https://github.com/11710615/SR.git
cd SR
cd src
```

## Training
```bash
python main.py --template proposed_x2 --scale 2 --patch_select grad_window --loss 1*RL1 --batch_size 8 --gpu_ids 2,3
```

## Testing
```bash
python main.py --template proposed_x2 --scale 2 --patch_select grad_window --loss 1*RL1 --batch_size 8 --gpu_ids 2,3 --test_only --save_results --pre_train */model_best.pt
```

