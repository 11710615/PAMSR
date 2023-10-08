
## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

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

