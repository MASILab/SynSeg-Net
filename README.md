# SynSeg-Net 
(End-to-end Synthesis and Segmentation Network)

## Adversarial Synthesis Learning Enables Segmentation Without Target Modality Ground Truth

This is our ongoing PyTorch implementation for end-to-end synthesis and segmentation without groudtruth.
The paper can be found in [arXiv](https://arxiv.org/abs/1712.07695) for ISBI 2018
The video can be found in [video](https://youtu.be/LTTh1WWPZ1o) on youtube

The code was written by [Yuankai Huo](https://sites.google.com/site/yuankaihuo/) and developed upon [CycleGAN Torch](https://github.com/junyanz/CycleGAN).


<img src='imgs/Figure3.jpg' width=300>
<img src='imgs/Figure2.jpg' width=300>


If you use this code for your research, please cite:

Yuankai Huo, Zhoubing Xu, Shunxing Bao, Albert Assad, Richard G. Abramson, Bennett A. Landman. [Adversarial Synthesis Learning Enables Segmentation Without Target Modality Ground Truth.](https://arxiv.org/abs/1712.07695)  In [arXiv](https://arxiv.org/abs/1712.07695) 2017.   

## Prerequisites
- Linux or macOS
- Python 2
- CPU or NVIDIA GPU + CUDA CuDNN
- pytorch 0.2

## Training Data and Testing Data
We used MRI and CT 2D slices (from coronal view) as well as MRI segmentatons as training data.
We used CT 2D slices (from coronal view) as testing data
The data orgnization can be seen in the txt files in `sublist` directory

## Training
- Train the model
```bash
python train_yh.py --dataroot ./datasets/yh --name yh_cyclegan_imgandseg --batchSize 4 --model cycle_seg --pool_size 50 --no_dropout --yh_run_model Train --dataset_mode yh_seg --input_nc 1  --seg_norm CrossEntropy --output_nc 1 --output_nc_seg 7 --checkpoints_dir /home-local/Cycle_Deep/Checkpoints/ --test_seg_output_dir /home-local/Cycle_Deep/Output/  --display_id 0 
```
- 'name' is 
`--model` "cycle_seg" means EssNet
`--yh_run_model`  " Train" means do training 
`--output_nv_seg` defines number of segmentation labels
`--checkpoints_dir`  the place to save checkpoint (model)
`--test_seg_output_dir`  the place to save the test segmentation

## Testing
- Test the synthesis
```bash
python train_yh.py --dataroot ./datasets/yh --name yh_cyclegan_imgandseg --batchSize 4 --model cycle_gan --pool_size 50 --no_dropout --yh_run_model Test --dataset_mode yh --input_nc 1 --output_nc 1 --checkpoints_dir /home-local/Cycle_Deep/Checkpoints/ --test_seg_output_dir /home-local/Cycle_Deep/Output/ --which_epoch 50
```

- Test the segmentation
```bash
python train_yh.py --dataroot ./datasets/yh --name yh_cyclegan_imgandseg --batchSize 4 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 1 --output_nc 1 --checkpoints_dir/home-local/Cycle_Deep/Checkpoints/ --test_seg_output_dir /home-local/Cycle_Deep/Output/ --which_epoch 50
```
- 'name' is 
`--which_epoch` which training epoch to load


## Citation
If you use this code for your research, please cite our papers.
```
@article{huo2017adversarial,
  title={Adversarial Synthesis Learning Enables Segmentation Without Target Modality Ground Truth},
  author={Huo, Yuankai and Xu, Zhoubing and Bao, Shunxing and Assad, Albert and Abramson, Richard G and Landman, Bennett A},
  journal={arXiv preprint arXiv:1712.07695},
  year={2017}
}
```



