# Cooperative Distribution Alignment via JSD Upper Bound
[Wonwoong Cho*](https://wonwoongcho.github.io), [Ziyu Gong*](https://www.linkedin.com/in/ziyu-gong-9700471b8/), [David I. Inouye](https://www.davidinouye.com)     

Purdue University

Neurips 2022

##### (*Equal contributions)

## Source code for the paper:

  > [Cooperative Distribution Alignment via JSD Upper Bound](https://openreview.net/forum?id=X82LFUs6g5Z&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions)) 
  > 
  > #### Code will be updated soon!

## Prerequisites
- Linux
- Python 3.8
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation
Download every file from https://anonymous.4open.science/r/AUB
(Note that the individual file should be downloaded respectively.)

### Dataset
The dataset is exactly same with the original MNIST data (http://yann.lecun.com/exdb/mnist/)

Just in case the link above does not work, you can download it here: https://drive.google.com/file/d/1E7Jggb1JCn-D7HazQuWlMxIT69vllFRU/view?usp=drive_link.

``` unzip data.zip```

#### Environment setup
1. `conda env create -f environment.yml`
2. `source activate aub`
   
### Usage 
```bash 

python run.py --multi_gpu False --setting demo --batch_size 128 --gpu_id 0 --lr 2e-4 --lambda_TC 0.0

```
- The option `--multi_gpu` is used for GPU parallelization. The code only support for running on a single GPU now. 
  The option `--setting` is the name of current experiment.
  The option `--batch_size` determines how large each batch should be feed to the GPU at once. The value of this option varies among different GPUs.
  The option `--gpu_id` select which GPU the experiment will be run on. Default is 0.
  Learning rate is determined by option `--lr`.
  Regularization for AUB is controlled by option `lambda_TC`, `0` means no regularization.
  
## BibTeX
If you use this code for your research, please cite our paper:
```
@inproceedings{cho2022AUB,
  title={Cooperative Distribution Alignment via JSD Upper Bound},
  author={Wonwoong Cho and Ziyu Gong and David I. Inouye},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
