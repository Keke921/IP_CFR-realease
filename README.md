## Identity-preserved Complete Face Recovering Network for Partial Face Image
Mengke Li, Yiu-ming Cheung 
_________________
A [pytorch](http://pytorch.org/) implementation of Identity-preserved Complete Face Recovering Network for Partial Face Image

Contact: mengkejiajia@qq.com

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.7.1
- [face_alignment](https://github.com/1adrianb/face-alignment)


### Dataset
- CelebA (for training)
- LFW (for testing) 

### Before training

We divide the CelebA dataset into training set and test set according to the identity of subjects. The images used for training are list in for example identity.txt in the following form:
```bash
image_name1.jpg label1
image_name2.jpg label2
```

Then read the identity.txt to load the images for training.

The images need to be pre-aligned before training. We use FAN [1] to align all images in CelebA and LFW according to the position of the eyes。


### Training 

- To train IP-CFR using the train script simply specify the parameters listed in **train.py** as a flag or manually change them.
We provide the example of training on CelebA dataset with this repo:
```bash
python train.py --root_path="E:\data"
		--dataset_name="prepared_image/img_align_celeba_crop" \
		--dataset_label="prepared_image/identity_CelebA.txt" \
		
```

### Evaluation
If the input is a single partial image, use FAN (the face_alignment function) to find the position of the existing face informace. Then align it to the corresponding position of the training images. 


## References
- [1] Bulat, Adrian, and Georgios Tzimiropoulos. "How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks)." Proceedings of the IEEE International Conference on Computer Vision. 2017.


## Citation

Please cite the paper if the codes are helpful for you research.

M.K. Li and Y.M. Cheung, "Identity-preserved Complete Face Recovering Network for Partial Face Image",  IEEE Transactions on Emerging Topics in Computational Intelligence, in press.
