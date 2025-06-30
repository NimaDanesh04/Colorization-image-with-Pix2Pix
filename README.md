<div align="center">

#  Pix2Pix Image Colorizer  

_An image-to-image translation project using a Pix2Pix GAN to restore colors to grayscale images._

![pix2pix-demo](https://user-images.githubusercontent.com/demo/colorization.gif)

</div>

---

##  About the Project

Pix2Pix Image Colorizer uses a Conditional GAN framework to learn the mapping between grayscale and colored image pairs. Built on the U-Net generator and PatchGAN discriminator, it minimizes adversarial + L1 reconstruction loss.

> Based on the paper: _"Image-to-Image Translation with Conditional Adversarial Networks"_ by Isola et al., 2017.

---

##  Features

-  Pix2Pix architecture (U-Net + PatchGAN)
-  Supports custom paired datasets
-  Adversarial + L1 loss combination
-  Optional evaluation: PSNR, SSIM
-  Easy training/inference scripts

---

##  Tech Stack

| Tool        | Version |
|-------------|---------|
| Python      | 3.8+    |
| tensorflow     | 2.1   |
| NumPy       | ✓       |
| OpenCV      | ✓       |

---

##  Quick Start

###  Install Dependencies

```bash
git clone https://github.com/yourusername/pix2pix-colorizer.git
cd pix2pix-colorizer
pip install -r requirements.txt
```
for the run the model for you own dataset create diractory with grayscale and color images and
run
```bash
python3 train.py
```
we put the notebook for all this code in this repository but if you want you can check it in my kaggle account 
in this like https://www.kaggle.com/code/nimadanesh/colorization-image-with-pix2pix and don't forget to vote.
thanks for following this project.
