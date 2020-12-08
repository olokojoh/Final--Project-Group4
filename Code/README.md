# Code part
This folder contains the source code of this project

## Running Order

### Training
 
 - To train GAN based on L\*a\*b\* color space. Simply run [Color_space_GAN .py](Color_space_GAN.py)
 - To train Pix2Pix GAN. Simply run [Pix2Pix_cGAN.py](Pix2Pix_cGAN.py)

### Testing

**GAN based on L\*a\*b\***
  - To colorized gray scale photo. Simply run [test_img.py](test_img.py) and put the photos you want colorized in the [customize_test](../Data/customize_test)
  - We have already put some customize photo in the [customize_test](../Data/customize_test). Just delete it.
  
  *(Note: Due to the size limiation of Github repository, only epochs 1, 100, 200, 300, 400, 500 are available for testing)*

**Conditional GAN (Image translation)**
  - Because the model is too large to upload onto github, people can get access to it by using google drive.
    https://drive.google.com/file/d/1KmwFMZxAyYgo5f8wohToARRqw4WIWfa6/view?usp=sharing
