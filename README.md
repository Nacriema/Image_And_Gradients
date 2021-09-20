# Image Gradients 

This repo contains my script for creating the gradients of image in different orientation.
More specifically: gradient along x-axis, y-axis, 45 degree, and -45 degree (based on the x-axis).

# Resulting images
### Input image
![](./Image/im_1.jpg)

### Gradient images from multiple directions:

![](./Image/im_2.png)

### Gradient magnitude and orientation:

![](./Image/mag_and_ori_2.jpg)

The gradient orientation is computed with: 

![](https://latex.codecogs.com/png.latex?%5Ctheta%20%3D%20tan%5E%7B-1%7D%28%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7D%28x%2C%20y%29%20/%20%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7D%28x%2C%20y%29%29%29)

and the magnitude of gradient at each pixel is computed with:

![](https://latex.codecogs.com/svg.latex?%5Cleft%20%7C%20%5Cbigtriangledown%20I%28x%2C%20y%29%29%20%5Cright%20%7C%20%3D%20%5Csqrt%7B%28%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7D%28x%2C%20y%29%29%5E2%20&plus;%20%28%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7D%28x%2C%20y%29%29%5E2%7D)
