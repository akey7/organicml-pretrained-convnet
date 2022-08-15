# organicml-pretrained-convnet
A classification attempt based on a VGG16 convoluational network. Uses keras on R.

Normally on the GTX 1650, training will run out of GPU memory. To prevent this, I launch RStudio on my Linux box with an environment variable like this:

```
TF_FORCE_GPU_ALLOW_GROWTH='true' /usr/bin/rstudio
```
