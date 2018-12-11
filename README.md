# Partial_Conv-Tensorflow
Simple Tensorflow implementation of [Partial Convolution based Padding](https://arxiv.org/abs/1811.11718)
![partial_conv](./assets/partial_conv.png)

## How to use
```python
  # typical convolution layer with zero padding
  x = conv(x, channels, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, scope='conv')
  
  # partial convolution based padding
  x = conv(x, channels, kernel=3, stride=2, pad=1, pad_type='partial', use_bias=True, scope='conv')

```
## Results
### Activation map
![activation_map](./assets/activation_map.png)

### ImageNet Classification
<div align="">
  <img src="./assets/imagenet_classification.png" width="600">
  <img src="./assets/best_top1_acc.png" width="900">
</div>

### Segmentation
![seg_1](./assets/segmentation_1.png)
![seg_2](./assets/segmentation_2.png)



## Author
Junho Kim
