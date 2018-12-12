# PartialConv-Tensorflow
Simple Tensorflow implementation of [Partial Convolution based Padding](https://arxiv.org/abs/1811.11718)
![partial_conv](./assets/partial_conv.png)

## How to use
```python
  # typical convolution layer with zero padding
  x = conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', scope='conv')
  
  # partial convolution based padding
  x = partial_conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', scope='conv')

```
## Results
### Activation map
![activation_map](./assets/activation_map.png)

### ImageNet Classification
![classification](./assets/classification.png)


### Segmentation (DeepLab V3+)
![seg_1](./assets/segmentation_1.png)
![seg_2](./assets/segmentation_2.png)


## Author
Junho Kim
