# Superresolution using an efficient sub-pixel convolutional neural network

This implementation illustrates how to use the efficient sub-pixel convolution layer described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as superresolution.

This code is from ["the Pytorch Super Resolution example repository"](https://github.com/pytorch/examples/tree/master/super_resolution), i just modified some parts to my needs and experiments

## Example Usage:

### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Super Resolve
`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`
