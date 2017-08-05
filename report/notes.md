## Texts

Since the convolution operation can be done regardless of the input size
as long as the input is bigger than the kernel, naturally a same
architecture can be used for inputs of different sizes.

## Figure captions

The CNN architecture used for FingerSpelling dataset. Its input is
a one-channel image of sized $83 \times 83$. The network contains ten
hidden layers. S stands for 'SAME' padding and V stands for 'VALID'
padding (see text).

The convolutional auto-encoder architecture with three convolutional layers
and three tranposed convolutional layer. Activation values of the middle
layen are taken as high-level features of the input image.
Inputs of the network can be of different sizes. We only use valid paddings
here.

The bimodal convolutional auto-encoder model that is used to learn
a shared multimodal representation. We simply take
the CAE architecture that is introduced earlier for each modaliy but
force them to have a shared middle layer by adding the
corresponding activation values. We then try to reconstruct the two
images separately through two disjoint paths.
