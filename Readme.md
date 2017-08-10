## Code structure

### Required libraries

The whole project is realized under the TensorFlow framework (v1.2). The
other libraries that are used include (mainly for the preprocessing part)
NumPy, SciPy, scikit-learn, librosa (avicar), and imageio (Montalbano).

Also notice that the project relies heavily on the use of TF-Slim
(tf.contrib.slim), which is a higher level API of the core TensorFlow library.

### Reading data

Adapting from the source code found on the github directory
tensorflow/models/slim
([link to the page](https://github.com/tensorflow/models/tree/master/slim)),
I use some ad hoc functions and classes provided by TF-Slim to deal
with datasets. In particular, the data is first read somewhere and converted
in TFRecords files (the standard format supported by TensorFlow) since
different datasets may have different file organizations and may store
data differently. Once TFRecords created, similar functions can be easily
used to deal with different datasets.

Despite the possibility of doing an abstraction and put all the common
routines related to the data pipeline in a same place, I chose to totally
separate the codes for different datasets (which corresponds to the
subdirectories of the `data` directory). In this way, corresponding files
can be picked up directly to reuse the same pipeline for another purpose.

At this point, three main families of functions should be mentioned.

1. `convert_*`: These are functions that create TFRecords from datasets.
The detailed implementation and accepted arguments may vary a lot
from case to case.

2. `get_split_*`: Given the path of the directory containing dataset
TFRecords and the name that indicates the part of the dataset that we're
interested in (ex: train/validation), we get the `dataset` instance that is
asked (the `dataset` class is defined in TF-Slim).

3. `load_batch_*`: These functions take the `dataset` instance as argument
and provide data in batches.

In general, once converted dataset in TFRcords using `convert_*`, we call
the functions `get_split_*` and `load_batch_*` to get the data used
for training or evaluation. Note that the queue runners must be started
otherwise the program will hang.

### Training, evaluation and visualization

In my internship, the three things that I do the most often are: training
models, evaluation of models and visualization of learned representations.
To avoid writing the same code every time, the directory `routines` contains
some basic skeletons for these tasks, including creating the graph, starting
the session, etc. A supervisor (`tf.train.supervisor`) is used for the
training part, but it seems that it's deprectaed
([Document that tf.train.Supervisor is deprecated. Issue #6263 - GitHub](https://github.com/tensorflow/tensorflow/issues/6263))
and one should use `tf.train.MonitoredSession` instead, though the use of
supervisor is always in the official tutorial that is on the site of TensorFlow
[Supervisor: Training Helper for Days-Long Trainings. | TensorFlow](https://www.tensorflow.org/programmers_guide/supervisor).

There are several principal flaws (or problems) of my implementations
concerning these classes and their subclasses. First, for the sake of
convenience, I didn't respect the leading-underscore convention for python,
but what's more important is the abuse of class attributes and `**kwargs`.
In fact, since the real algorithms that are implemented can have a great
variety, it's extremely difficult to predict the variables that will be
used and instance variables can be defined in any method.
It acts somehow differently from a classic python class
that really represents something to be described. Despite all of this,
there is still a lack of plasticity and if we want to train some networks in
a totally different way it may not be appropriate (generally an
end-to-end training scheme is much easier with these classes).

The three functions `train`, `evaluate` and `visualize` are just defined
for convenience and only work for subclasses that take exactly one network
architecture as input during initialization.

### Network architectures

All network architectures are functions that take some inputs and produce
different outputs at different levels. The arguement `final_endpoint`
is thus used to indicate the layer that we want to use as output
(it's by default the last layer, but for example we may want to get
an internal representation using the middle layer of an auto-encoder).
Unfortunately, with related to the scope names and endpoint names of
the networks, there isn't a global convention that works for all due
to the large variety of network architectures that I experimented during my
internship. Sometimes we are also able to decide the entrypoint of
the network (relatively rare).

The implementations of different networks are separated in different files
(ex: `images/CNN_architecture.py`, `multimodal/gesture/architecture.py`)
depending on the objective of the network.

### Classification

Since I focused a lot on the problem of classification during the eight weeks
(though it's not at all what I meant to do for my internship), in the
directory `classify` we find common routines used for classification.
The class `TrainClassify` contains a large part of things that we often
do to train a classifier, such as using the cross-entropy loss function
and evaluate the prediction accuracy throughout the training process.
To define a subclass of it, the methods that must be defined are

* `compute`: mainly call `compute_logits` with the right arguments

* `compute_logits`: the main function that computes the tensor logits
which is mapped to probabilities using the softmax function, in some
special cases one may not want to use this function
(see `multimodal\AVSR\transfer.py`)

* `get_data` & `decide_used_data`: for reading data, like any other
training classes

One can refer to `TrainClassifyImagesCNN` defined in
`images/classify_routines.py` which inherits from `TrainClassifyCNN` and
`TrainClassifyImages` for an example. Similarly, `EvaluateClassify` and
their subclasses are used for evaluations for a classification model.
Also see `test/classify_images` for basics uses of these classes (image part).

### Images, audios and videos

In my internship, I worked with different kinds of inputs and used different
architectures to deal with them (for classification and representation learning).
For the codes that are specially used for a single modality, they can be
found in the directories `images`, `audio` and `video`. The file `basics.py`
is usally just for loading the data and may contains some basic
visualization classes inheriting from `Visualization` defined in
`routines/visualization.py`. In the `images` directory we find the definition
of convolution auto-encoder architectures and the routines that are used to train
and evaluate them. Refer to `test/CAE.py` for an example of their uses.

### Multimodal experiments

In the directory `multimodal` one can find the codes for all the multimodal
experiments that are described in my report.
The directory `multimodal/gesture` contains mainly the shared reprsentation
learning experiment. The files `multimodal/gesture/fusion_AE.py` and
`multimodal/gesture/classify_routines.py` implement classes to train and
evaluate these models. The file `test/fusion.py` allows us to easily
replicating the experiment by varing function arguments.
`multimodal/gesture/embedding.py` is an exception and is used for training
and visualizing a common embedding from the two modalities, but better
loss function needs to be found to get it work.
The codes used for the AVSR knowledge transfer experiment are in
`mutlimodal/AVSR/transfer.py`.

### nets\_base

The file `arg_scope` defines a common argument scopes used by all the models.
Other files are taken directly from
tensorflow/models/[slim](https://github.com/tensorflow/models/tree/master/slim)
and are not modified.
