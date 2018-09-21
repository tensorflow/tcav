# Discovering and Testing Concept Activation Vectors (DTCAV) [Under Submission]

Amirata Ghorbani, Been Kim

## What is DTCAV?

Discovering and Testing Concept Activation Vectors (TCAV) is a new
interpretability method to understand what signals your neural networks models
uses for prediction.

### What's special about DTCAV compared to other methods?

Typical interpretability methods show importance weights in each input feature
(e.g, pixel). DTCAV instead shows importance of high level concepts (e.g.,
color, texture, object) that frequently appear in class. Typical
interpretability methods require you to have one particular image that you are
interested in understanding. TCAV gives an explanation that is generally true
for a class of interest, beyond one image (global explanation).

For example, for a given class, we can discover and show how much the presense
of human concept was important for chain saw classifications in InceptionV3.
Even though neither human is part of the training input and nor it is trivial
that it has to do anything with classifying chain saws.

### How is DTCAV different from TCAV?

In order to use TCAV, you need to first choose a concept and then hand-label a
few examples of it. DTCAV needs none of that. You just need to have your trained
network and examples of the data distribution that you trained the network on. Additionally,
unlike TCAV, discovered concepts are patches of the class images themselves and therefore
it attributes concept importance of data to its own original features.

### The consumer of the explanation may not know machine learning too well. Can they understand the explanation?

Yes. DTCAV library is designed to be user friendly. If you run the run.py
script, you will get results prepared and saved in a human interpretable format.

### Sounds good. Do I need to change my network to use DTCAV?

No. You don't need to change or retrain your network to use TCAV.

## How to use DTCAV

The script run.py performs all you need. First of all, you need to put the
examples of the target class you want to run DTCAV for(e.g. zebra) in a folder.
You then need to create "num_random_exp" other folders each containing a set of
random images from the same data set. (e.g. each random folder can be a random
subsampling of all the existing classes in the original data set). The script
needs you to name the random folders "random500_0", "random500_1", ... In the
end, you should put the target class folder and all random folders into the same
directory that we call "source_dir". The script needs you to write a model
wrapper just like the one described in //third_party/py/tcav/tcav.ipynb:

### Create a ConceptDiscovery instance.

`cd = ConceptDiscovery( mymodel, target_class, random_concept, bottlenecks, sess,source_dir, activations_dir, cavs_dir, num_random_exp=num_random_exp,
channel_mean=True, max_imgs=max_imgs, min_imgs=min_imgs,
num_discovery_imgs=max_imgs, num_workers=25)`

### Transform examples of the class to resized patches

Created patches are saved as cd.patches, resized versions as cd.dataset, and the
original image each patch belongs to in cd.image_numbers:

`cd.create_patches(param_dict={'n_segments': [15, 50, 80]})`

### Discover concepts of the class in the given model

The output is accessible through cd.dic:

`cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})`

### Calculate CAVs and TCAV scores:

`cav_accuraciess = cd.cavs(min_acc=0.0) scores = cd.tcavs(test=False)` ```

## How to run unit tests

`python dtcav_test.py` (requires tensorflow, scikit-learn, and scipy installed)

`python dtcav_helpers_test.py` ```


