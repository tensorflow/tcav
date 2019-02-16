#  Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) [ICML 2018]

 Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres

ICML Paper: https://arxiv.org/abs/1711.11279


## What is TCAV?

Testing with Concept Activation Vectors (TCAV) is a new interpretability method to understand what signals your neural networks models uses for prediction.

### What's special about TCAV compared to other methods?
Typical interpretability methods show importance weights in each input feature (e.g, pixel). TCAV instead shows importance of high level concepts (e.g., color, gender, race) for a prediction class - this is how humans communicate!

Typical interpretability methods require you to have one particular image that you are interested in understanding. TCAV gives an explanation that is generally true for a class of interest, beyond one image (global explanation).

For example, for a given class, we can show how much race or gender was important for classifications in InceptionV3. Even though neither race nor gender labels were part of the training input!

### Cool, where do these concepts come from?

TCAV learns concepts from examples. For instance, TCAV needs a couple of examples of female, and something not female to learn a "gender" concept. We have tested a variety of concepts: color,  gender,  race, textures and many others.

### Why use high level concepts instead of input features?

Humans think and communicate using concepts, and not using numbers (e.g., weights to each feature).  When there are lots of numbers to combine and reason about (many features), it becomes harder and harder for humans to make sense of the information they are accounting for. TCAV instead delivers explanations in the way humans communicate to each other.

### The consumer of the explanation may not know machine learning too well. Can they understand the explanation?
Yes. TCAV is designed to make sense to everyone - as long as they can understand the high level concept!

### Sounds good. Do I need to change my network to use TCAV?
No. You don't need to change or retrain your network to use TCAV.

## How to use TCAV

See Run TCAV.ipynb for step by step guide.

```python
mytcav = tcav.TCAV(sess,
                   target,
                   concepts,
                   bottlenecks,
                   act_gen,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=2)

results = mytcav.run()
```

## How to run unit tests

```python cav_test.py``` (requires scikit-learn and scipy installed)

```python tcav_test.py```

```python utils_test.py```
