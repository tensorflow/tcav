# Imagenet and TCAV

On this folder, you'll an example where we use TCAV on imagenet. We will use TCAV
to determine if different kinds of textures(striped, dotted, zigzagged) are important for
classifying zebras. We will download a pretrained Inception model, the zebra concept 
from imagenet and different textures from the Broden dataset.

This code will:

- Download the Broden dataset and the zebra concept from imagenet
- Create random folders to be used by TCAV for statistical significance testing.
- Create concept and target datasets from Imagenet and Broden and organize them in a TCAV readable structure.
- Run TCAV on this model for the concepts and targets we selected.

## Installation

Tensorflow must be installed to use TCAV. But it isn't included in the TCAV pip
package install_requires as a user may wish to use it with either the tensorflow
or tensorflow-gpu package. So please pip install tensorflow or tensorflow-gpu as
well as the tcav package.

> pip install tcav

### Requirements

See requirements.txt for a list of python dependencies used in testing TCAV.
These will all be installed during pip installation of tcav with the exception
of tensorflow, as mentioned above.


## Running the code

Run script creating data and models. Namely:

Determine a path for your source_dir, or where you want the script to download models and data. Then, run
```
python download_and_make_datasets.py --source_dir=YOUR_PATH --number_of_images_per_folder=50 --number_of_random_folders=3

```
Finally, go to Run TCAV.ipynb:

Set source dir to the source directory you used on and change the model and labels path to use the model located on:
 ```source_dir/inception5h/....```

Finally, run all cells in the Run TCAV.ipynb.

## Requirements

-   tensorflow
-   numpy
-   Pillow
-   matplotlib
-   scikit-learn
-   scipy

## Common issues

- Make sure your tensorflow version is 1.3.5. This code is not compatible with tensorflow v2.0
- Ensure that your virtual environment has all the necessary packages installed
- If you have issues with your network cancelling the download of Broden, please try again with a different
source_dir path