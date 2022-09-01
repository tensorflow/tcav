borden1_224 joined segmentation data set
========================================
Joins the following data sets:
    pascal: 10103 images
    ade20k: 22210 images
    opensurfaces: 25352 images
    dtd: 5640 images
size: (224, 224)
segmentation_size: (112, 112)
crop: False
splits: OrderedDict([('train', 0.7), ('val', 0.3)])
min_frequency: 10
min_coverage: 0.5
synonyms: <function synonyms at 0x7fc7549f2aa0>
test_limit: None
single_process: False

generated at: 2017-04-24 20:29
git label: b8f81f618de145e20c663a7565c19511f8397e11



This directory contains the following data and metadata files:

    images/[datasource]/...
         images drawn from a specific datasource are reformatted
         and scaled and saved as jpeg and png files in subdirectories
         of the images directory.

    index.csv
        contains a list of all the images in the dataset, together with
        available labeled data, in the form:

        image,split,ih,iw,sh,sw,[color,object,material,part,scene,texture]

        for examplle:
        dtd/g_0106.jpg,train,346,368,346,368,dtd/g_0106_color.png,,,,,314;194

        The first column is always the original image filename relative to
        the images/ subdirectory; then image height and width and segmentation
        heigh and width dimensions are followed by six columns for label
        data in the six categories.  Label data can be per-pixel or per-image
        (assumed to apply to every pixel in the image), and 0 or more labels
        can be specified per category.  If there are no labels, the field is ''.
        If there are multiple labels, they are separated by semicolons,
        and it is assumed that dominant interpretations are listed first.
        Per-image labels are represented by a decimal number; and per-image
        labels are represented by a filename for an image which encodes
        the per-pixel labels in the (red + 256 * green) channels.

    category.csv
        name,first,last,count,frequency

        for example:
        object,12,1138,529,208688

        In the generic case there may not be six categories; this directory
        may contain any set of categories of segmentations.  This file
        lists all segmentation categories found in the data sources,
        along with statistics on now many class labels apply to
        each category, and in what range; as well as how many
        images mention a label of that category

    label.csv
        number,name,category,frequency,coverage,syns

        for example:
        10,red-c,color(289),289,9.140027,
        21,fabric,material(36);object(3),39,4.225474,cloth

        This lists all label numbers and corresponding names, along
        with some statistics including the number of images for
        which the label appears in each category; the total number
        of images which have the label; and the pixel portions
        of images that have the label.

    c_[category].csv (for example, map_color.csv)
        code,number,name,frequency,coverage

        for example:
        4,31,glass,27,0.910724454131

        Although labels are store under a unified code, this file
        lists a standard dense coding that can be used for a
        specific subcategory of labels.
