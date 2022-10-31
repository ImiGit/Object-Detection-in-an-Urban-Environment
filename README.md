# Object Detection in an Urban Environment
Final project in the [Udacity](https://www.udacity.com/) Data Scientist Nanodegree program.
by **Iman Babaei**



## Data

For this project, data from the [Waymo Open dataset](https://waymo.com/open/) was already provided in the workspace, so I did not need to download it separately.

## Structure

### Data

The data I used for training, validation and testing was organized as below on ***my working space***:
```
/home/workspace/data
    - train: contain the train data 
    - val: contain the val data 
    - test: contains 3 files to test your model and create inference videos
```

The `train` and `val` folders contained files that have been downsampled: they have been selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

### build
The build folder contains the necessary file and instruction to run the project on a docker:
```
build/
    - Dockerfile
    - README.md
    - requirements.txt
```

### experiments
The experiments folder will be organized as follow:
```
experiments/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - label_map.pbtxt
```
### Graphs

This folder contains the results for the pre-trained reference experiment and trials to improve the results. Refer to the [Results.md file](Results.md) for in depth analysis of these files.

```
Graphs/
    - Experiment_0: - Results of the pretrained reference model.
        - DetectionBoxes_precision.PNG
        - DetectionBoxes_recall.PNG
        - Learning_rate.PNG
        - learning_rate.svg
        - loss_curve_outliers_off.PNG
        - loss_curve_outliers_on.PNG
    - Experiment_1: - Results of the first trial to improve the model by adding augmentations.
        - image_01.png
        - image_02.png
        - image_03.png
        - image_04.png
        - learning_rate.svg
        - loss_curves.png
    - Experiment_2: - Results of the second trial to improve the model.
        - loss_curves.PNG
    - Experiment_3: - Results of the third trial to improve the model by changing the model to faster rccn.
        - loss_curves.PNG
    - animation.gif - Animated results of the saved model
    - animation_2.gif - Animated results of the saved model
    - animation_3.gif - Animated results of the saved model
    - dataset-classes_01.png
    - dataset-classes_bicycles.png
    - dataset-classes_cars.png
    - dataset-classes_pedestrians.png
    - dataset-image_01.png - Example of an image from dataset.
    - dataset-image_02.png - Example of an image from dataset.
    - dataset-image_03.png - Example of an image from dataset.
    - dataset-image_04.png - Example of an image from dataset.
    - dataset-image_05.png - Example of an image from dataset.
    - dataset-image_06.png - Example of an image from dataset.
    - dataset-image_07.png - Example of an image from dataset.
    - dataset-image_08.png - Example of an image from dataset.
    - dataset-image_09.png - Example of an image from dataset.
    - dataset-image_10.png - Example of an image from dataset.
```

### Other files

```
- .gitignore - files and paths that git will ignore
- CODEOWNERS
- create_splits.py
- download_process.py
- edit_config.py - Python file used to edit pipeline.config and make new one
- Exploratory Data Analysis.ipynb - Jupyter notebook used for exploratory data analysis
- Explore augmentations.ipynb - Jupyter notebook showing the augmentations on the dataset files.
- filenames.txt
- inference_video.py
- label_map.pbtxt
- LICENSE.md
- pipeline.config
- README.md
- Results.md - Write-up file containing the results of the project.
- utils.py
```

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```
