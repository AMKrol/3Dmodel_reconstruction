# Image based 3D model recontruction

###### Still under development, refactoring needed


Aim of the project is to create a 3D model of real human forearm with ceratain anatomical structers based on images from criosection published by [National Library of Medicine, USA](https://www.nlm.nih.gov/research/visible/visible_human.html). The image analysis function is provided by [OpenCV](https://opencv.org/) library for [Python](https://pl.python.org/).

## Features

- Working on PNG images
- In default all operation will start automatic
- To speed up calcuation the multiprocessing approach was used. 
- Export results to XYZ file. It can be read as point cloud by for example [Meshlab](https://www.meshlab.net/).
- All intermediate steps in image analysis was saved to folder /results
- Final model can by found in formder /model3D


## Tech

This is the project of develop of 3D model of human forearm intended to make mesh model for mas and energy calculations in Ansys Fluent environment. The main assumption of project is to create monel with minimal geometrical difference to real world. The image-based approach allows to reach the maximum accuracy on level 0.33mm. This is the size of pixel in real world provided by [National Library of Medicine, USA](https://www.nlm.nih.gov/research/visible/visible_human.html). Next bases on OpenCV library, the image analysis was performed to get the conturs of certain anatomical elements of forearm. At last the conturs was transformed to point clound.

## Installation

After clone this repository coping of ceratin PNG file from [link](https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Female-Images/PNG_format/index.html).

Create virtual repository and install dependences:

```sh
cd 3Dmodel_reconstruction/
python -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

Now the program can be run by following:
```sh
python main.py
```