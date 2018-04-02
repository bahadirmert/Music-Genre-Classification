<br>
<h2 align="center">Music Genre Classification </h2>
<p align="center"> 
  Project carried out by José Jesús Torronteras Hernández, for the subject of Neural Networks, 
Master in: Artificial Intelligence and Robotics University of  Rome  "La Sapienza"
</p>
  

## Table of contents

- [Quick start](#quick-start)
- [What's included](#whats-included)
- [GTZAN Dataset](#gtzan-dataset)
- [Execution](#execution) (*****)
- [Documentation](#documentation)(**(*****)***)
- [Contributing](#contributing)
- [Community](#community)(*****)
- [Versioning](#versioning)(*****)
- [Creators](#creators)(*****)
- [Copyright and license](#copyright-and-license)(*****)

## Quick start
The present code has been developed under python3. The simplest way to run the program is creating a virtual environment, for this it is necessary to have installed [pip](https://pypi.python.org/pypi/pip) and [virtualenv](https://github.com/pypa/virtualenv).

```bash
# We create the environment
$ virtualenv music-genre-classification
# We activate the environment
$ source music-genre-classification/bin/activate
# Install all the necessary python packages
$ pip install -r requirements.txt
```
Once we have our environment created and the packages installed, we can proceed to the [execution](#execution). (It is necessary to have downloaded and unzipped [GTZAN dataset)](#gtzan-dataset) in the folder `data / genres` [See structure Project](#whats-included))

## What's included
```
music-genre-classification/
├── data/
│   ├── array/
│   │   ├── arr_TSNE.npy (*)
│   │   ├── X_train.npy (*)
│   │   ├── ...
│   ├── assets/
│   │   ├── TSNE-PLT.png (*)
│   │   ├── ...
│   ├── csv_files/
│   │   ├── X_train.csv (*)
│   │   ├── X_test.csv (*)
│   │   ├── ...
│   ├── genres
│   │   ├── blues/
│   │   │   ├── blues.00000.au
│   │   │   ├── ...
│   │   ├── classical/
│   │   ├── ...
|   ├── config.py
	⁃	└── scripts/
    ├── Extract_Audio_Features.py
    ├── Get_Train_Test_Data.py
    ├── TSNE_figure.py
    ├── KNeighbors_Classifier.py
    └── neuraln.py
(*) Python execution will generate this files. (see examples)
```

## GTZAN Dataset
http://marsyasweb.appspot.com/download/data_sets/