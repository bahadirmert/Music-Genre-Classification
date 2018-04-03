<br>
<h2 align="center">Music Genre Classification </h2>
<p align="center"> 
  Project carried out by José Jesús Torronteras Hernández, for the subject of Neural Networks, 
Master in: Artificial Intelligence and Robotics University of  Rome  "La Sapienza"
</p>

 
## Table of contents
- [Abstract](#abstract)
- [Quick start](#quick-start)
- [What's included](#whats-included)
- [GTZAN Dataset](#gtzan-dataset)
- [Execution](#execution)
- [Documentation](#documentation)(****)
- [Code](#code)
- [Results](#results)(*****)
- [Versioning](#versioning)(*****)
- [Creators](#creators)(*****)
- [Copyright and license](#copyright-and-license)(*****)

## Abstract


En este proyecto entrenamos un sistema de clasificación de género musical personalizado con nuestros propios géneros y datos. El modelo toma como entrada el spectogram de marcos de música y analiza la imagen usando una Red Neural Convolucional (CNN) más una Red Neuronal Recurrente (RNN). La salida del sistema es un vector de géneros predichos para la canción.

Ponemos a punto su modelo con un pequeño conjunto de datos (30 canciones por género) y lo probamos en el conjunto de datos GTZAN proporcionando una precisión final del 80%.

## Quick start

It is necessary to have installed: Python 3.5.2

The present code has been developed under python3. The simplest way to run the program is creating a virtual environment, for this it is necessary to have installed [pip](https://pypi.python.org/pypi/pip) and [virtualenv](https://github.com/pypa/virtualenv).

```bash
# We create the environment
$ virtualenv --python python3 music-genre-classification-env
# We activate the environment
$ source music-genre-classification-env/bin/activate
# Install all the necessary python packages
$ cd music-genre-classification-env
$ git clone https://github.com/xexuew/Music-Genre-Classification.git .
$ pip3 install -r requirements.txt
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
│   ├── config.py
├── src/
│   ├── Extract_Audio_Features.py
│   ├── Get_Train_Test_Data.py
│   ├── TSNE_figure.py
│   ├── KNeighbors_Classifier.py
│   └── neuraln.py
├── requirements.txt
├── Procfile
└── Others Files (.travis.yml, .gitignore)

(*) Python execution will generate this files. (see examples)
```

## GTZAN Dataset
http://marsyasweb.appspot.com/download/data_sets/

## Execution

I have designed a main program that executes the programs that the user wants.

```python3
$ python3 main.py
```

To make the classification it is necessary first, to extract the data of the songs. It can be done locally or on a server.


## Code 