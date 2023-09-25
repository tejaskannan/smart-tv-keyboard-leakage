# Smart TV Acoustic Keystroke Leakage
This repository contains the code for discovering keystrokes from the audio of Apple and Samsung Smart TVs.

## Overview
This repository has two main portions: audio extraction and string recovery. The phases are split for checkpointing reasons. The audio extraction module is written in Python, and the string recovery is in Java. The folder `smarttvleakage/audio` contains most of the code related to audio extraction. The directory `smarttvleakage/search/smarttvsearch` holds the Java code involving string recovery.

This document describes how to create emulation benchmarks (to test string recovery in isolation), process recordings of users interacting with Smart TVs, and recover strings from the audio's intermediate representation. The code has only been tested on Ubuntu `20.04`. The installation instructions may differ on other systems.

We include the results of intermediate steps, as well as the dictionary priors, in this ![Google Drive](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) folder.

For all Python scripts, you can use the command line option `--help` (e.g., `python make_spectrogram.py --help`) to get a description of each argument.

## Setup

### Python
We recommend configuring the Python portion of this project inside an Anaconda environment. We have tested everything using ![Anaconda](https://docs.anaconda.com/free/anaconda/install/) Version `23.0.1`. The first step is to create a virtual environment, as shown below (named `smarttv`).
```
conda env create --name smarttv -f environment.yml
```
This command will both create the environment and install the relevant Python packages. Upon completion, you will need to install the local `smarttvleakage` package for development via the command below. You must run this from the root directory (e.g., where `setup.py` is).
```
pip install -e .
```
To verify this installation, navigate to the `smarttvleakage` directory and run `python make_spectrogram.py --help`. This program should *not* error and instead should print out the command line arguments.

### Java
The Java code requires an installation of the ![Java Runtime Environment](https://ubuntu.com/tutorials/install-jre#1-overview). We use `openjdk` version `17.05`. The Java portion of the project uses `json` parsing, local `SQL` databases, and `junit`. The `jar` files for these libraries are in the `jars` directory. You need to specify both paths to these `jars` and a path to the Java project. To add these references, you must update your `$CLASSPATH$` environment variable. This step is accomplished by adding the following lines to your `~/.bashrc` file.
```
export SMARTTV_HOME=<PATH-TO-PROJECT-DIRECTORY>
export CLASSPATH=$CLASSPATH:$SMARTTV_HOME/smarttvleakage/search:$SMARTTV_HOME/jars/junit-4.10.jar:$SMARTTV_HOME/jars/json-20220924.jar:$SMARTTV_HOME/jars/sqlite-jdbc-3.36.0.3.jar:.
``` 
The `<PATH-TO-PROJECT-DIRECTORY>` should be the full path to the directory of this repository in your filesystem. You will need to restart your terminal for this change to take effect. After this change, you can check to ensure the references work by navigating to the `smarttvleakage/search/smarttvsearch` folder and running the following.
```
javac SearchRunner.java
java smarttvsearch.SearchRunner
```
This command should show the following error, which is expected because the above command does *not* provide the required command line arguments.
```
Exception in thread "main" java.lang.IllegalArgumentException: Must provide the (1) --input-file, (2) --output-file, (3) --password-prior, (4) --english-prior, and (5) --zip-prior
```

## Audio Extraction
The codebase processes video recordings of Smart TV interactions. We take videos for debugging purposes--the attack strips away the video and uses audio signals alone (e.g., see the file `smarttvleakage/audio/audio_extractor.py`). This section describes how to execute the audio extraction phase of the attack. Each holding can hold multiple interactions with the keyboard (i.e., keyboard instances). The output of this phase is a `json` file containing the move count sequences for each indentified keyboard instance. A single move count sequences looks as follows.
```
[{"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 512, "end_time": 831, "move_times": [528, 569, 773]}, {"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 877, "end_time": 1019, "move_times": [892, 925, 958]},...]
```

We describe two methods for creating these move count sequences.
1. **Emulation:** Creates the move count sequence `json` file algorithmically using the keyboard layout.
2. **Real Recordings:** Extracts move count sequences from the audio of real interactions with Smart TVs.

To facilitate reproducibility, you many find the recordings of users entering passwords, credit card details, and web searches into Apple and Samsung Smart TVs in this ![Box drive](https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf).

For the remainder of this section, you should navigate into the `smarttvleakage` folder. Unless otherwise specified, you should run all scripts from this directory.


### Emulation: Creating Benchmarks



### Realistic Users
The codebase processes video recordings of Smart TV interactions. We take videos for debugging purposes--the attack strips away the video and uses audio signals alone (e.g., see the file `smarttvsearch. 


Process Recording, Make Spectrogram, Split Keyboard Instances

## String Recovery
