# Smart TV Acoustic Keystroke Leakage
This repository contains the code for discovering keystrokes from the audio of Apple and Samsung Smart TVs.

## Overview
This repository has two main portions: audio extraction and string recovery. The phases are split for checkpointing reasons. The audio extraction module is written in Python, and the string recovery is in Java. The folder `smarttvleakage/audio` contains most of the code related to audio extraction. The directory `smarttvleakage/search/smarttvsearch` holds the Java code involving string recovery.

This document describes how to create emulation benchmarks (to test string recovery in isolation), process recordings of users interacting with Smart TVs, and recover strings from the audio's intermediate representation. The code has only been tested on Ubuntu `20.04`. The installation instructions may differ on other systems.

We include the results of intermediate steps, as well as the dictionary priors, in this ![Google Drive](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) folder. In the remainder of this document, we use the term `<PATH-TO-GDRIVE>` to refer to the path holding the location of the Google Drive folder when downloaded locally.

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
The codebase processes video recordings of Smart TV interactions. We take videos for debugging purposes--the attack strips away the video and uses audio signals alone (e.g., see the file `smarttvleakage/audio/audio_extractor.py`). This section describes how to execute the audio extraction phase of the attack. Each holding can hold multiple interactions with the keyboard (i.e., keyboard instances). The output of this phase is a `json` file containing the move count sequences for each identified keyboard instance. A single move count sequence looks as follows.
```
[{"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 512, "end_time": 831, "move_times": [528, 569, 773]}, {"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 877, "end_time": 1019, "move_times": [892, 925, 958]},...]
```

We describe two methods for creating these move count sequences.
1. **Emulation:** Creates the move count sequence `json` file algorithmically using the keyboard layout.
2. **Real Recordings:** Extracts move count sequences from the audio of real interactions with Smart TVs.

To facilitate reproducibility, we have shared the recordings of users entering passwords, credit card details, and web searches into Apple and Samsung Smart TVs using this ![Box Drive](https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf). Note that this drive is large, and it can help to start with a single subject (e.g., Subject A). We include the extracted move count sequences for every subject in the Google Drive folder if it is infeasible to run the attack on every file.

For the remainder of this section, you should navigate into the `smarttvleakage` folder. Unless otherwise specified, you should run all scripts from this directory.


### Emulation: Creating Benchmarks
We support two types of benchmarks: `passwords` and `credit cards`. For better reproducibility, we include a list of existing benchmarks in the accompanying ![Google Drive folder](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) (see the folder `benchmarks` in the drive). Thus, creating new benchmarks is optional. The instructions in this section create new benchmarks for each type. We note that there is randomness in the generation process, so new benchmarks may create slightly different attack results.

#### Passwords
The `generate_password_benchmark.py` file creates new password benchmarks. The script takes the password list (as a text file) and TV type as input. The output is a folder of move count sequence files and the corresponding true passwords. We write the outputs in batches of `500` elements (into folders labeled `part_N`). The benchmark forcefully selects passwords with special characters, uppercase letters, and numbers. Passwords with such properties are thus *over-represented* in the benchmark. The provided password benchmark uses the `phpbb.txt` password list with `6,000` total passwords. An example of generating a password benchmark is below. The command should take no more than a few minutes.
```
python generate_password_benchmark.py --input-path <PATH-TO-GDRIVE>/word-lists/phpbb.txt --output-folder benchmarks --max-num-records 6000 --tv-type samsung
```
The script allows the user to optionally supply the keyboard type; the default follows the provided TV. Of interest here is the `abc` keyboard type, which uses a different layout on the Samsung Smart TV.

For Samsung TVs, the script will also classify the keyboard type as either `suggestions` or `password` (see the `suggestions_model` folder for specifics on this process). Apple TVs and other keyboard layouts are assumed to hold passwords.

#### Credit Cards
We generate credit card benchmarks using fake details from Visa, Mastercard, and American Express. We use ![this site](https://www.creditcardvalidator.org/generator) to get credit card numbers (CCN), expiration dates, and security codes (CVV). We then attach ZIP codes by sampling according to population.

The script `generate_credit_card_benchmark.py` creates the benchmark. The program expects input CSV files containing lines of the form `CCN,MM/YY,CVV` (see below). An example of this file is in the Google Drive folder (under `word_lists/credit_cards.csv`).
```
379852449392213,03/26,7747
```
The program also requires a path to a text file containing ZIP codes and populations. This file is at `dictionaries/zip_codes.txt` in the Google Drive folder. An example of executing this command is below. The script should take no more than a few minutes.
```
python generate_credit_card_benchmark.py --input-files <PATH-TO-GDRIVE>/word-lists/credit_cards.csv --zip-code-file <PATH-TO-GDRIVE>/dictionaries/zip_codes.txt --output-folder <OUTPUT-FOLDER>
```
The provided benchmark contains `6,000` credit cards, with `2,000` coming from each of the three providers. The output file format follows that of password benchmarks. We only test credit cards on the Samsung TV. This specification is hard-coded into the script.

### Real Recordings
We process video recordings of Smart TV interactions in two phases. The first phase strips the audio (`make_spectrogram.py`). The second phase identifies Smart TV sounds, classifies the TV type, and finds keyboard instances (`split_keyboard_instances.py`).

We include the recordings from our user study in the Box drive. We use the variable `<PATH-TO-BOX>` to refer to the location of the Box drive downloaded locally.

The first phase strips the audio from the video file and creates a spectrogram. The script `make_spectrogram.py` handles this process. The output is a `pkl.gz` file containing the raw spectrogram placed in the folder containing the video. The command below shows an example of processing the `samsung_passwords.mp4` file from subject `a`.
```
python make_spectrogram.py --video-path <PATH-TO-BOX>/subject-a/samsung_passwords.mp4
```

The second phase determines the TV type, splits the recording into individual instances, and extracts the move count sequences. The file `split_keyboard_instances.py` implements this functionality. Continuing from the example of processing subject `a`, the command below shows how to run this script.
```
python split_keyboard_instances.py --spectrogram-path <PATH-TO-BOX>/subject-a/samsung_passwords.pkl.gz
```
The script should print out that it finds `10` keyboard instances. The result is in two files stored in the same directory: `samsung_passwords.json` and `samsung_passwords_labels.json`. The first file contains the serialized move count sequences, and the second contains the actual passwords. To ensure this process was successful, you can compare this file to the same file we have included in the Google Drive folder. The command below makes this comparison.
```
diff -w <PATH-TO-BOX>/subject-a/samsung_passwords.json <PATH-TO-GDRIVE>/user-study/subject-a/samsung_passwords.json
```
Note that the output file names will change when you change the input video. For instance, if you process the `appletv_passwords.mp4` file, the output files will be named `appletv_passwords`.

If you have access to an AppleTV or a Samsung Smart TV, you can supply your own video recordings to this phase. We note that the audio extraction might display some errors due to changes in the recording environment.


## String Recovery




