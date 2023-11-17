# Smart TV Acoustic Keystroke Leakage
This repository contains the code for discovering keystrokes from the audio of Apple and Samsung Smart TVs. This attack was acknowledged by Samsung in their [Bug Bounty Hall of Fame for Smart TVs, Audio, and Displays](https://samsungtvbounty.com/hallOfFame).

This code is the artifact accompanying the conference paper "Acoustic Keystroke Leakage on Smart Televisions," accepted at NDSS 2024. A draft of this paper is included in the repository under the file `acoustic_keystroke_leakage_smart_tv.pdf`. If you use this work in any way, we ask that you cite this paper.

## Overview
This repository has two main portions: audio extraction and string recovery. The phases are split for checkpointing reasons. The audio extraction module is written in Python, and the string recovery is in Java. The folder `smarttvleakage/audio` contains most of the code related to audio extraction. The directory `smarttvleakage/search/smarttvsearch` holds the Java code involving string recovery.

This document describes how to create emulation benchmarks (to test string recovery in isolation), process recordings of users interacting with Smart TVs, and recover strings from the audio's intermediate representation. The code has only been tested on Ubuntu `20.04`. The installation instructions may differ on other systems.

We designed the attack to extract English keystrokes from Apple and Samsung Smart TVs in the United States. We tested and verified our implementation against an A1625 AppleTV with tvOS version 16.3.2 and a model UN55MU6300 Samsung Smart TV running Tizen software version T-KTMAKUC-1310.1. Other TV versions may exhibit differences, e.g., in the layout of the default keyboard.

We include the results of intermediate steps, as well as the dictionary priors, in this [Google Drive](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) folder. We provide the recordings from our user study in this [Box Drive](https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf).

For all Python scripts, you can use the command line option `--help` (e.g., `python make_spectrogram.py --help`) to get a description of each argument. For this reason, we do not enumerate all command line arguments in this `README`.

## Outline
This document contains four main sections: `Setup`, `Audio Extraction`, `String Recovery`, and `Analysis`.

1. `Setup` is required to execute the provided code.
2. `Audio Extraction` outlines how to generate benchmarks and process recordings of users typing on Smart TVs.
3. `String Recovery` implements how the attack guesses strings from processed recordings.
4. `Analysis` visualizes the results.

If you only wish to reproduce the figures in the paper using the precomputed results, you can skip the `Audio Extraction` and `String Recovery` sections.


## Setup

### Python
We recommend configuring the Python portion of this project inside an Anaconda environment. We have tested everything using [Anaconda](https://docs.anaconda.com/free/anaconda/install/) version `23.0.1`. The first step is to create a virtual environment, as shown below (named `smarttv`).
```
conda env create --name smarttv -f environment.yml
```
This command will both create the environment and install the relevant Python packages. You should then activate the environment as shown below. All following operations must be completed within the virtual environment.
```
conda activate smarttv
```
You will then need to install the local `smarttvleakage` package for development via the command below. You must run this from the root directory (e.g., where `setup.py` is).
```
pip install -e .
```
To verify this installation, navigate into the `smarttvleakage` directory and run `python make_spectrogram.py --help`. This program should *not* error and instead should print out the command line arguments.

### Java
The Java code requires an installation of the [Java Runtime Environment](https://ubuntu.com/tutorials/install-jre#1-overview). We use `openjdk` version `17.05`. The Java portion of the project uses `json` parsing, local `SQL` databases, and `junit`. The `jar` files for these libraries are in the `jars` directory. You need to specify both paths to these `jars` and a path to the Java project. To add these references, you must update your `$CLASSPATH$` environment variable. This step is best accomplished by adding the following lines to your `~/.bashrc` file.
```
export SMARTTV_HOME=<PATH-TO-PROJECT-DIRECTORY>
export CLASSPATH=$CLASSPATH:$SMARTTV_HOME/smarttvleakage/search:$SMARTTV_HOME/jars/junit-4.10.jar:$SMARTTV_HOME/jars/json-20220924.jar:$SMARTTV_HOME/jars/sqlite-jdbc-3.36.0.3.jar:.
``` 
The `<PATH-TO-PROJECT-DIRECTORY>` should be the full path to the directory of this repository in your file system. You will need to restart your terminal for this change to take effect. After this change, you can check to ensure the references work by navigating to the `smarttvleakage/search/smarttvsearch` folder and running the following.
```
javac SearchRunner.java
java smarttvsearch.SearchRunner
```
This command should show the following error, which is expected because the above command does *not* provide the required command line arguments.
```
Exception in thread "main" java.lang.IllegalArgumentException: Must provide the (1) --input-file, (2) --output-file, (3) --password-prior, (4) --english-prior, and (5) --zip-prior
```

### Data Sources
For convenience, we list both data sources here.
1. This [Google Drive](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) (link: https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) folder contains the dictionary priors, word lists, and intermediate / final results. You should download this entire folder (about 1 GB of disk space uncompressed). The directory should have the following structure.
```
Smart-TV-Acoustic-Leakage-Supplementary-Materials/
    benchmarks/
    dictionaries/
    user-study/
    word-lists/
```
You should move this folder into the root directory of this repository and rename the folder to `supplementary'. After this action, you should have a directory structure that looks as follows.
```
smart-tv-keyboard-leakage/
    environment.yml
    jars/
    LICENSE
    README.md
    requirements.txt
    setup.py
    smarttvleakage/
    supplementary/
        benchmarks/
        dictionaries/
        user-study/
        word-lists/
```

2. This [Box Drive](https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf) (link: https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf) contains the videos of users typing on Smart TVs. Note that this drive is large, and it can help to start with a single subject (e.g., Subject A). We include the extracted move count sequences for every subject in the Google Drive folder (under `user-study`). When downloading the files, use the `Primary` videos when possible. You should create a folder for each subject within a single directory and place the subject's videos directly in their corresponding folder. The resulting file structure should look as follows. Note that you should place the base `user-study` folder into the root directory of this repository.
```
smart-tv-keyboard-leakage/
    environment.yml
    jars/
    LICENSE
    README.md
    requirements.txt
    setup.py
    smarttvleakage/
    supplementary/
        benchmarks/
        ...
    user-study/
        subject-a/
            appletv_passwords.mp4
            credit_card_details.mp4
            samsung_passwords.mp4
            web_searches.mp4
        subject-b/
            appletv_passwords.mp4
            ...
        ...
```
For the remainder of this document, we refer to the file paths in this directory structure. If you wish to reference other files, during custom experiments, you can change the command line arguments for each file accordingly.

## Audio Extraction
The codebase processes video recordings of Smart TV interactions. We take videos for debugging purposes--the attack strips away the video and uses audio alone (see `smarttvleakage/audio/audio_extractor.py`). This section describes how to execute the audio extraction phase of the attack. Each recording can have multiple interactions with the keyboard (we call each interaction a keyboard *instance*). The output of this phase is a `json` file containing the move count sequences for each identified keyboard instance. A single move count sequence looks as follows.
```
[{"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 512, "end_time": 831, "move_times": [528, 569, 773]}, {"num_moves": 3, "end_sound": "key_select", "num_scrolls": 0, "directions": "any", "start_time": 877, "end_time": 1019, "move_times": [892, 925, 958]},...]
```

We describe two methods for creating these move count sequences.
1. **Emulation:** Creates the move count sequence `json` file algorithmically using the keyboard layout.
2. **Real Recordings:** Extracts move count sequences from the audio of real interactions with Smart TVs.

We include the recordings of users entering passwords, credit card details, and web searches into Apple and Samsung Smart TVs using this [Box Drive](https://uchicago.box.com/s/1td9b0ltk115eg0uyp21d7u2wrdlnjhf). See the section on Data Sources above for more details.

For the remainder of this section, you should navigate into the `smarttvleakage` folder. Unless otherwise specified, you should run all scripts from this directory.

### Emulation: Creating Benchmarks (Optional)
We support two types of benchmarks: `passwords` and `credit cards`. For better reproducibility, we include a list of existing benchmarks in the accompanying [Google Drive folder](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) (under `benchmarks`). Thus, creating new benchmarks is optional. There is randomness in the generation process, so new benchmarks may create slightly different attack results. However, the general patterns should remain the same.

#### Passwords
The `generate_password_benchmark.py` file creates new password benchmarks. The script takes the password list (as a text file) and TV type (either `samsung` or `apple_tv`) as input. The output is a folder of move count sequence files and the corresponding true passwords. We write the outputs in batches of `500` elements (into sub-folders labeled `part_N`). The benchmark selects passwords with special characters, uppercase letters, and numbers. Passwords with such properties are thus *over-represented*. The paper's results come from the `phpbb.txt` password list with `6,000` total passwords. An example of generating a password benchmark is below. The output directory is `benchmarks/samsung_passwords`, and you will have to make the folder `benchmarks` first. The command should take no more than a few minutes.
```
python generate_password_benchmark.py --input-path ../supplementary/word-lists/phpbb.txt --output-folder benchmarks/samung_passwords --max-num-records 6000 --tv-type samsung
```
The script allows the user to optionally supply the keyboard type; the default follows the provided TV. Of interest here is the `abc` keyboard type, which uses a different layout on the Samsung Smart TV.

For Samsung TVs, the script will also classify the keyboard type as either `suggestions` or `standard` (see the `suggestions_model` folder for specifics on this process). Apple TVs and other keyboard layouts are assumed to not have dynamic behavior.

#### Credit Cards
We generate credit card benchmarks using fake details from Visa, Mastercard, and American Express. We use [this site](https://www.creditcardvalidator.org/generator) to get credit card numbers (CCN), expiration dates, and security codes (CVV). We then attach ZIP codes by sampling according to population. Our practical experiments with credit card details only use the Samsung TV. Thus, we only generate credit card benchmarks for this TV type.

The script `generate_credit_card_benchmark.py` creates the benchmark. The program expects input CSV files containing lines of the form `CCN,MM/YY,CVV` (see below). An example of this file is in the Google Drive folder (under `word_lists/credit_cards.csv`).
```
379852449392213,03/26,7747
```
The program also requires a path to a text file containing ZIP codes and populations. This file is at `dictionaries/zip_codes.txt` in the Google Drive folder. An example of executing this command is below. The output format is equivalent to that of password benchmarks. The script should take no more than a few minutes.
```
python generate_credit_card_benchmark.py --input-files ../supplementary/word-lists/credit_cards.csv --zip-code-file ../supplementary/dictionaries/zip_codes.txt --output-folder benchmarks/credit_cards
```
The provided benchmark contains `6,000` credit cards, with `2,000` coming from each of the three providers. The output file format follows that of password benchmarks.

### Real Recordings
We process video recordings of Smart TV interactions in two phases. The first phase strips the audio (`make_spectrogram.py`). The second phase identifies Smart TV sounds, classifies the TV type, and finds keyboard instances (`split_keyboard_instances.py`). We include the recordings from our user study in the Box drive.

The first phase strips the audio from the video file and creates a spectrogram. The script `make_spectrogram.py` handles this process. The output is a `pkl.gz` file containing the raw spectrogram placed in the folder containing the video. The command below shows an example of processing the `samsung_passwords.mp4` file from subject `a`.
```
python make_spectrogram.py --video-path ../user-study/subject-a/samsung_passwords.mp4
```

The second phase determines the TV type, splits the recording into individual instances, and extracts the move count sequences. The file `split_keyboard_instances.py` implements this functionality. Continuing from the example of processing subject `a`, the command below shows how to run this script.
```
python split_keyboard_instances.py --spectrogram-path ../user-study/subject-a/samsung_passwords.pkl.gz
```
The script should print out the number of instances, the sequence type (standard or credit cards), and the TV type (see below).
```
TV Type: SAMSUNG
Sequence Type: STANDARD
Number of splits: 10
```
The result is in a file called `samsung_passwords.json` stored in the same directory. This file contains the serialized move count sequences. To ensure this process was successful, you can compare this file to the same file we have included in the Google Drive folder. The command below makes this comparison (run this from within the `smarttvleakage` folder). The expectation is no difference (i.e., the command prints nothing).
```
diff -w ../user-study/subject-a/samsung_passwords.json ../supplementary/user-study/subject-a/samsung_passwords.json
```
The output file names will change when you change the input video. For instance, if you process the `appletv_passwords.mp4` file, the output files will be named `appletv_passwords.json` and `appletv_passwords_labels.json`. We emphasize that the code will classifies the TV type based on the audio profile and does not use the TV name specified in the file path.

We note that both subjects `g` and `j` have two password recording files for the Samsung TV. This split happened as a result of the subject needing to pause during the experiment. On these users, you can process each video individually. Then, you can merge the move count sequence files using the script `scripts/passwords/merge_password_extractions.py`. The command below shows an example.
```
python merge_password_extractions.py --extracted-paths ../../user-study/subject-g/samsung_passwords_part_0.json ../../user-study/subject-g/samsung_passwords_part_1.json --output-path ../../user-study/subject-g/samsung_passwords.json
```
You should then use the resulting file during the string recovery phase.

If you have access to an AppleTV or a Samsung Smart TV, you can supply your own video recordings to this phase. For Samsung TVs, you may provide any example of typing into the default keyboard (e.g., typing a WiFi password). On Apple TVs, we execute the attack on the keyboard used when entering passwords (e.g., logging into an Apple ID). You should take a video using a camera pointed at the TV. The TV must be audible during the recording. We note that the audio extraction might display some errors due to changes in the recording environment. You will need to list the true typed strings in a file with the name `<experiment-name>_labels.json` placed in the same folder as the video. This `json` file has a single key called `labels` which holds a list of the true typed strings in the order they appear in the recording. See the file `smart-tv-keyboard-leakage/supplementary/user-study/subject-a/samsung_passwords_labels.json` (in the supplementary materials) for an example.

## String Recovery
The folder `smarttvleakage/search/smarttvsearch` contains the code related to string recovery. This code is written in Java for efficiency reasons. Overall, this module uses the extracted move count sequences to discover the likely typed strings.

In the remainder of this section, you should navigate into the `smarttvleakage/search/smarttvsearch` directory. We continue the example of processing Samsung passwords from `subject-a` and assumes the relevant files are in the downloaded Box folder (e.g., `user-study/subject-a/samsung_passwords.json`). You can change the input file for the recovery program by, for example, referencing one of the benchmarks or user study examples in the downloaded Google Drive directory. The first five are required and the remaining are optional.

The recovery process will terminate upon reaching either the correct string or making the maximum number of guesses. The correct strings for each experiment are in the Google Drive folder at paths of the form `supplementary/user-study/subject-a/<experiment-name>_labels.json`. For example, the path for the Samsung passwords for `subject-a` is `supplementary/user-study/subject-a/samsung_passwords_labels.json`. Before executing the string recovery, you must copy the labels into the `user-study/subject-<N>/` directories in the root directory. The recovery program expects the labels file to be in the same folder as the `json` file containing the move count sequences.

The file `SearchRunner.java` contains the entry point into the search process. This program takes the following command line arguments.

1. `--input-file`: Path to the `json` file containing the move count sequences (e.g., `samsung_passwords.json`)
2. `--output-file`: Path to the `json` file in which to save the results. You should name the file `recovered_<INPUT-FILE-NAME>.json` (e.g., `recovered_credit_card_details.json`). For passwords, you should also specify the prior (see argument `3`) in the output file name (e.g., `recovered_samsung_passwords_phpbb.json`).
3. `--password-prior`: Path to the prior distribution of passwords to use during the search. We use either `supplementary/dictionaries/phpbb.db` or `supplementary/dictionaries/rockyou-5gram.db` in our experiments.
4. `--english-prior`: Path to the prior distribution of English words to use during the search. We use `supplementary/dictionaries/wikipedia.db` in our experiments.
5. `--zip-prior`: Path to a text file containing valid ZIP codes. We use `supplementary/dictionaries/zip_codes.txt` in our experiments.
6. `--ignore-directions`: An *optional* flag indicating the search should ignore any inferred directions. You should prefix the output file name with `no_directions_` (e.g., `no_directions_recovered_samsung_passwords_phpbb.json`).
7. `--ignore-suboptimal`: An *optional* flag telling the search to ignore possible suboptimal movements (useful for speeding up large benchmarks).
8. `--force-suggestions`: An *optional* flag to force the search to use keyboards with suggestions. You should name the output file `forced_recovered_<INPUT-FILE-NAME>.json` (e.g., `forced_recovered_web_searches.json`).
9. `--use-exhaustive`: An *optional* flag to get the search to exhaustively enumerate suboptimal paths instead of ordering by timing. When using this flag on credit cards, name the output file `exhaustive.json`.

An example of executing this program is below for the Samsung Passwords typed by `subject-a`. Executing this program can take `5-10` minutes. Again, you can specify move sequences from credit cards, Apple TV passwords, and web searches as the input file. The program will handle the information type accordingly.
```
javac SearchRunner.java
java smarttvsearch.SearchRunner --input-file ../../../user-study/subject-a/samsung_passwords.json --output-file ../../../user-study/subject-a/recovered_samsung_passwords_phpbb.json --password-prior ../../../supplementary/dictionaries/phpbb.db --english-prior ../../../supplementary/dictionaries/wikipedia.db --zip-prior ../../../supplementary/dictionaries/zip_codes.txt
```
The program should find `6 / 10` passwords on this user.

When executing the search on large benchmarks, use the flag `--ignore-suboptimal`. The benchmarks contain no suboptimal moves, and ignoring such paths speeds up the search considerably.

You can use the `shell` scripts `run_password_benchmark.sh` and `run_credit_card_benchmark.sh` to execute the recovery for all benchmark files. You will need to alter the script to point to the correct input and dictionary directories (i.e., change `INPUT_BASE` and `DICTIONARY_BASE`). The files `run_user_passwords.sh`, `run_user_credit_cards.sh` and `run_user_searches.sh` perform the recovery for all users (on passwords, credit cards, and web searches). You will need to change the `USER_BASE` and `DICTIONARY_BASE` variables to point to the folder containing the subject data and prior dictionaries, respectively. Note that each one of these scripts can take `30+` minutes to complete.


## Analysis
The `smarttvleakage/analysis` folder provides a variety of tools to analyze the attack's results. We describe these tools with a particular emphasis on generating the figures shown in the paper. You should navigate to the `smarttvleakage/analysis` folder an run all scripts from this directory.

### Viewing Recovery Results
The scripts `view_password_results.py` and `view_credit_card_results.py` print out the strings recovered by the search procedure. These scripts both take in the recovery `json` file (the output of the search step) and the label `json` file. An example of this script for Samsung passwords is below (assumes the completion of password recovery for `subject-a` in the previous section).
```
python view_password_recovery.py --recovery-file ../../user-study/subject-a/recovered_samsung_passwords_phpbb.json --labels-file ../../user-study/subject-a/samsung_passwords_labels.json 
```
This command prints the following.
```
Found: naarf666 (rank 6), pva81-ph (rank 1), .sagara. (rank 1), 8b7ce7df (rank 1), tutuphpbb (rank 1), williame (rank 4)
Not Found: function84 , p5ych0#7 , chevy_1954 , bubba?51879
Recovery Accuracy: 60.0000% (6 / 10)
```
Note that you can change the `--recovery-file` to any password or web search recovery file. The script also works on benchmarks, although it will not print the exact strings for space reasons.

The command below shows an example of printing the credit card recovery results for `subject-a`. The example assumes you have run the credit card recovery for this subject. Otherwise, replace the input paths with the precomputed results from the Google Drive folder.
```
python view_credit_card_recovery.py --recovery-file ../../user-study/subject-a/recovered_credit_card_details.json --labels-file ../../user-study/subject-a/credit_card_details_labels.json 
```
The printed output looks as follows. A rank of `-1` means `Not Found`.
```
Target: CCN -> 2295229331701537, CVV -> 043, Month -> 07, Year -> 26, ZIP -> 10305
Ranks: CCN -> 11, CVV -> 4, Month -> 1, Year -> 1, ZIP -> 1, Overall -> 251
======
Target: CCN -> 4388910972580132, CVV -> 030, Month -> 02, Year -> 25, ZIP -> 50606
Ranks: CCN -> -1, CVV -> -1, Month -> 1, Year -> 1, ZIP -> 3, Overall -> -1
======
Target: CCN -> 371901290375583, CVV -> 1792, Month -> 07, Year -> 30, ZIP -> 60804
Ranks: CCN -> 33, CVV -> 11, Month -> 1, Year -> 1, ZIP -> 1, Overall -> 2383
======
```

### Attack Results in Emulation
The examples in this section assume the use of the pre-computed results provided in the supplementary materials. You may change the file paths that point to local directories containing results generated from executing the attack on new benchmarks.

#### Credit Cards
The script `benchmark_ccn_recovery.py` creates a plot showing the top-`K` accuracy on the credit card number and full credit card details. When running this program on the provided benchmark results (from the supplementary materials), the resulting plot should match Figure `8` in the paper. Note that if you change the benchmark, the results may differ slightly; the overall trends should remain the same.

The command below shows an example of executing this script. You must provide the folder containing the `part_N` directories.
```
python benchmark_credit_card_recovery.py --benchmark-folder ../../supplementary/benchmarks/credit-cards
```
The script also prints out the top-`K` rates for each provider on both the credit card number and the full details. The top-`10` rates on the full details should match those in the final paragraph of Section `V.B`. The program additionally prints the average fraction of potential guesses that are *valid* credit card numbers. The printed amount should match the `16.26%` value listed in the final paragraph of Section `V.B`. 

#### Passwords
The file `benchmark_password_recovery.py` shows the password recovery results. The script takes in multiple folders, each containing the results for a single TV; the user must also supply the `--tv-types` in the same order as the provided folders. Below is an example of executing this script on the provided results.
```
python benchmark_password_recovery.py --benchmark-folders ../../supplementary/benchmarks/samsung-passwords/ ../../supplementary/benchmarks/appletv-passwords/ --tv-types samsung appletv
```
The program will display a plot which should match Figure `9`.

The script also prints out the accuracy on different password types for both priors (`phpbb` and `rockyou-5gram`). The top-`1` values should match those listed in the second-to-last paragraph in Section `V.C`. Finally, the script also prints the minimum factor by which the attack improves over random guessing for the `rockyou` prior. As listed in the paper (Section `V.C`, Paragraph 3), the improvement is over `330x` for both TVs.

Finally, we include a script which shows the keyboard suggestion model's accuracy on the password benchmark (see the folder `smarttvleakage/suggestions_model` for more details). The file `suggestions_benchmark_accuracy.py` performs this comparison. It also runs a small parameter sweep over different cutoffs to observe the predictor's recall. Note that the command can take a few minutes to complete.
```
python suggestions_benchmark_accuracy.py --benchmark-folder /local/smart-tv-benchmarks/samsung-passwords/
```
The results look as below. The `99.23%` figure matches that of fourth paragraph in Section `V.C`.
```
Observed Accuracy: 99.2333%
Cutoff: 0.50, Accuracy: 98.000%
Cutoff: 0.55, Accuracy: 98.600%
Cutoff: 0.60, Accuracy: 99.000%
Cutoff: 0.65, Accuracy: 99.267%
```

### Attack Results on Users
The examples in this section assume that you are displaying the results from the files in the supplementary materials folder containing the outputs for all users. If you wish to use your own results, there are two options. First, you can run the attack on all users for every data type (this can take some time and requires downloading all videos). Second, you can run the attack on *some* users and copy the results from the remaining using the supplementary materials. In this second case, you should place the results in the `smart-tv-keyboard-leakage/user-study/subject-<N>` folders and replace all references to `../../supplementary/user-study` below with `../../user-study`.

#### Credit Cards
The file `user_ccn_recovery.py` displays the results for the top-`K` accuracy on both credit card numbers and full payment details. The outputs are similar to those from emulation. The command below is an example for how to run this script.
```
python user_credit_card_recovery.py --user-folder ../../supplementary/user-study
```
The resulting plot should match Figure `10` in the paper. The script also prints out the results for each provider, and these numbers should match those listed in the third paragraph of Section `VI.B`.

We further compare the results of searching for credit card details when exhaustively enumerating suboptimal paths. This baseline compares to the attack's current method of ordering suboptimal paths using keystroke timing. To generate these results on your own, run the search on all users with the `--use-exhaustive` flag and name the output files `exhaustive.json` in each subject's folder. The script `compare_suboptimal_modes.py` displays the results for these two strategies. The command below shows an example.
```
python compare_suboptimal_modes.py --user-folder ../../supplementary/user-study
```
The resulting plot should match Figure `13`.

Finally, we find the approximate number of times that users traverse an optimal path between keys when entering credit card numbers. The file `ccn_optimal_paths.py` handles this computation. An example is below.
```
python ccn_optimal_paths.py --user-folder ../../supplementary/user-study
```
The resulting accuracy should match the `89.35%` figure listed in the final paragraph of Section `VI.B`.

#### Passwords
The script `user_password_recovery.py` displays the recovery results for passwords typed by human users. The format of inputs and outputs are similar to those from the benchmark. The command below shows an example.
```
python user_password_recovery.py --user-folder ../../supplementary/user-study --tv-types samsung appletv
```
The resulting plot should match Figure `11` in the paper. The program also prints out three key pieces of information.
1. The `95%` confidence interval in the top-100 password accuracy.
2. The top-`K` accuracy for passwords with different characters. These figures should match those listed in the third paragraph of Section `VI.C`.
3. The factor by which the attack improves over random guessing. The first paragraph of Section `VI.C` lists the improvement of over `100x` with the RockYou prior on the Samsung TV.

We further compare password recovery for the *same* strings typed by different users. The script `password_user_comparison.py` handles this computation. The example below computes the recovery rate across users for each of the `50` unique passwords.
```
python password_user_comparison.py --user-folder ../../supplementary/user-study --tv-type samsung
```
The result looks as below. The `phpbb` rate should match that of the third paragraph in Section `VI.C`.
```
phpbb Prior. Duplicate Recovery Rate: 42.00000\% (21 / 50)
rockyou-5gram Prior. Duplicate Recovery Rate: 8.00000\% (4 / 50)
```

Additionally, we provide a method to find the rate at which users traverse optimal paths when typing passwords. We perform a comparison across TV types (`samsung` and `appletv`). The file `compare_user_keyboard_accuracy.py` implements this comparison.
```
python compare_user_keyboard_accuracy.py --user-folder ../../supplementary/user-study
```
The script prints out the rate at which users traverse an optimal path on both platforms. The result of the above command is below.
```
Samsung Accuracy. Mean -> 85.9436, Std -> 4.2318, Max -> 92.5926, Min -> 80.0000
AppleTV Accuracy. Mean -> 44.3455, Std -> 10.0268, Max -> 54.9296, Min -> 29.5455
```
These mean accuracy values should match those listed in paragraph two of Section `VI.C`.

Finally, we compare the performance discrepancy of the attack with and without direction inference. When generating new results for this comparison, you must run the search both with and without the `--ignore-directions` flag. The file `compare_directions.py` performs the comparison, as shown below.
```
python compare_directions.py --user-folder ../../supplementary/user-study --prior rockyou-5gram
```
The result looks as follows.
```
With: 33, Without: 38, Password: 20340437
With: 54, Without: 64, Password: deanna69!
With: 7, Without: 8, Password: never.di
Num Improved: 19 / 19 (1.00000)
Num Better: 3 / 19 (0.15789)
Total Counts. With: 19, Without: 19
```
The improvement of `3 / 19` matches the description in the third paragraph of Section `VI.E`.

#### Web Searches
The file `user_search_recovery.py` displays the results for recovering web searches on keyboards with suggestions. Below is an example of how to use this script.
```
python user_search_recovery.py --user-folder ../../supplementary/user-study
```
The resulting plot should match Figure `12` from the paper. The script also has an option (`--use-forced`) to use the results when informing the search of a keyboard with suggestions (e.g., using the option `--force-suggestions` during the search).
```
python user_search_recovery.py --user-folder ../../supplementary/user-study --use-forced
```
The paper cites (Section `VI.D`, paragraph two) the top-`10` (`12%`) and top-`100` (`27%`) accuracy when forcing suggestions.

Finally, we analyze the accuracy of the keyboard type classifier on Samsung devices. The script `suggestions_user_accuracy.py` implements this functionality for both passwords and web searches. Below is an example.
```
python suggestions_user_accuracy.py --user-folder ../../supplementary/user-study
```
The script outputs the following.
```
Password Accuracy: 99.0196% (101 / 102)
Searches Accuracy: 42.7083% (41 / 96)
```
The password accuracy should match that of the fourth paragraph in Section `VI.C`. The searches accuracy should be equivalent to that of the second paragraph in Section `VI.D`.
