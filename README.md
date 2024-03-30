# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goals: Make it run, and run fast, on any platform. Document it better. And build it out as a useful toolset for music research generally.

See https://ar5iv.labs.arxiv.org/html/2306.09025 for the July 2023 research paper that accompanied the original CoverHunter repo.

# Requirements

1. Either: 
    1. Apple computer with an Apple M-series chip
    2. Other computer with an Nvidia GPU (including free cloud options like Google Colab)
2. python3 (minimum version 3.10, tested on 3.11) with these libraries:
    1. torch torchaudio (compiled with CUDA or MPS or other GPU support as appropriate for your hardware)
    2. librosa
    3. nnAudio
    4. Optional: tensorboard
3. sox

# Usage

Either:
- Clone this repo or download it to a folder on your computer. Run the following OS command-line commands from that folder. These commands assume you have a Unix/Linux/MacOS environment, but Windows equivalents exist - [see issue #10](https://github.com/alanngnet/CoverHunterMPS/issues/10).

- Or run this project in Google Colab, using this Colab notebook:
https://colab.research.google.com/drive/1HKVT3_0ioRPy7lrKzikGXXGysxZrHpOr?usp=sharing

## Data Preparation

Follow the example of the prepared Covers80 dataset included with the original CoverHunter. Directions here are for using that prepared data. See also the "dataset.txt format" heading below.

1. Download and extract the contents of the `covers80.tgz` file from http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
2. Abandon the 2-level folder structure that came inside the covers80.tgz file, flattening so all the .mp3 files are in the same folder. One way to do this is:
    1. In Terminal, go to the extracted `coversongs` folder as the current directory. Then: 
    2. `cd covers32k && mv */* .; rm -r */`
3. Convert all the provided .mp3 files to .wav format. One way to do this is:
    1. `setopt EXTENDED_GLOB; for f in *.mp3; do sox "$f" -r 16000 "${f%%.mp3}.wav" && rm "$f"; done`
4. Move all these new .wav files to a new folder called `wav_16k` in the project's `data/covers80` folder.
5. You can delete the rest of the downloaded `covers80.tgz` contents.

## Feature Extraction

You must run this before proceeding to the Train step. And you can't run this without first doing the Data Preparation step above.

From the project root folder, run:

`python3 -m tools.extract_csi_features data/covers80/`

Side note: I attempted MPS optimization of CQT feature extraction but failed. The main issue is that PyTorch MPS implementation is very incomplete for many of the tensor operations involved in this CQT implementation. Even the core `fft()` function just became available with PyTorch torch-2.3.0.dev20240222 or higher. See my comments in `extract_csi_features.py` and `cqt.py`.

## Training

CoverHunter includes a prepared configuration to run a training session on the Covers80 dataset located in the 'egs/covers80' subfolder of the project. *Important note:* the default configuration that the CoverHunter authors provided was a nonsense or toy configuration that only demonstrated that you have a working project and environment. It used the same dataset for both training and validation, so by definition it rapidly converged and overfit.

This fork added a train/validate/test data-splitting function in the extract_csi_features tool, along with corresponding new training hyperparameters. Note that CoverHunter used the terms "train / train-sample / dev" for train / validate / test.

Specify the path where the training hyperparameters are available (in this case using the provided example for covers80) and where the model output will go, as the one required command-line parameter:

`python -m tools.train egs/covers80/`

This fork also added an optional `--runid` parameter so you can distinguish your training runs in TensorBoard in case you are experimenting:

`python -m tools.train egs/covers80/ --runid 'first try'`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`

Optionally edit the `hparams.yaml` configuration file in the folder `egs/covers80/config` before starting a training run. If you run into memory limits, start with decreasing the batch size from 64 to 32.

This fork added the hyperparameter `early_stopping_patience` to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

Note: Don't use the `torchrun` launch command offered in original CoverHunter. In the single-computer Apple Silicon context, it is not only irrelevant, it actually slows down performance. In my tests it slowed down tools.train performance by about 20%.

The training script's output consists of checkpoint files and embedding vectors, described below in the "Training checkpoint output" section.

## Evaluation

This script evaluates your trained model by providing mAP and MR1 metrics and an optional t-SNE clustering plot (compare Fig. 3 in the CoverHunter paper).

1. Have a pre-trained CoverHunter model's output checkpoint files available. You only need your best set (typically your highest-numbered one). If you use original CoverHunter's pre-trained model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you rename to, in this example, 'pretrained_model'.
2. Run your query data through `extract_csi_features.py`. In the `hparams.yaml` file for the feature extraction, turn off all augmentation. See `data/covers80_testset/hparams.yaml` for an example configuration to treat covers80 as the query data:<br> `python3 -m tools.extract_csi_features data/covers80_testset`<br>
The important output from that is `full.txt` and the `cqt_feat` subfolder's contents.
3. Run the evaluation script. This example assumes you are using the trained model you created in `egs/covers80` and you want to use all the optional features I added in this fork:<br>
`python3 -m tools.eval_testset egs/covers80 data/covers80_testset/full.txt data/covers80_testset/full.txt -plot_name="egs/covers80/tSNE.png" -dist_name='distmatrix' -test_only_labels='data/covers80/dev-only-song-ids.txt'`

CoverHunter only shared an evaluation example for the case when query and reference data are identical, presumably to do a self-similarity evaluation of the model. But there is an optional 4th parameter for `query_in_ref_path` that would be relevant if query and reference are not identical. See the "query_in_ref" heading below under "Input and Output Files."

The optional `plot_name` argument is a path or just a filename where you want to save the t-SNE plot output. If you provide just a filename, `model_dir` will be used as the path. See example plot below. Note that your query and reference files must be identical to generate a t-SNE plot (to do a self-similarity evaluation).

The optional `test_only_labels` argument is a path to the text file generated by `extract_csi_features.py` if its hyperparameters asked for some song_ids to be reserved exclusively for the test aka "dev" dataset. The t-SNE plot will then mark those for you to see how well your model can cluster classes (song_ids) it has never seen before.

This figure shows the results of training from scratch on the covers80 dataset with a train/val/test split of 8:1:1 and 3 classes (song_ids) reserved exclusively for the test dataset.
![t-SNE plot for Covers80](tSNE-example.png)

The optional `dist_name` argument is a path where you want to save the distance matrix and ref labels so that you can study the results separately, such as perhaps doing custom t-SNE plots, etc.
 
See the "Training checkpoint output" section below for a description of the embeddings saved by the `eval_for_map_with_feat()` function called in this script. They are saved in a new subfolder of the `pretrained_model` folder named `embed_NN_tmp` where NN is the highest-numbered epoch subfolder in the `pretrained_model` folder.

## Coarse-to-Fine Alignment Training

CoverHunter did not include an implementation of the coarse-to-fine alignment training described in the research paper. (Liu Feng confirmed to me that his employer considers it proprietary technology). [See issue #1](https://github.com/alanngnet/CoverHunterMPS/issues/1). But it did include this script which apparently could be useful as part of an implementation we could build ourselves. The command to launch the alignment script that CoverHunter included is:

`python3 -m tools.alignment_for_frame pretrained_model data/covers80/full.txt data/covers80/alignment.txt`

Arguments to pass to the script:
1. Folder containing a pretrained model. For example if you use original CoverHunter's model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you rename to `pretrained_model` at the top level of your project folder. That folder in turn must contain a `pt_model` subfolder that contains the do_000[epoch] and g_000[epoch] checkpoint files.
2. The output file from the feature-extraction script described above. It must include `song_id` key-values for each `utt` (unlike the raw `dataset.txt` file that CoverHunter provided for covers80).
3. The `alignment.txt` file will receive the output of this script.

# Input and Output Files

## Hyperparameters (hparams.yaml)

There are two different hparams.yaml files, each used at different stages. 

1. The one located in the folder you provide on the command line to tools.extract_csi_features.py is used only by that script.

| key | value |
| --- | --- |
|add_noise| Original CoverHunter provided the example of: <div>{<br> &nbsp; `prob`: 0.75,<br> &nbsp; `sr`: 16000,<br> &nbsp; `chunk`: 3,<br> &nbsp; `name`: "cqt_with_asr_noise",<br> &nbsp; `noise_path`: "dataset/asr_as_noise/dataset.txt"<br>}<br>However, the CoverHunter repo did not include whatever might supposed to be in "dataset/asr_as_noise/dataset.txt" file nor does the CoverHunter research paper describe it. If that path does not exist in your project folder structure, then tools.extract_csi_features will just skip the stage of adding noise augmentation. At least for training successfully on Covers80, noise augmentation doesn't seem to be needed.|
| aug_speed_mode | list of ratios used in tools.extract_csi_features for speed augmention of your raw training data. Example: [0.8, 0.9, 1.0, 1.1, 1.2] means use 80%, 90%, 100%, 110%, and 120% speed variants of your original audio data.|
| train-sample_data_split | percent of training data to reserve for validation aka "train-sample" expressed as a fraction of 1. Example for 10%: 0.1 |
| train-sample_unseen | percent of song_ids from training data to reserve exclusively for validation aka "train-sample" expressed as a fraction of 1. Example for 2%: 0.02 |
| test_data_split | percent of training data to reserve for test aka "dev" expressed as a fraction of 1. Example for 10%: 0.1 |
| test_data_unseen | percent of song_ids from training data to reserve exclusively for test aka "dev" expressed as a fraction of 1. Example for 2%: 0.02  |

2. The one located in the "config" subfolder of the path you provide on the command line to tools.train uses all the other parameters listed below during training.

### Data sources

| key | value |
| --- | --- |
| covers80 | Test dataset for model evaluation purposes. "covers80" is the only example provided with the original CoverHunter.<br>Subparameters:<br>`query_path`: "data/covers80/full.txt"<br>`ref_path`: "data/covers80/full.txt"<br>`every_n_epoch_to_dev`: 1 # validate after every n epoch<br>These can apparently be the same path as `train_path` for doing self-similarity evaluation.|
| dev_path | Compare `train_path` and `train_sample_path`. This dataset is used in each epoch to run the same validation calculation as with the `train_sample_path`. But these results are used for the `early_stopping_patience` calculation. Presumably one should include both classes and samples that were excluded from both `train_path` and `train_sample_path`. |
| query_path | TBD: can apparently be the same path as `train_path`. Presumably for use during model evaluation and inference. |
| ref_path | TBD: can apparently be the same path as `train_path`. Presumably for use during model evaluation and inference. |
| train_path | path to a JSON file containing metadata about the data to be used for model training (See full.txt below for details) |
| train_sample_path | path to a JSON file containing metadata about the data to be used for model validation. Compare `dev_path` above. Presumably one should include a balanced distribution of samples that are *not* included in the `train_path` dataset, but do include samples for the classes represented in the `train_path` dataset.(See full.txt below for details) 

### Training parameters
| key | value |
| --- | --- |
| batch_size | Usual "batch size" meaning in the field of machine learning. An important parameter to experiment with. |
| chunk_frame | list of numbers used with `mean_size`. CoverHunter's covers80 config used [1125, 900, 675]. "chunk" references in this training script seem to be the chunks described in the time-domain pooling strategy part of their paper, not the chunks discussed in their coarse-to-fine alignment strategy. See chunk_s. | 
| chunk_s | duration of a `chunk_frame` in seconds. Apparently you are supposed to manually calculate `chunk_s` = `chunk_frame` / frames-per-second * `mean_size`. I'm not sure why the script doesn't just calculate this itself using CQT hop-size to get frames-per-second? |
| cqt: hop_size: | Fine-grained time resolution, measured as duration in seconds of each CQT spectrogram slice of the audio data. CoverHunter's covers80 setting is 0.04 with a comment "1s has 25 frames". 25 frames per second is hard-coded as an assumption into CoverHunter in various places. |
| data_type | "cqt" (default) or "raw" or "mel". Unknown whether CoverHunter actually implemented anything but CQT-based training |
| device | 'mps' or 'cuda', corresponding to your GPU hardware and PyTorch library support. Theoretically 'cpu' could work but untested and probably of no value. |
| early_stopping_patience | how many epochs to wait for validation loss to improve before early stopping |
| mean_size | See `chunk_s` above. An integer used to multiply chunk lengths to define the length of the feature chunks used in many stages of the training process. |
| mode | "random" (default) or "defined". Changes behavior when loading training data in chunks in AudioFeatDataset. "random" described in CoverHunter code as "cut chunk from feat from random start". "defined" described as "cut feat with 'start/chunk_len' info from line"|
| m_per_class | From CoverHunter code comments: "m_per_class must divide batch_size without any remainder" and: "At every iteration, this will return m samples per class. For example, if dataloader's batch-size is 100, and m = 5, then 20 classes with 5 samples iter will be returned." |
| spec_augmentation | spectral(?) augmentation settings, used to generate temporary data augmentation on the fly during training. CoverHunter settings were:<br>`random_erase`:<br> &nbsp; `prob`: 0.5<br> &nbsp; `erase_num`: 4<br>`roll_pitch`:<br> &nbsp; `prob`: 0.5<br> &nbsp; `shift_num`: 12 |

### Model parameters
| key | value |
| --- | --- |
| embed_dim | 128 |
| encoder | # model-encode<br>Subparameters:<br>`attention_dim`: 256 # "the hidden units number of position-wise feed-forward"<br>`output_dims`: 128<br>`num_blocks`: 6 # number of decoder blocks |
| input_dim | 96 |


## dataset.txt

A JSON formatted or tab-delimited key:value text file (see format defined in the utils.py::line_to_dict() function) expected by extract_csi_features.py that describes the training audio data, with one line per audio file.
| key | value |
| --- | --- |
| utt | Unique identifier. Probably an abbreviation for "utterance," borrowed from speech-recognition ML work. Example "cover80_00000000_0_0". In a musical context we should call this a "performance." |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| song | title of the song. Example "A_Whiter_Shade_Of_Pale" The `_add_song_id()` function in extract_csi_features assumes that this string is a unique identifier for the parent cover song (so it can't handle musically distinct songs that happen to have the same title). |
| version | Not used by CoverHunter. Example "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3" |

## full.txt 

full.txt is the JSON-formatted training data catalog for tools.train.py, generated by tools.extract_csi_features. In case you do your own data prep instead of using tools.extract_csi_features, here's the structure of full.txt.

| key | value |
| --- | --- |
| utt | (see dataset.txt) |
| wav | (see dataset.txt) |
| dur_s |(see dataset.txt) |
| song | (see dataset.txt) |
| version | (see dataset.txt) |
| feat | path to the CQT features of this utt stored as .npy array. Example: "data/covers80/cqt_feat/sp_0.8-cover80_00000146_71_0.cqt.npy" |
| feat_len | output of len(np.load(feat)). Example: 9198 |
| song_id | internal, arbitrary unique identifier for the song. This is what teaches the model which utts (performances) are considered by humans to be the "same song." Example: 0 |
| version_id | internal, arbitrary unique identifier for each artificially augmented variant of the original utt (performance). Example: 0 |

Note: Original CoverHunter omitted the unmodified audio by accident due to a logic error at lines 104-112 of tools.extract_csi_features, by unintentionally appending the next value of `sp_utt` to the beginning of `local_data['utt']`. And if and only if the '1.0' member of the aug_speed_mode hyperparameter was not listed first, the result then was not only that the 1.0 variant was omitted, but also a duplicate copy of the 90% variant was created and included in the final output of full.txt in the end, both entries in full.txt pointing to the same cqt.npy file, just with different version_id values. 

That bug didn't prevent successful training, but fixing the bug did, until I discovered that because then, when the model was being fed the intended number of song versions (augmented from 1 to 5 instead of to 4 versions), CoverHunter's preset batch size of 16 became a barrier to success. Increasing the batch size hyperparameter to 32 and larger made a huge difference, resulting in much faster convergence and higher mAP than the original CoverHunter code.

## song_id.map 

Text file crosswalk between "song" (unique identifying string per song) and the "song_id" number arbitrarily assigned to each "song" by the extract_csi_features.py script. Not used by any scripts in this project currently, but definitely useful as a reference for human interpretation of training results.


## Other files generated by extract_csi_features.py

| filename | comments |
|---|---|
| cqt_feat subfolder | Numpy array files of the CQT data for each file listed in full.txt. Needed by train.py |
| data.init.txt | Copy of dataset.txt after sorting by `utt` and de-duping. Not used by train.py |
| dev.txt | A subset of full.txt generated by the `_split_data_by_song_id()` function intended for use by train.py as the `dev` dataset. |
| dev-only-song-ids.txt | Text file listing one song_id per line for each song_id that the train/val/test splitting function held out from train/val to be used exclusively in the test aka "dev" dataset. This file can be used by `eval_testset.py` to mark those samples in the t-SNE plot. |
| full.txt | See above detailed description. Contains the entire dataset you provided in the input file. | 
| song_id.map | Text file, with 2 columns per line, separated by a space, sorted alphabetically by the first column. First column is a distinct "song" string taken from dataset.txt. Second column is the `song_id` value assigned to that "song." |
| sp_aug subfolder | Sox-modified wav speed variants of the raw training .wav files, at the speeds defined in hparams.yaml. Not used by train.py |
| sp_aug.txt | Copy of data.init.txt but with addition of 1 new row for each augmented variant created in sp_aug/*.wav. Not used by train.py. |
| train.txt | A subset of full.txt generated by the `_split_data_by_song_id()` function intended for use by train.py as the `train` dataset. |
| train-sample.txt | A subset of full.txt generated by the `_split_data_by_song_id()` function intended for use by train.py as the `train-sample` dataset. |


Original CoverHunter also generated the following files, but were not used by their published codebase, so I commented out those functions:

| filename | comments |
| song_id_num.map | Text file, not used by train.py, maybe not by anything else? |
| song_name_num.map | Text file, not used by train.py, maybe not by anything else? |


## Training checkpoint output

Using the default configuration, training saves checkpoints after each epoch in the egs/covers80 folder.

The `pt_model` subfolder gets two files per epoch: do_000000NN and g_000000NN where NN=epoch number. The do_ files contain the AdamW optimizer state. The g_ files contain the model's state dictionary. "g" might be an abbreviation for "generator" given that a transformer architecture is involved?

The `eval_for_map_with_feat()` function, called at the end of each epoch, also saves data in a separate new subfolder for each epoch, named epoch_NN_covers80. This in turn gets a `query_embed` subfolder containing the model-generated embeddings for every sample in the training data, plus the embeddings for time-chunked sections of those samples, named with a suffix of ...__start-N.npy where N is the timecode in seconds of where the chunk starts. The saved embeddings are 1-dimensional arrays containing 128 double-precision (float64) values between -1 and 1. The epoch_NN_covers80 folder also gets an accompanying file `query.txt` (with an identical copy as `ref.txt`) which is a text file listing the attributes of every training sample represented in the `query_embed` subfolder, following the same format as described above for `full.txt`.

## query_in_ref

The file you can prepare for the `tools/eval_testset.py` script to pass as the 4th parameter `query_in_ref_path` (CoverHunter did not provide an example file or documentation) assumes:
- JSON or tab-delimited key:value format
- The only line contains a single key "query_in_ref" with a value that is itself a list of tuples, where each tuple represents a mapping between an index in the query input file and an index in the reference input file.
This mapping is only used by the `_generate_dist_matrix()` function. That function explains: "List[(idx, idy), ...], means query[idx] is in ref[idy] so we skip that when computing mAP."

# Code Map

Hand-made visualization of how core functions of this project interact with each other. Also includes additional beginner-friendly or verbose code-commenting that I didn't add to the project code.

https://miro.com/app/board/uXjVNkDkn70=/ 
