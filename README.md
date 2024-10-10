# BOOK2PPT
BOOK2PPT is a slide generation system that specialises in storybook slide generation. It is an extension of [DOC2PPT](https://doc2ppt.github.io/) which utilises hierarchical sequential model architectures to extract relevant information from source papers to produce generated slide decks. 

BOOK2PPT is developed using three main Jupyter notebooks:
1. `training_experiments`: It used for training experiments for Progress Tracker and Object Placer.
2. `generating_stories`: It used for generating the dataset of paper-slide pairs of BookSum, test slide deck generations with visual generators, and evaluating pipeline variations.
3. `component evaluation`: It used for evaluating text quality and visual relevance; also used for processing results of human surveys.

Required Python libraries are listed in `requirements.txt`. Python version `3.9.19` has been used for this development.

Alongside these Jupyter notebooks are code files that consist of the Progress tracker and Object Placer architecture:
1. `dataset`: This is code file helps load the dataset to the Progress Tracker model
2. `model`: This is the architecture of the Progress Tracker. `model_w_dropout` is a Progress Tracker variant implementing dropout.
3. `mlp_layout`: This is the architecture of the Object Placer. `mlp_layout_dropout` is a Object Placer variant implementing dropout and `mlp_layout_bias` is a Object Placer variant with bias.

## Subdirectories
### Build_dataset
It contains files that build the dataset of paper-slide pairs from [BookSum](https://arxiv.org/abs/2105.08209). 

`chapter_summary_aligned_train_split.jsonl.gathered` is the file containing the raw text of BookSum novels. It has been processed and extracted using instructions found in the [BookSum code repository](https://github.com/salesforce/booksum).

### Books
It contains PKL files of paper-slide pairs for novels of BookSum. There is a total of 20 books with an overall chapter count of 201.

### Data
This folder contains PKL files of paper-slide pairs downloaded from the [DOC2PPT repository](https://drive.google.com/drive/folders/1s2zJ04WZYifZhotRCXpk4OGtCHWXuM0b) for academic papers of the original DOC2PPT dataset. 

**NOTE: Due to the large size of the PKL files, only `record_human.pkl` is retained in this repository for evaluation. The remaining files can be downloaded from `data_processed.tar.gz` via the link provided and should be stored in the `v1.0` subfolder.**

Furthermore, it contains json files in subfolder `v1.0` used to split up both the original dataset and BookSum dataset into training, validation, and testing sets:
- `train_val_test`: This is the original partitions for DOC2PPT papers.
- `train_val_test_2`: Similar to the first json file, but only papers from 14 conferences out of 19 are included. 
- `book`: This is a generated json file for partitions of novel chapters. Distribution of chapters are randomised.

### Training_scripts
This folder contains scripts used for training experimentations. It contains the following subfolders:
- `transfer_learning_progress_tracker`: They contain training scripts for fine-tuning the Progress Tracker on the BookSum dataset.
- `obj_placer_experiment_1`: They contain training scripts for training the Object Placer with the original dataset (academic papers).
- `obj_placer_experiment_2`: They contain training scripts for training and fine-tuning the Object Placer with the BookSum dataset (novels).

It also contains the following documents:
- `main`: This is the provided Progress Tracker training code from the [DOC2PPT paper](https://doc2ppt.github.io/)
- `make_json_split`: It used for spliting the BookSum dataset into training, validation, and testing sets.

### Models
This folder contains the final models for each training variation from fine-tuning the Progress Tracker, training the Object Placer with the academic papers (Experiment 1), and training/fine-tuning the Object Placer with the novels (Experiment 2). 

It also contains `model_hse-tf.pt` which is from [DOC2PPT code repository](https://doc2ppt.github.io/). It is the pretrained weights of the original Progress Tracker.

### Generated_test_slides
This folder contains generated slide decks for three randomly chosen novel chapters. Each chapter has two slide decks: 
- Slide deck with AnimateDiff-generated visuals
- Slide deck with StableDiff-generated visuals

Futhermore, this folder also stores additional information about these slide decks. They include:
- Records for number of slides dedicated for each section for each deck
- Prompts used for visual generation