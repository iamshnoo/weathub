# Global Voices, Local Biases: Socio-Cultural Prejudices across Languages

This repository contains code for our submission to EMNLP 2023. The paper will
be presented at the main conference in Singapore.

The dataset used in this paper is available in this repository and also on
HuggingFace at this [link](https://huggingface.co/datasets/iamshnoo/WEATHub).

![weathub-dalle-img](assets/dalle3_weathub.png)

## Requirements - External libraries

Clone the repository and create a virtual environment with the following
libraries from pypi and a python version >= 3.6 to execute all the files with
full functionality.

```bash
numpy
pandas
matplotlib
seaborn
tqdm
fasttext
transformers
torch
openai
scikit-learn
scipy
```

## Reproduction steps

The code is contained in the ```src``` directory.

- ```load_annotations.py``` loads data from annotations folder and processes it
  to remove spaces and other issues before saving it to json files in the ```data``` folder.
- ```weat.py``` defines a class for the WEAT test. It also includes an example of
  how to use the class.
- ```encoding_utils.py``` defines different types of encoding methods. This
  assumes that fasttext is installed for downloading and using fasttext models,
  and transformers is installed for downloading and using BERT models and openAI
  for using the paid Ada API. Note that, to use the ADA option, you need to have
  an API key from OpenAI stored in a ```secrets.txt``` file in the src folder.
- ```run_weat.py``` gives a very efficient way to call the WEAT class with the
  corresponding encoding utils for a given language and save the results in a
  csv. It includes an example usage. It can be run as ```python
  run_weat.py```. This is the main file to be run to reproduce the results.
- ```compare_embeddings.py``` is the file where we perform the bias sensitivity
  analysis mentioned in our paper.
- ```load_valence.py``` creates the valence experiments mentioned by 2 out of 3
  reviewers and ```valence_weat.py``` runs them. Results are found in
  ```final_results/valence```. Currently, all results for M5 (main paper results
  ) are included.

## Results

Results for all experiments referred to in the paper are given in the
```final_results``` folder. It includes csv files organized into subfolders, and
also corresponding auto-generated latex table versions of those csv files.

The main structure of the repository is as follows :

```bash
.
├── __init__.py
├── annotations
│   ├── ...
├── data
│   ├── ar_all
│   │   ├── ...
│   ├── ar_gt
│   │   ├── ...
│   ├── ar_human
│   │   ├── ...
│   ├── ar_new
│   │   ├── ...
│   ...
│   ├── zh_all
│   │   ├── ...
│   ├── zh_gt
│   │   ├── ...
│   ├── zh_human
│   │   ├── ...
│   └── zh_new
│       ├── ...
├── ft_embeddings
│   ├── cc.en.300.bin
│   ├── ...
├── *.egg-info
├── results
│   ├── ar
│   │   ├── ...
│   ├── consolidated
│   │   ├── ...
│   ...
│   └── zh
│       ├── ...
├── setup.py
└── src
    ├── __init__.py
    ├── compare_embeddings.py
    ├── encoding_utils.py
    ├── load_annotations.py
    ├── run_weat.py
    ├── secret.txt
    └── weat.py
```
