# Global Voices, Local Biases: Socio-Cultural Prejudices across Languages

This repository contains code for our paper accepted at EMNLP 2023.

The dataset developed in this paper is available in this repository and also on
HuggingFace at this [link](https://huggingface.co/datasets/iamshnoo/WEATHub).
Refer to the HuggingFace README for more details on the dataset format for the hub.

<p align="center">
  <img src="assets/dalle3_weathub.png" width="250" height="250">
</p>

## Requirements - External libraries

Clone the repository and create a virtual environment with the following
libraries from pypi and a python version >= 3.6 to execute all the files with
full functionality.

<details>
  <summary>Click me</summary>

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
</details>

## Minimal example

Refer to ```src/hf_demo.py``` file for a minimal example of how to use the dataset
from huggingface.

```python
from datasets import load_dataset
from weat import WEAT
from encoding_utils import encode_words

dataset = load_dataset("iamshnoo/WEATHub")

example = dataset["original_weat"][0]

target_set_1 = example["targ1.examples"]
target_set_2 = example["targ2.examples"]
attribute_set_1 = example["attr1.examples"]
attribute_set_2 = example["attr2.examples"]

# method M5 from main paper, using DistilmBERT embeddings
args = {
    "lang": example["language"],
    "embedding_type": "contextual",
    "encoding_method": "4",
    "phrase_strategy": "average",
    "subword_strategy": "average",
}

weat = WEAT(
    encode_function=encode_words,
    target_set_1=target_set_1,
    target_set_2=target_set_2,
    attribute_set_1=attribute_set_1,
    attribute_set_2=attribute_set_2,
    num_partitions=100000,
    normalize_test_statistic=True,
    encode_args=args,
)

print("Effect size : ", weat.effect_size)
print("p value : ", weat.p_value)
```

## Reproduction steps

The code is contained in the ```src``` directory.

<details>

  <summary>Click me</summary>

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
    ```final_results/valence```.

</details>

## Results

Results for all experiments referred to in the paper are given in the
```final_results``` folder. It includes csv files organized into subfolders, and
also corresponding auto-generated latex table versions of those csv files.

<details>

  <summary>Click me</summary>
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
      ├── hf_demo.py
      ├── load_annotations.py
      ├── run_weat.py
      ├── secret.txt
      └── weat.py
  ```
</details>
