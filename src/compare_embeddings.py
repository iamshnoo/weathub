import itertools
import json
import math
import os
import pickle
import random
import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from encoding_utils import *
from run_weat import WEATExpRunner
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from tqdm import tqdm
from weat import WEAT

METHOD = "cosine"

# this method calculates the variance in the pairwise distances
# between all the word embeddings from a given embedding strategy
def embedding_variance(embeddings, method=METHOD):
    pairwise_distances = pdist(embeddings, method)
    return np.var(pairwise_distances)

# To run this script, you need to do the folllowing:
# Uncomment a language:path pair, and write the language name in the list
# langs
# Uncomment model name in bert_name
if __name__ == "__main__":

    BERT_NAME = "distilbert-base-multilingual-cased"
    langs = ["fa"]
    data_paths = {
        # "en": ["data/en"],
        # "el": ["data/el_all"],
        # "zh": ["data/zh_all"],
        # "ko": ["data/ko_all"],
        # "tr": ["data/tr_all"],
        # "tl": ["data/tl_all"],
        # "te": ["data/te_all"],
        # "ru": ["data/ru_all"],
        # "ar": ["data/ar_all"],
        # "it": ["data/it_all"],
        # "hi": ["data/hi_all"],
        # "bn": ["data/bn_all"],
        # "es": ["data/es_all"],
        # "mr": ["data/mr_all"],
        # "pa": ["data/pa_all"],
        # "th": ["data/th_all"],
        # "ur": ["data/ur_all"],
        # "vi": ["data/vi_all"],
        # "ja": ["data/ja_all"],
        # "de": ["data/de_all"],
        # "da" : ["data/da_all"],
        # "fr": ["data/fr_all"],
        # "ckb": ["data/ckb_all"],
        "fa": ["data/fa_all"],
    }

    strategies = {
        "0" : {
            "lang": langs[0],
            "embedding_type": "static_fasttext",
            "phrase_strategy": "average",
            "encoding_method" : None,
            "subword_strategy": None,
        },
        "1a": {
            "lang": langs[0],
            "embedding_type": "static_bert",
            "phrase_strategy": "average",
            "subword_strategy": "first",
            "encoding_method": "0",
        },
        "1b": {
            "lang": langs[0],
            "embedding_type": "static_bert",
            "phrase_strategy": "average",
            "subword_strategy": "average",
            "encoding_method": "0",
        },
        "2": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "2",
            "phrase_strategy": None,
            "subword_strategy": None,
        },
        "3a": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "1",
            "phrase_strategy": "average",
            "subword_strategy": "first",
        },
        "3b": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "1",
            "phrase_strategy": "average",
            "subword_strategy": "average",
        },
        "4a": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "3",
            "phrase_strategy": "average",
            "subword_strategy": "first",
        },
        "4b": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "3",
            "phrase_strategy": "average",
            "subword_strategy": "average",
        },
        "5a": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "4",
            "phrase_strategy": "average",
            "subword_strategy": "first",
        },
        "5b": {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "4",
            "phrase_strategy": "average",
            "subword_strategy": "average",
        },
    }

    strategy_id_to_name = {
        "0": "fasttext",
        "1a": "bert_0_sws_first",
        "1b": "bert_0_sws_avg",
        "2":  "bert_last_CLS",
        "3a": "bert_last_sws_first",
        "3b": "bert_last_sws_avg",
        "4a": "bert_2nd_last_sws_first",
        "4b": "bert_2nd_last_sws_avg",
        "5a": "bert_all_sws_first",
        "5b": "bert_all_sws_avg",
    }

    all_results = []
    for strategy_id, args in strategies.items():
        print(f"Running strategy {strategy_id}")
        for lang, paths in data_paths.items():
            args_lang = args["lang"]
            if args_lang != lang:
                break
            args_embedding_type = args["embedding_type"]
            args_phrase_strategy = args["phrase_strategy"]
            args_subword_strategy = args["subword_strategy"]
            args_encoding_method = args["encoding_method"]
            for path in paths:
                arg = {
                    "lang": args_lang,
                    "embedding_type": args_embedding_type,
                    "encoding_method": args_encoding_method,
                    "subword_strategy": args_subword_strategy,
                    "phrase_strategy": args_phrase_strategy,
                }
                weat_exp_runner = WEATExpRunner(
                    encode_function=encode_words,
                    encode_args=arg,
                    data_path=path,
                    num_partitions=100000,
                    normalize_test_statistic=True,
                    seed=42,
                    bert_name=BERT_NAME,
                    generate_bootstraps=True,
                    bootstrap_size=5000,
                    bootstrap_confidence=0.95,
                )
                tests = weat_exp_runner.generate_tests()
                for test in tests:
                    print(f"Running {test}")

                    encs = weat_exp_runner.load_json(
                        os.path.join(
                            weat_exp_runner.data_path,
                            f"{test}{weat_exp_runner.test_ext}",
                        )
                    )
                    target1 = encs["targ1"]["examples"]
                    target2 = encs["targ2"]["examples"]
                    attr1 = encs["attr1"]["examples"]
                    attr2 = encs["attr2"]["examples"]
                    X = encode_words(target1, arg)
                    Y = encode_words(target2, arg)
                    A = encode_words(attr1, arg)
                    B = encode_words(attr2, arg)
                    var_X = embedding_variance(np.array(X))
                    var_Y = embedding_variance(np.array(Y))
                    var_A = embedding_variance(np.array(A))
                    var_B = embedding_variance(np.array(B))
                    # check if var_X is nan
                    if np.isnan(var_X):
                        var_X = "NaN"
                    if np.isnan(var_Y):
                        var_Y = "NaN"
                    if np.isnan(var_A):
                        var_A = "NaN"
                    if np.isnan(var_B):
                        var_B = "NaN"
                    # Compute and store pairwise distance variance
                    store = {
                        "strategy": strategy_id_to_name[strategy_id],
                        "test": test,
                        "target1": var_X,
                        "target2": var_Y,
                        "attr1": var_A,
                        "attr2": var_B,
                    }
                    all_results.append(store)
                    print("-" * 80)

    df = pd.DataFrame(all_results)
    print(df.head())
    df = df.sort_values(by=["test", "strategy"])
    df.to_csv(f"fasttext_cosine_variances_{langs[0]}.csv", index=False)
