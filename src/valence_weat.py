# inspired from https://github.com/McGill-NLP/bias-bench/tree/main implementation

import json
import os
import random
import re

import numpy as np
import pandas as pd
from encoding_utils import *
from tqdm import tqdm
from weat import WEAT
import itertools


class WEATExpRunner:
    # defaults to English JSON WEAT files with 100,000 partitions and normalized
    # test statistic with seed 42
    def __init__(
        self,
        encode_function,
        encode_args,
        data_path,
        num_partitions=100000,
        normalize_test_statistic=True,
        seed=42,
        bert_name="distilbert-base-multilingual-cased",
        generate_bootstraps=True,
        bootstrap_size=5000,
        bootstrap_confidence=0.95,
    ):
        assert encode_function is not None, "encode_function cannot be None"
        assert encode_args is not None, "encode_args cannot be None"
        assert data_path is not None, "data_path cannot be None"
        assert encode_args["lang"] in data_path, "data_path must contain lang"

        self.encode_function = encode_function
        self.encode_args = encode_args
        self.encode_method_name = self.find_method_name(data_path, bert_name)
        self.lang = self.encode_args["lang"]
        self.DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = f"{self.DIRECTORY}/{data_path}"
        self.test_ext = ".json"
        self.num_partitions = num_partitions
        self.normalize_test_statistic = normalize_test_statistic
        self.generate_bootstraps = generate_bootstraps
        self.bootstrap_size = bootstrap_size
        self.bootstrap_confidence = bootstrap_confidence
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def find_method_name(self, data_path, bert_name):
        if self.encode_args["embedding_type"] == "static_bert":
            name = f"{bert_name}_encode_words_{self.encode_args['embedding_type']}_encoding_method_bert_layer_0_subword_{self.encode_args['subword_strategy']}_phrase_{self.encode_args['phrase_strategy']}"
        elif self.encode_args["embedding_type"] == "static_fasttext":
            name = f"encode_words_{self.encode_args['embedding_type']}_lang_{self.encode_args['lang']}_phrase_{self.encode_args['phrase_strategy']}"
        elif self.encode_args["embedding_type"] == "contextual":
            name = {
                "1": "bert_last_hidden_state",
                "2": "bert_CLS_token",
                "3": "bert_last-1_layer",
                "4": "bert_average_hidden_states",
            }
            name = f"{bert_name}_encode_words_{self.encode_args['embedding_type']}_encoding_method_{name[self.encode_args['encoding_method']]}_subword_{self.encode_args['subword_strategy']}_phrase_{self.encode_args['phrase_strategy']}"
        else:  # openai_ada
            name = f"encode_words_{self.encode_args['embedding_type']}_lang_{self.encode_args['lang']}_phrase_{self.encode_args['phrase_strategy']}"

        if "gt" in data_path:
            name += "_google_translated"
        if "human" in data_path:
            name += "_human_translated"
        if "all" in data_path:
            name += "_all_translated"
        if "new" in data_path:
            name += "_new_dimensions"
        if "india" in data_path:
            name += "_india_dimensions"

        # additions for valence_weat experiments that are done in this file
        name += "_valence"

        # print(f"Method name: {name}")
        return name

    def sort_tests(self, test):
        """
        Sort tests by number.
        Example : ["weat1", "weat2"] -> [("weat", 1), ("weat", 2)]
        """
        split_test = re.split(r"(\d+)", test)
        return tuple(int(x) if x.isdigit() else x for x in split_test)

    def load_json(self, sent_file):
        """Load data from json files for WEAT."""
        print(f"Loading {sent_file}...")

        with open(sent_file, "r") as file:
            all_data = json.load(file)

        return {k: {"examples": v["examples"]} for k, v in all_data.items()}

    def generate_experiment_id(self, name):
        """Generate an experiment ID."""
        normalization = "_normalized" if self.normalize_test_statistic else ""
        return (
            f"{name}_lang_{self.lang}_{self.encode_method_name}"
            f"_seed_{self.seed}_partitions_{self.num_partitions}{normalization}"
        )

    def generate_tests(self):
        """Generate all tests to run."""

        def is_valid_test(entry):
            return (
                not entry.startswith(".")
                and entry.endswith(self.test_ext)
                # and "b" not in entry
                and "b" in entry
            )

        all_tests = (
            entry[: -len(self.test_ext)]
            for entry in os.listdir(self.data_path)
            if is_valid_test(entry)
        )

        return sorted(all_tests, key=self.sort_tests)

    def run_experiments(self, all_tests, show_plot=False):
        return [
            self.run_single_experiment(test, show_plot)
            for test in tqdm(
                all_tests, desc="Running tests", unit="test", total=len(all_tests)
            )
        ]

    def run_single_experiment(self, test, show_plot=False):
        print(f"Running test {test}")

        # Load the test data.
        encs = self.load_json(os.path.join(self.data_path, f"{test}{self.test_ext}"))
        target1 = encs["targ1"]["examples"]
        target2 = encs["targ2"]["examples"]
        attr1 = encs["attr1"]["examples"]
        attr2 = encs["attr2"]["examples"]

        weat = WEAT(
            encode_function=self.encode_function,
            target_set_1=target1,
            target_set_2=target2,
            attribute_set_1=attr1,
            attribute_set_2=attr2,
            num_partitions=self.num_partitions,
            normalize_test_statistic=self.normalize_test_statistic,
            encode_args=self.encode_args,
            generate_bootstraps=self.generate_bootstraps,
            bootstrap_size=self.bootstrap_size,
            bootstrap_confidence=self.bootstrap_confidence,
        )

        result = {
            "weat_effect_size": format(weat.effect_size, ".3f"),
            "weat_p_value": format(weat.p_value, ".3f"),
            "weat_effect_size_confidence_interval_lower": format(
                weat.effect_size_ci[0], ".3f"
            ),
            "weat_effect_size_confidence_interval_upper": format(
                weat.effect_size_ci[1], ".3f"
            ),
            "weat_test_statistic": format(weat.test_statistic, ".3f"),
            "weat_test_statistic_confidence_interval_lowerr": format(
                weat.test_statistic_ci[0], ".3f"
            ),
            "weat_test_statistic_confidence_interval_upper": format(
                weat.test_statistic_ci[1], ".3f"
            ),
            "experiment_id": self.generate_experiment_id(test),
        }

        if show_plot:
            weat.plot_test_statistic()

        return result

    def run(self, show_plot=False):
        # Run all tests.
        all_tests = self.generate_tests()
        return self.run_experiments(all_tests, show_plot=show_plot)

    def save_results(self, results):
        """Save results to a JSON file."""

        # create results directory if it doesn't exist
        if not os.path.exists(f"{self.DIRECTORY}/results"):
            os.makedirs(f"{self.DIRECTORY}/results")

        # create a subfolder for the language if it doesn't exist
        if not os.path.exists(f"{self.DIRECTORY}/results/{self.lang}"):
            os.makedirs(f"{self.DIRECTORY}/results/{self.lang}")

        csv_name = self.encode_method_name
        # remove any "/"
        csv_name = csv_name.replace("/", "_")
        with open(
            f"{self.DIRECTORY}/results/{self.lang}/all_tests_{csv_name}.json",
            "w",
        ) as file:
            json.dump(results, file, indent=4, ensure_ascii=False)


# Helper method
def save_results_to_df(all_results, mode="sheet1"):
    if mode == "sheet1":
        all_tests = ["weat1", "weat2", "weat6", "weat7", "weat8", "weat9"]
    elif mode == "sheet2":
        all_tests = ["weat11", "weat12", "weat13", "weat14", "weat15"]
    elif mode == "sheet2b":
        all_tests = ["weat12b", "weat13b"]
    elif mode == "sheet3":
        all_tests = ["weat16", "weat17", "weat18", "weat19", "weat20", "weat21", "weat22", "weat23", "weat24", "weat25", "weat26"]
    # save all results to a dataframe
    df = pd.DataFrame(columns=["Type", "DataSource", "Method"] + all_tests)

    # add the results to the df
    for result in all_results:
        row = {}
        if "static" in result[0]["experiment_id"]:
            Type = "static"
        elif "contextual" in result[0]["experiment_id"]:
            Type = "contextual"
        else:
            Type = "openai"

        if "google" in result[0]["experiment_id"]:
            DataSource = "GT"
        elif "human" in result[0]["experiment_id"]:
            DataSource = "HT"
        elif "all" in result[0]["experiment_id"]:
            DataSource = "ALL"
        elif "new" in result[0]["experiment_id"]:
            DataSource = "NEW_HT"
        elif "india" in result[0]["experiment_id"]:
            DataSource = "HT"
        else:
            DataSource = "GT"

        if "fasttext" in result[0]["experiment_id"]:
            Method = "fasttext"
        elif "bert" in result[0]["experiment_id"]:
            Method = "distilbert" #"mono-bert"#"indic-bert" #BERT_NAME
        else:
            Method = "text-embedding-ada-002"

        if "last_hidden_state" in result[0]["experiment_id"]:
            Method += "-last-hidden"
        elif "CLS" in result[0]["experiment_id"]:
            Method += "-CLS"
        elif "average_hidden_states" in result[0]["experiment_id"]:
            Method += "-avg-all-hidden"
        elif "last-1_layer" in result[0]["experiment_id"]:
            Method += "-2nd-last-layer"
        elif "layer_0" in result[0]["experiment_id"]:
            Method += "-layer-0"

        if "subword_first" in result[0]["experiment_id"]:
            Method += "-subword-first"
        elif "subword_average" in result[0]["experiment_id"]:
            Method += "-subword-avg"

        for test in all_tests:
            for weat in result:
                if test in weat["experiment_id"]:
                    row[test] = f'{weat["weat_effect_size"]} ({weat["weat_p_value"]})'
                    # row[
                    #     test
                    # ] = f'{weat["weat_effect_size"]} ({weat["weat_effect_size_confidence_interval_lower"]}, {weat["weat_effect_size_confidence_interval_upper"]})'
                    # f'{weat["weat_effect_size"]} (esize_CI = {weat["weat_effect_size_confidence_interval_lower"]}, {weat["weat_effect_size_confidence_interval_upper"]}) (p : {weat["weat_p_value"]}) (s = {weat["weat_test_statistic"]}) (s_CI = {weat["weat_test_statistic_confidence_interval_lower"]}, {weat["weat_test_statistic_confidence_interval_upper"]})'
        row["Type"] = Type
        row["DataSource"] = DataSource
        row["Method"] = Method

        # add row to df, use pd.concat
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # save the df to a csv
    df = df.sort_values(['Type', 'DataSource'])
    save_path = f"{langs[0]}_results.csv"
    if mode == "sheet2b":
        save_path = f"{langs[0]}_results_valence.csv"
    df.to_csv(save_path, index=False)
    return df


# DEMO :
# To run, change the bert name as needed, set the same bert name in
# encoding_utils.py
# set the languages and data paths as needed and set the lang name in langs
if __name__ == "__main__":
    BERT_NAME = "distilbert-base-multilingual-cased"
    LANG = "zh"
    langs = [f"{LANG}"]
    data_paths = {
      f"{LANG}": [f"data/{LANG}_new"],
    }

    # Method M5 in paper is strategy 5b here
    # Method M1 in paper is strategy 1b here
    # Method M8 in paper is strategy 3b here
    # Method M10 in paper is strategy 0 here
    strategies = {
        # "0" : {
        #     "lang": langs[0],
        #     "embedding_type": "static_fasttext",
        #     "phrase_strategy": "average",
        #     "encoding_method" : None,
        #     "subword_strategy": None,
        # },
        # "1a" : {
        #     "lang": langs[0],
        #     "embedding_type": "static_bert",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "first",
        #     "encoding_method" : "0",
        # },
        # "1b" : {
        #     "lang": langs[0],
        #     "embedding_type": "static_bert",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "average",
        #     "encoding_method" : "0",
        # },
        # "2" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "2",
        #     "phrase_strategy": None,
        #     "subword_strategy": None,
        # },
        # "3a" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "1",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "first",
        # },
        # "3b" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "1",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "average",
        # },
        # "4a" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "3",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "first",
        # },
        # "4b" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "3",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "average",
        # },
        # "5a" : {
        #     "lang": langs[0],
        #     "embedding_type": "contextual",
        #     "encoding_method": "4",
        #     "phrase_strategy": "average",
        #     "subword_strategy": "first",
        # },
        "5b" : {
            "lang": langs[0],
            "embedding_type": "contextual",
            "encoding_method": "4",
            "phrase_strategy": "average",
            "subword_strategy": "average",
        },
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
                print(f"Running WEAT for {args} on {path}...")
                weat_exp_runner = WEATExpRunner(
                    encode_function=encode_words,
                    encode_args={
                        "lang": args_lang,
                        "embedding_type": args_embedding_type,
                        "encoding_method": args_encoding_method,
                        "subword_strategy": args_subword_strategy,
                        "phrase_strategy": args_phrase_strategy,
                    },
                    data_path=path,
                    num_partitions=100000,
                    normalize_test_statistic=True,
                    seed=42,
                    bert_name=BERT_NAME,
                    generate_bootstraps=True,
                    bootstrap_size=5000,
                    bootstrap_confidence=0.95,
                )

                results = weat_exp_runner.run(show_plot=False)
                weat_exp_runner.save_results(results)
                all_results.append(results)
                print("-" * 80)

    df = save_results_to_df(all_results, mode="sheet2b")
    print(df.head())
