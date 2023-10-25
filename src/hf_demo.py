from datasets import load_dataset
from weat import WEAT
from encoding_utils import encode_words

dataset = load_dataset("iamshnoo/WEATHub", cache_dir="../src/data_cache")

example = dataset["original_weat"][0]

target_set_1 = example["targ1.examples"]
target_set_2 = example["targ2.examples"]
attribute_set_1 = example["attr1.examples"]
attribute_set_2 = example["attr2.examples"]

# corresponding to method M5 used in main paper
# currently using distilbert-base-multilingual-cased
# if you need to use different models, line 61 would change in encoding_utils.py
# TODO: do this in a more elegant way
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

print()
print("Effect size:", weat.effect_size)
print("P value :", weat.p_value)
print("-" * 50)
