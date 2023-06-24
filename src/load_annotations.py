import pandas as pd
import numpy as np
import os
import pprint
import json

from collections import namedtuple

WEAT = namedtuple("WEAT", ["name", "targets", "attributes"])

# Original WEAT categories
weat1 = WEAT(
    name="weat1",
    targets=["Flowers", "Insects"],
    attributes=["Pleasant", "Unpleasant"],
)
weat2 = WEAT(
    name="weat2",
    targets=["Instruments", "Weapons"],
    attributes=["Pleasant", "Unpleasant"],
)
weat6 = WEAT(
    name="weat6",
    targets=["Male Names", "Female Names"],
    attributes=["Career", "Family"],
)
weat7 = WEAT(
    name="weat7",
    targets=["Math", "Art"],
    attributes=["Male Terms", "Female Terms"],
)
weat8 = WEAT(
    name="weat8",
    targets=["Science", "Art"],
    attributes=["Male Terms", "Female Terms"],
)
weat9 = WEAT(
    name="weat9",
    targets=["Mental Disease", "Physical Disease"],
    attributes=["Temporary", "Permament"],
)
# New dimensions
weat11 = WEAT(
    name="weat11",
    targets=["Offensive Words", "Respectful Words"],
    attributes=["Female Terms", "Male Terms"],
)
weat12 = WEAT(
    name="weat12",
    targets=["Insult Words", "Disability Words"],
    attributes=["Female Terms", "Male Terms"],
)
weat13 = WEAT(
    name="weat13",
    targets=["LGBTQ+ Words", "Straight Words"],
    attributes=["Prejudice", "Pride"],
)
weat14 = WEAT(
    name="weat14",
    targets=["Educated Terms", "Non-educated Terms"],
    attributes=["Higher Status Words", "Lower Status Words"],
)
weat15 = WEAT(
    name="weat15",
    targets=["Immigrant Terms", "Non-immigrant Terms"],
    attributes=["Disrespectful Words", "Respectful Words 2"],
)
# India specific WEATs
# Gender bias
weat16 = WEAT(
    name="weat16",
    targets=["Stereotypical male adjectives", "Stereotypical female adjectives"],
    attributes=["Male terms", "Female terms"],
)
weat17 = WEAT(
    name="weat17",
    targets=["Male verbs", "Female verbs"],
    attributes=["Male terms", "Female terms"],
)
weat18 = WEAT(
    name="weat18",
    targets=["Male adjectives", "Female adjectives"],
    attributes=["Male terms", "Female terms"],
)
weat19 = WEAT(
    name="weat19",
    targets=["Male entities", "Female entities"],
    attributes=["Male terms", "Female terms"],
)
weat20 = WEAT(
    name="weat20",
    targets=["Male titles", "Female titles"],
    attributes=["Male terms", "Female terms"],
)
# Caste bias
weat21 = WEAT(
    name="weat21",
    targets=["Upper caste occupations", "Lower caste occupations"],
    attributes=["Upper caste names", "Lower caste names"],
)
weat22 = WEAT(
    name="weat22",
    targets=["Upper caste adjectives", "Lower caste adjectives"],
    attributes=["Upper caste names", "Lower caste names"],
)
# Religion bias
weat23 = WEAT(
    name="weat23",
    targets=["Positive adjectives", "Negative adjectives"],
    attributes=["Hindu religiion names", "Muslim religion names"],
)
weat24 = WEAT(
    name="weat24",
    targets=["Positive adjectives", "Negative adjectives"],
    attributes=["Hindu last names", "Muslim last names"],
)
weat25 = WEAT(
    name="weat25",
    targets=["Hindu religiion names", "Muslim religion names"],
    attributes=["Hindu religion", "Muslim religion"],
)
# Urban-rural bias
weat26 = WEAT(
    name="weat26",
    targets=["Upper caste adjectives", "Lower caste adjectives"],
    attributes=["Urban occupations", "Rural occupations"],
)


# This file is in ./src and data is in ./annotations
DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotations/"
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/"
)


def process_annotation_file(filepath, mode="sheet1"):
    if mode == "sheet1":
        return process_sheet_one(filepath)
    # the excel file has 1 sheet, load it into a dataframe
    df = pd.read_excel(filepath, sheet_name=0)

    # remove any occurence of the characters "" in any cell
    df = df.replace(r'"', "", regex=True)

    # also remove "," from the cells
    df = df.replace(r",", "", regex=True)

    # convert everything to string
    df = df.astype(str)

    return df


def process_sheet_one(filepath):
    # the excel file has 2 sheets, load each sheet into a dataframe
    df1 = pd.read_excel(filepath, sheet_name=0)
    df2 = pd.read_excel(filepath, sheet_name=1)

    # remove any occurence of the characters "" in any cell
    df1 = df1.replace(r'"', "", regex=True)
    df2 = df2.replace(r'"', "", regex=True)

    # also remove "," from the cells
    df1 = df1.replace(r",", "", regex=True)
    df2 = df2.replace(r",", "", regex=True)

    # convert everything to string
    df1 = df1.astype(str)
    df2 = df2.astype(str)

    return df1, df2


def get_weat_google_human(df1, weat, lang="tr", column_num=0, folder_name="default"):
    """Get WEAT data from the dataframe."""
    # get the targets and attributes from the WEAT object
    targets = weat.targets
    attributes = weat.attributes

    d = {"targ1": {}, "targ2": {}, "attr1": {}, "attr2": {}}
    d["targ1"]["category"] = targets[0]
    d["targ2"]["category"] = targets[1]
    d["attr1"]["category"] = attributes[0]
    d["attr2"]["category"] = attributes[1]

    for v in d.values():
        v["category"] = v["category"].replace(" ", "")
        if v["category"] == "Art":
            v["category"] = "Arts"
        if v["category"] == "Permament":
            v["category"] = "Permanent"

    categories = list(d.keys())

    for i in range(len(targets)):
        # get the target words for rows where the category is the target
        target_words = (
            df1.loc[df1["category"] == targets[i]].iloc[:, column_num].tolist()
        )
        # add the target words to the dictionary
        d[categories[i]]["examples"] = target_words

    for i in range(len(attributes)):
        # get the attribute words for rows where the category is the attribute
        if attributes[i] == "Permament":
            # match either "Permanent" or "Permament"
            attr_words = (
                df1.loc[
                    (df1["category"] == attributes[i])
                    | (df1["category"] == "Permanent")
                ]
                .iloc[:, column_num]
                .tolist()
            )
        else:
            attr_words = (
                df1.loc[df1["category"] == attributes[i]].iloc[:, column_num].tolist()
            )

        # add the attribute words to the dictionary
        d[categories[i + 2]]["examples"] = attr_words

    # if weat1, then for attr1 and attr2, trim to the first 25 words
    if weat in [weat1, weat2]:
        for i in range(2, 4):
            d[categories[i]]["examples"] = d[categories[i]]["examples"][:25]

    if weat == weat7:
        # keep the firs 8 words of attr1 and attr2
        for i in range(2, 4):
            d[categories[i]]["examples"] = d[categories[i]]["examples"][:8]
        # also remove the last word of targ2
        d[categories[1]]["examples"] = d[categories[1]]["examples"][:-1]

    if weat == weat8:
        # remove the second last word of targ2, but keep the last word
        d[categories[1]]["examples"] = (
            d[categories[1]]["examples"][:-2] + d[categories[1]]["examples"][-1:]
        )

        # targ2["examples"] now has 8 words. i want the last word to be the third
        # word. so that [1,2,3,4,5,6,7,8] becomes [1,2,8,3,4,5,6,7]
        d[categories[1]]["examples"] = (
            d[categories[1]]["examples"][:2]
            + d[categories[1]]["examples"][-1:]
            + d[categories[1]]["examples"][2:-1]
        )

        # attr1["examples"] and attr2["examples"] has 11 words [1,2,3,4,5,6,7,8,9,10,11].
        # i want the words [4,9,10, 11, 8, 5, 7, 6].
        d[categories[2]]["examples"] = (
            d[categories[2]]["examples"][3:4]
            + d[categories[2]]["examples"][8:11]
            + d[categories[2]]["examples"][7:8]
            + d[categories[2]]["examples"][4:5]
            + d[categories[2]]["examples"][6:7]
            + d[categories[2]]["examples"][5:6]
        )
        d[categories[3]]["examples"] = (
            d[categories[3]]["examples"][3:4]
            + d[categories[3]]["examples"][8:11]
            + d[categories[3]]["examples"][7:8]
            + d[categories[3]]["examples"][4:5]
            + d[categories[3]]["examples"][6:7]
            + d[categories[3]]["examples"][5:6]
        )

    # check if the language subfolder exists
    if not os.path.exists(os.path.join(DATA, folder_name)):
        os.mkdir(os.path.join(DATA, folder_name))

    # save the json file
    json.dump(
        d,
        open(os.path.join(DATA, folder_name, weat.name + ".json"), "w"),
        indent=4,
        ensure_ascii=False,
    )

    # return the target and attribute words
    return d


def get_weat_complete(df2, weat, lang="tr", column_num=0, folder_name="default"):
    """Get WEAT data from the dataframe."""
    # get the targets and attributes from the WEAT object
    targets = weat.targets
    attributes = weat.attributes

    d = {"targ1": {}, "targ2": {}, "attr1": {}, "attr2": {}}
    d["targ1"]["category"] = targets[0]
    d["targ2"]["category"] = targets[1]
    d["attr1"]["category"] = attributes[0]
    d["attr2"]["category"] = attributes[1]

    for v in d.values():
        if "india" not in folder_name:
            v["category"] = v["category"].replace(" ", "")
        if v["category"] == "Art":
            v["category"] = "Arts"
        if v["category"] == "Permament":
            v["category"] = "Permanent"
        if weat.name in ["weat25"]:
            d["targ1"]["category"] = "Hindu religion terms"
            d["targ2"]["category"] = "Muslim religion terms"
        if weat.name in ["weat23"]:
            d["attr1"]["category"] = "Hindu religion terms"
            d["attr2"]["category"] = "Muslim religion terms"

    categories = list(d.keys())

    for i in range(len(targets)):
        # get the target words for rows where the category is the target
        target_words = (
            df2.loc[df2["category"] == targets[i]].iloc[:, column_num].tolist()
        )
        # add the target words to the dictionary
        d[categories[i]]["examples"] = target_words

    for i in range(len(attributes)):
        # get the attribute words for rows where the category is the attribute
        if attributes[i] == "Permament":
            # match with either "Permament" or "Permanent"
            attr_words = (
                df2.loc[
                    (df2["category"] == attributes[i])
                    | (df2["category"] == "Permanent")
                ]
                .iloc[:, column_num]
                .tolist()
            )
        else:
            attr_words = (
                df2.loc[df2["category"] == attributes[i]].iloc[:, column_num].tolist()
            )

        # add the attribute words to the dictionary
        d[categories[i + 2]]["examples"] = attr_words

    if weat.name == "weat25":
        # remove the first word of targ1 and targ2
        d[categories[0]]["examples"] = d[categories[0]]["examples"][1:]
        d[categories[1]]["examples"] = d[categories[1]]["examples"][1:]

    # check if the language subfolder exists
    if not os.path.exists(os.path.join(DATA, folder_name)):
        os.mkdir(os.path.join(DATA, folder_name))

    # save the json file
    json.dump(
        d,
        open(os.path.join(DATA, folder_name, weat.name + ".json"), "w"),
        indent=4,
        ensure_ascii=False,
    )

    # return the target and attribute words
    return d

# DEMO:
# To run the script, just uncommen any file name in annotations dictionary and run the script.
if __name__ == "__main__":
    annotations = {
        # "arabic_sheet1.xlsx": "ar",
        # "hindi_sheet1.xlsx": "hi",
        # "italian_sheet1.xlsx": "it",
        # "russian_sheet1.xlsx": "ru",
        # "telugu_sheet1.xlsx": "te",
        # "chinese_sheet1.xlsx": "zh",
        # "filipino_sheet1.xlsx": "tl",
        # "greek_sheet1.xlsx": "el",
        # "korean_sheet1.xlsx": "ko",
        # "turkish_sheet1.xlsx": "tr",
        # "punjabi_sheet1.xlsx": "pa",
        # "urdu_sheet1.xlsx": "ur",
        # "bengali_sheet1.xlsx": "bn",
        # "spanish_sheet1.xlsx": "es",
        # "thai_sheet1.xlsx": "th",
        # "marathi_sheet1.xlsx": "mr",
        # "german_sheet1.xlsx": "de",
        # "danish_sheet1.xlsx": "da",
        # "french_sheet1.xlsx": "fr",
        # "vietnamese_sheet1.xlsx": "vi",
        # "japanese_sheet1.xlsx": "ja",
        # "kurdish_sheet1.xlsx": "ckb",
        # "persian_sheet1.xlsx": "fa",
        # "kurmanji_sheet1.xlsx": "ku",
        # "bengali_sheet2.xlsx": "bn",
        # "hindi_sheet2.xlsx": "hi",
        # "chinese_sheet2.xlsx": "zh",
        # "greek_sheet2.xlsx": "el",
        # "arabic_sheet2.xlsx": "ar",
        # "danish_sheet2.xlsx": "da",
        # "filipino_sheet2.xlsx": "tl",
        # "french_sheet2.xlsx": "fr",
        # "german_sheet2.xlsx": "de",
        # "japanese_sheet2.xlsx": "ja",
        # "korean_sheet2.xlsx": "ko",
        # "kurdish_sheet2.xlsx": "ckb",
        # "persian_sheet2.xlsx": "fa",
        # "punjabi_sheet2.xlsx": "pa",
        # "russian_sheet2.xlsx": "ru",
        # "spanish_sheet2.xlsx": "es",
        # "telugu_sheet2.xlsx": "te",
        # "thai_sheet2.xlsx": "th",
        # "turkish_sheet2.xlsx": "tr",
        # "urdu_sheet2.xlsx": "ur",
        # "vietnamese_sheet2.xlsx": "vi",
        # "italian_sheet2.xlsx": "it", #(column_num=0)
        # "marathi_sheet2.xlsx": "mr", #(column_num=0)
        # "kurmanji_sheet2.xlsx": "ku",
        # "english_sheet2.xlsx": "en", #(column_num=0)
        # "hindi_sheet3.xlsx": "hi",
        # "marathi_sheet3.xlsx": "mr",
        # "punjabi_sheet3.xlsx": "pa",
        # "telugu_sheet3.xlsx": "te",
        # "urdu_sheet3.xlsx": "ur",
        "bengali_sheet3.xlsx": "bn",
    }


    for annotation_file, lang in annotations.items():
        filename = annotation_file
        filepath = os.path.join(DIRECTORY, filename)

        if "sheet1" in filename:
            df1, df2 = process_annotation_file(filepath, mode="sheet1")

            for weat in [weat1, weat2, weat6, weat7, weat8, weat9]:
                get_weat_complete(df2, weat, lang, column_num=0, folder_name=lang + "_all")

            for column_id in range(1, 3):
                folder_name = lang + "_gt" if column_id == 1 else lang + "_human"
                for weat in [weat1, weat2, weat6, weat7, weat8, weat9]:
                    get_weat_google_human(
                        df1, weat, lang, column_num=column_id, folder_name=folder_name
                    )
        elif "sheet2" in filename:
            df1 = process_annotation_file(filepath, mode="sheet2")
            for weat in [weat11, weat12, weat13, weat14, weat15]:
                get_weat_complete(df1, weat, lang, column_num=2, folder_name=lang + "_new")

        elif "sheet3" in filename:
            df1 = process_annotation_file(filepath, mode="sheet3")
            for weat in [weat16, weat17, weat18, weat19, weat20, weat21, weat22, weat23, weat24, weat25, weat26]:
                get_weat_complete(df1, weat, lang, column_num=0, folder_name=lang + "_india")
