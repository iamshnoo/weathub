import json

PATH = "../data"
LANG = "zh"
if LANG == "en":
  data_paths = {
    f"{LANG}": [f"{PATH}/{LANG}/weat1.json", f"{PATH}/{LANG}_new/weat12.json", f"{PATH}/{LANG}_new/weat13.json"],
  }
else:
  data_paths = {
    f"{LANG}": [f"{PATH}/{LANG}_all/weat1.json", f"{PATH}/{LANG}_new/weat12.json", f"{PATH}/{LANG}_new/weat13.json"],
  }

for lang, paths in data_paths.items():
  assert lang == LANG, "Language mismatch"
  weat1_path = paths[0]
  weat12_path = paths[1]
  weat13_path = paths[2]

  with open(weat1_path, "r") as f:
    weat1 = json.load(f)

  with open(weat12_path, "r") as f:
    weat12 = json.load(f)

  with open(weat13_path, "r") as f:
    weat13 = json.load(f)

  weat12["attr1"] = weat1["attr2"]
  weat12["attr2"] = weat1["attr1"]

  weat13["attr1"] = weat1["attr2"]
  weat13["attr2"] = weat1["attr1"]

  # print(weat12)
  # print(weat13)

  # save
  weat12b_path = weat12_path.replace("weat12", "weat12b")
  weat13b_path = weat13_path.replace("weat13", "weat13b")

  with open(weat12b_path, "w") as f:
    json.dump(weat12, f, indent=4, ensure_ascii=False)

  with open(weat13b_path, "w") as f:
    json.dump(weat13, f, indent=4, ensure_ascii=False)
