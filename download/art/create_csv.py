import csv, random
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

import os, sys
def full_path(filename):
    return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}/{filename}"


colors = Path(full_path("color"))
color_images = {}
for color in colors.glob("*"):
    for filename in Path(color).glob("*"):
        if filename.name not in color_images:
            color_images[filename.name] = []
        color_images[filename.name].append(color.name)

color_filenames = [filename for filename in color_images.keys()]
random.shuffle(color_filenames)

color_mlb = MultiLabelBinarizer(["Black", "Blue", "Colorless", "Green", "Red", "White"])

with open(full_path("colors.csv"), "w") as writer:
    csv_writer = csv.writer(writer)
    for filename in color_filenames:
        csv_writer.writerow([filename, *color_mlb.fit_transform([color_images[filename]])[0]])


types = Path(full_path("type"))
type_images = {}
for type_name in types.glob("*"):
    for filename in Path(type_name).glob("*"):
        if filename.name not in type_images:
            type_images[filename.name] = []
        type_images[filename.name].append(type_name.name)

type_filenames = [filename for filename in type_images.keys()]
random.shuffle(type_filenames)

type_mlb = MultiLabelBinarizer(["Artifact", "Creature", "Enchantment", "InstantSorcery", "Land", "Planeswalker"])
with open(full_path("types.csv"), "w") as writer:
    csv_writer = csv.writer(writer)
    for filename in type_filenames:
        csv_writer.writerow([filename, *type_mlb.fit_transform([type_images[filename]])[0]])
