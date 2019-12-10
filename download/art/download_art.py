import json, shutil, os, sys, random, csv

import urllib.request
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

class CardDownloader:
    colors_mapping = {"W": "White",
              "U": "Blue",
              "B": "Black",
              "R": "Red",
              "G": "Green",
              "C": "Colorless"}

    types_mapping = {"Enchantment":"Enchantment",
             "Creature":"Creature",
             "Artifact":"Artifact",
             "Instant":"InstantSorcery",
             "Sorcery":"InstantSorcery",
             "Land":"Land",
             "Planeswalker":"Planeswalker"}

    face_types = {0 : "front",
                  1 : "back"}

    def __init__(self, number_of_cards = 0):
        with open(CardDownloader.get_absolute_path("scryfall-artwork-cards.json")) as reader:
            self.cards = json.load(reader)

        self.labelled_cards = {}

        self.make_directories()

        self.cards[:] = [card for card in self.cards if self.valid_card(card)]
        
        if number_of_cards:
            num_downloads = number_of_cards
        else:
            num_downloads = len(self.cards)
        
        self.setup_class_names()
        self.mlb = MultiLabelBinarizer(self.class_names)

        # Download how many cards we want
        for index, card in tqdm(enumerate(self.cards), total=num_downloads):
            if number_of_cards:
                if index == number_of_cards:
                    break
            self.download_card(card)

        # Create csvs
        self.create_csv()

    def setup_class_names(self):
        colors_class_names = [class_name for real_name, class_name in self.colors_mapping.items()]
        types_class_names = [class_name for real_name, class_name in self.types_mapping.items()]
        types_class_names = list(set(types_class_names))

        self.class_names = colors_class_names
        self.class_names.extend(types_class_names)
        self.class_names.sort()

    def create_csv(self):
        with open(CardDownloader.get_absolute_path("images.csv"), "w") as writer:
            csv_writer = csv.writer(writer)
            csv_writer.writerow(["Filename", *self.class_names])
            card_filenames = [filename for filename in self.labelled_cards.keys()]
            random.shuffle(card_filenames)
            for card_filename in card_filenames:
                csv_writer.writerow([card_filename, *self.labelled_cards[card_filename]])

    def download_card(self, card):
        for face in self.get_card_details(card):
            self.download_face(*face)

    def download_face(self, name, colors, card_types, url):
        if len(colors) == 1 and len(card_types) == 1:
            self.download_from_url(name, url)
            all_labels = colors
            all_labels.extend(card_types)
            self.labelled_cards[f"{name}.jpg"] = self.mlb.fit_transform([all_labels])[0]

    @sleep_and_retry
    @limits(calls=10, period=1)
    def download_from_url(self, name, url):
        download_filename = CardDownloader.get_absolute_path(f"images/{name}.jpg")
        urllib.request.urlretrieve(url, download_filename)

    def get_card_details(self, card):
        """ Returns card details in form: name, colors, card_types, url """
        if card["layout"] == "transform":
            faces = []
            for index, face in enumerate(card["card_faces"]):
                face_name = self.face_types[index]
                name = f'{card["set"]}_{card["collector_number"]}_{face_name}'
                faces.append(self.get_face_details(name, face))
            return faces
        else:
            assert(card["layout"] in ["normal", "token", "leveler", "adventure"])
            name = f'{card["set"]}_{card["collector_number"]}'
            return [self.get_face_details(name, card)]

    def get_face_details(self, name, face):
        if "color_identity" in face:
            colors = self.get_card_colors(face["color_identity"])
        else:
            colors = self.get_card_colors(face["colors"])
        card_types = self.get_card_types(face["type_line"])
        image_url = face["image_uris"]["art_crop"]

        return name, colors, card_types, image_url

    def get_card_colors(self, colors):
        if colors:
            translated_colors = []
            for color in colors:
                translated_colors.append(self.colors_mapping[color]) 
            return translated_colors
        else:
            return [self.colors_mapping["C"]]

    def valid_type_line(self, type_line):
        for card_type in type_line.split(" —")[0].split(" "):
            for valid_type in self.types_mapping:
                if card_type == valid_type:
                    return True
        return False

    def get_card_types(self, type_line):
        card_types = []
        for card_type in type_line.split("//")[0].split(" —")[0].split(" "):
            if card_type in self.types_mapping:
                card_types.append(self.types_mapping[card_type])
        assert(card_types)
        return card_types

    def valid_card(self, card):
        if card["layout"] not in ["transform", "adventure", "normal", "token", "leveler"]:
            return False
        if card["set_type"] == "funny":
            return False
        if card["lang"] != "en":
            return False
        if not self.valid_type_line(card["type_line"]):
            return False
        return True

    def make_directories(self):
        if not os.path.isdir("images"):
            os.mkdir("images")

    def get_absolute_path(filename):
        return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}/{filename}"



CardDownloader()
