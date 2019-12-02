import json, shutil, os

import urllib.request
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

class CardDownloader:
    types = {"Enchantment":"Enchantment",
             "Creature":"Creature",
             "Artifact":"Artifact",
             "Instant":"InstantSorcery",
             "Sorcery":"InstantSorcery",
             "Land":"Land",
             "Planeswalker":"Planeswalker"}

    colors = {"W": "White",
              "U": "Blue",
              "B": "Black",
              "R": "Red",
              "G": "Green",
              "C": "Colorless"}

    face_types = {0 : "front",
                  1 : "back"}

    def __init__(self, number_of_cards = 0):
        with open("scryfall-artwork-cards.json") as reader:
            self.cards = json.load(reader)

        self.make_directories()

        self.cards[:] = [card for card in self.cards if self.valid_card(card)]
        
        if number_of_cards:
            num_downloads = number_of_cards
        else:
            num_downloads = len(self.cards)
        
        # Download how many cards we want
        for index, card in tqdm(enumerate(self.cards), total=num_downloads):
            if number_of_cards:
                if index == number_of_cards:
                    break
            self.download_card(card)

    @sleep_and_retry
    @limits(calls=10, period=1)
    def download_card(self, card):
        for face in self.get_card_details(card):
            self.download_from_url(*face)

    def download_from_url(self, name, colors, card_types, url):
        first_download_filename = f"type/{card_types[0]}/{name}.jpg"
        urllib.request.urlretrieve(url, first_download_filename)
        for card_type in card_types[1:]:
            shutil.copy2(first_download_filename, f"type/{card_type}/{name}.jpg")

        for color in colors:
            shutil.copy2(first_download_filename, f"color/{color}/{name}.jpg")

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
                translated_colors.append(self.colors[color]) 
            return translated_colors
        else:
            return [self.colors["C"]]

    def valid_type_line(self, type_line):
        for card_type in type_line.split(" —")[0].split(" "):
            for valid_type in self.types:
                if card_type == valid_type:
                    return True
        return False

    def get_card_types(self, type_line):
        card_types = []
        for card_type in type_line.split("//")[0].split(" —")[0].split(" "):
            if card_type in self.types:
                card_types.append(self.types[card_type])
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
        if not os.path.isdir("type"):
            os.mkdir("type")
        for card_type, translation in self.types.items():
            if not os.path.isdir(f"type/{translation}"):
                os.mkdir(f"type/{translation}")

        if not os.path.isdir("color"):
            os.mkdir("color")
        for color, translation in self.colors.items():
            if not os.path.isdir(f"color/{translation}"):
                os.mkdir(f"color/{translation}")


CardDownloader()
