import csv
import scrython

import urllib.request
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

class CardDownloader:
    def __init__(self, number_of_cards = 0):
        self.processed_cards = {}

        with open("cards.csv") as reader:
            csv_reader = csv.DictReader(reader)
            # Get total cards to set completion bar
            total_cards = 0
            for line in csv_reader:
                total_cards += 1
            # Reset reader to first line and remove header
            reader.seek(0)
            reader.readline()
            # Download how many cards we want
            for index, line in tqdm(enumerate(csv_reader), total=total_cards):
                if number_of_cards:
                    if index == number_of_cards:
                        break
                self.process_line(line)

    @sleep_and_retry
    @limits(calls=5, period=1)
    def process_line(self, line): 
        if line["name"] not in self.processed_cards:
            self.processed_cards[line["name"]] = {}
            self.processed_cards[line["name"]]["scryfall_ids"] = []

        if line["scryfallId"] in self.processed_cards[line["name"]]["scryfall_ids"]:
            return

        self.processed_cards[line["name"]]["scryfall_ids"].append(line["scryfallId"])

        num_of_samples = len(self.processed_cards[line["name"]]["scryfall_ids"])
        face_name, image_url = self.get_name_and_url(line["name"], line["scryfallId"])

        self.download_from_url(face_name, num_of_samples, image_url)

    def download_from_url(self, name, index, url):
        urllib.request.urlretrieve(url, f"scryfall_data/{name}.{index}.jpg")

    def get_name_and_url(self, name, scryfall_id):
        try:
            card = scrython.cards.Id(id=scryfall_id)
            if card.layout() == "transform":
                for face in card.card_faces():
                    if face["name"] == name:
                        return name, face["image_uris"]["small"]
            else:
                name = name.replace("//", "$")
                return name, card.image_uris()["small"]
        except Exception:
            print(scryfall_id)


CardDownloader()
