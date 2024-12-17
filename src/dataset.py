from transformers import RobertaTokenizer
import numpy as np
from datasets import Dataset
import json
import utils
import unicodedata
import inflect
import os
from collections import Counter
import re
from tqdm import tqdm

with open('../data/modern_banlist_2024-10-29.txt') as f:
    banlist = [line.rstrip() for line in f.readlines()]

small_dataset, json_filepath, preserved_information, ignore_lands, include_side_deck, svd_components = utils.parse_config()

with open(json_filepath, "r", encoding="UTF-8") as file:
    json_data = json.load(file)
    json_data = utils.transform_json(json_data)



def prepare_mtg_dataset():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    texts = []
    labels = []

    for card_name in card2id.keys():
        if card_name not in json_data:
            continue

        card_data = json_data[card_name]

        # Text zusammenbauen
        text = f"Name: {card_name} "
        if 'manaCost' in card_data:
            text += f"Mana: {card_data['manaCost']} "
        if 'type' in card_data:
            text += f"Type: {card_data['type']} "
        if 'text' in card_data:
            text += f"Text: {card_data['text']}"
        if 'BACK:manaCost' in card_data:
            text += f"Back Mana: {card_data['BACK:manaCost']}"
        if 'BACK:type' in card_data:
            text += f"Back Type: {card_data['BACK:type']}"
        if 'BACK:text' in card_data:
            text += f"Back Text: {card_data['BACK:text']}"


        # Magic-Symbole ersetzen
        text = text.replace("{T}", "tap")
        text = text.replace("{W}", "white mana")
        text = text.replace("{U}", "blue mana")
        text = text.replace("{B}", "black mana")
        text = text.replace("{R}", "red mana")
        text = text.replace("{G}", "green mana")

        # Labels vorbereiten
        count = float(total_counts[card2id[card_name]])  # Als Float konvertieren
        normalized_count = float(np.log1p(count))  # log(1+x) Transformation

        texts.append(text)
        labels.append(normalized_count)

    # Dataset erstellen
    dataset = Dataset.from_dict({
        "text": texts,
        "labels": labels
    })

    # Tokenizer-Funktion
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    # Dataset tokenisieren
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # Behalte die labels!
    )

    return tokenized_dataset


class Deck:
    def __init__(self, title, main_deck, side_deck=None):
        self.title = title
        self.main_deck = main_deck
        self.side_deck = side_deck
        self.error_messages = list()

        #normalize card names
        self.main_deck = [unicodedata.normalize('NFKD', card).encode('ascii', 'ignore').decode('ascii') for card in main_deck]
        self.side_deck = [unicodedata.normalize('NFKD', card).encode('ascii', 'ignore').decode('ascii') for card in side_deck]


        #rename dual-named cards
        card_lookup = {card.split(" //")[0]: card for card in json_data.keys() if " //" in card}
        for target in [self.main_deck, self.side_deck]:
            for index, deck_card in enumerate(target):
                target[index] = re.sub(r"\b/\b", " // ", deck_card) #fix card names with only single slash
                if deck_card in card_lookup:
                    target[index] = card_lookup[deck_card]


        if len(self.main_deck) >= 60 <= 100:
            self.ready = True
        else:
            self.ready = False


        #check for banned cards
        self.banned_cards = [card for card in self.main_deck + self.side_deck if card in banlist]


        #check for errors
        main_counts, side_counts = Counter(main_deck), Counter(side_deck)
        for card, count in main_counts.items():
            if count > 4:
                self.more_cards_allowed_str = [f" deck can have up to {inflect.engine().number_to_words(num)} cards named {json_data[card]['name']}" for num in range(5, 100)]
                if not 'Basic' in json_data[card].get('supertypes', [] ) and not f"A deck can have any number of cards named {json_data[card]['name']}" in json_data[card].get('text', "") and not any([x in json_data[card].get('text', "") for x in self.more_cards_allowed_str]):
                    self.error_messages += [f"The card '{card}' is contained more than four times."]
        self.unknown_cards = set()
        for card in self.main_deck + self.side_deck:
            if card not in json_data.keys() and card not in banlist:
                self.unknown_cards.add(card)
        self.error_messages += [f"Card '{card}' not found." for card in self.unknown_cards]
        for message in self.error_messages:
            print(f"\033[31mWarning: {message}\033[0m")


    @classmethod
    def from_txt(cls, filepath):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        main_decklist, side_decklist = list(), list()
        after_blank_line = False
        for line in lines:
            if not line:
                after_blank_line = True
                continue
            if not after_blank_line:
                number, title = int(*line.split()[:1]), " ".join(line.split()[1:])
                for _ in range(number):
                    main_decklist.append(title)
            else:
                number, title = int(*line.split()[:1]), " ".join(line.split()[1:])
                for _ in range(number):
                    side_decklist.append(title)
        return cls(filepath.split("/")[-1][:-4], main_decklist, side_decklist)


# Lade Decks
data_dir = os.listdir('../data/internet_decks/')[:1000] if small_dataset else os.listdir('../data/internet_decks/')
all_decks = [Deck.from_txt(f'../data/internet_decks/{deck}') for deck in tqdm(list(data_dir), desc="Loading decks") if deck.endswith('.txt')]
all_decks += [Deck.from_txt(f'../data/selfmade_decks/{deck}') for deck in os.listdir('../data/selfmade_decks/') if deck.endswith('.txt')]

# Filtere Decks
filtered_decks = []
for deck in all_decks:
    if deck.banned_cards:
        continue
    if not all(card in json_data for card in (deck.main_deck + deck.side_deck)):
        continue
    filtered_decks.append(deck)
all_decks = filtered_decks


# Sammle verwendete Kartennamen
used_card_names = set().union(*(deck.main_deck for deck in all_decks))
if include_side_deck:
    used_card_names.update(*(deck.side_deck for deck in all_decks))

# Ignoriere Länder falls gewünscht
if ignore_lands:
    for deck in all_decks:
        deck.main_deck = [card for card in deck.main_deck if
                          'Land' not in json_data.get(card, {}).get('types', [])]
        deck.side_deck = [card for card in deck.side_deck if
                          'Land' not in json_data.get(card, {}).get('types', [])]


# Erstelle Mappings
card2id = {card: i for i, card in enumerate(used_card_names)}
id2card = {i: card for i, card in enumerate(used_card_names)}
deck2id = {deck.title: i for i, deck in enumerate(all_decks)}
id2deck = {i: deck.title for i, deck in enumerate(all_decks)}

# Erstelle Design-Matrix
design_matrix = np.zeros((len(all_decks), len(used_card_names)))
for i, deck in enumerate(all_decks):
    for card in deck.main_deck:
        design_matrix[i, card2id[card]] += 1


# Berechne total_counts und cooc_X
total_counts = np.maximum(np.sum(design_matrix, axis=0), 1)
cooc_X = design_matrix.T @ design_matrix - np.diag(total_counts)
