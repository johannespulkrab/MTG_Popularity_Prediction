from datetime import date, datetime
from matplotlib import pyplot as plt
import yaml
import json
from copy import copy
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error



with open("../data/SetList.json", "r", encoding="UTF-8") as file:
    set_data = json.load(file)
    release_dates = dict()
    for expansion in set_data['data']:
        release_dates[expansion['code']] = datetime.strptime(expansion['releaseDate'], "%Y-%m-%d")
    release_dates = dict(sorted(release_dates.items(), key=lambda x: x[1]))
    date2set = {v: k for k, v in release_dates.items()}

def parse_config():
    with open('../src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    small_dataset = config['dataset']['small_dataset']
    json_filepath = config['dataset']['json_filepath']
    banlist_path = config['dataset']['banlist_path']
    preserved_information = config['card_features']['preserved_information']
    ignore_lands = config['dataset']['ignore_lands']
    include_side_deck = config['dataset']['include_side_deck']
    svd_components = config['model_parameters']['svd_components']
    return small_dataset, json_filepath, preserved_information, ignore_lands, include_side_deck, svd_components, banlist_path

small_dataset, json_filepath, preserved_information, ignore_lands, include_side_deck, svd_components, banlist_path = parse_config()


def transform_json(file):
    new_json = dict()
    for card in file['data']:
        if file['data'][card][0].get('asciiName'):
            new_json[file['data'][card][0]['asciiName']] = dict()
            for info in preserved_information:
                if file['data'][card][0].get(info):
                    new_json[file['data'][card][0]['asciiName']][info] = file['data'][card][0][info]

        else:
            new_json[card] = dict()
            for info in preserved_information:
                if file['data'][card][0].get(info):
                    new_json[card][info] = file['data'][card][0][info]
                if len(file['data'][card]) > 1:
                    if file['data'][card][1].get(info):
                        new_json[card]['<BACK> ' + info] = file['data'][card][1][info]
    for key in copy(new_json):
        if ' // ' in key and key.split(' // ')[0] == key.split(' // ')[1]:
            new_json[key.split(' // ')[0]] = new_json[key]
            del new_json[key]
    return new_json




def save_time_series(total_counts, id2card, card2id, json_data, used_card_names):
    """Erstellt ein Zeitreihendiagramm der Karten nach ihrem ersten Release"""
    if small_dataset:
        print("Warning: Using function save_time_series for small dataset.")
    first_release_sets = {json_data.get(card, {}).get('firstPrinting', '###'): 0 for card in used_card_names}
    for idx, count in enumerate(total_counts):
        card = id2card[idx]
        first_release = json_data.get(card, {}).get('firstPrinting', '###')
        if first_release == 'OAFC':
            print(card2id[idx])
        first_release_sets[first_release] += int(count)

    nr_of_cards_of_unknown_first_printings = first_release_sets.pop('###')

    first_release_sets = dict(sorted(first_release_sets.items(), key=lambda x: release_dates[x[0]]))
    expansions = list(first_release_sets.keys())
    counts = list(first_release_sets.values())
    expansions = [f"{exp}-{release_dates[exp].strftime('%Y-%m')}" for exp in expansions]

    plt.figure(figsize=(14, 7))
    plt.bar(expansions, counts, color='skyblue'), plt.xlabel('Release Date'), plt.ylabel('Number of Cards')
    plt.title('Number of Cards by First Release Date'), plt.xticks(expansions, expansions, rotation=90, ha='center', fontsize=8)
    plt.text(0.01, 0.99, f'Total Cards: {int(sum(total_counts))}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.01, 0.95, f'Ignored Cards: {nr_of_cards_of_unknown_first_printings}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.01, 0.91, date.today().strftime('%d %B, %Y'), transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout(), plt.savefig('../visual_output/first_release_time_series.png'), plt.close()


def evaluate_model(model, dataloader, device="cuda"):
    model.eval()  # Modell in Evaluierungsmodus setzen
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():  # Deaktiviert den Gradientenabstieg
        for batch in dataloader:
            # Daten auf das richtige Device verschieben
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, labels)
            predictions = outputs["logits"]
            loss = outputs["loss"]

            # Sammle Vorhersagen und Labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Sammle Loss
            total_loss += loss.item()
            num_batches += 1

    # Konvertiere zu Arrays für einfachere Berechnung
    predictions_array = np.array(all_predictions)
    labels_array = np.array(all_labels)

    # Berechne Metriken
    metrics = {
        "avg_loss": total_loss / num_batches,
        "mse": mean_squared_error(labels_array, predictions_array),
        "rmse": np.sqrt(mean_squared_error(labels_array, predictions_array)),
        "mae": mean_absolute_error(labels_array, predictions_array)
    }

    return metrics



