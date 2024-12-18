import json

import numpy as np
import yaml
from scipy.stats import pearsonr
from transformers import TrainingArguments, Trainer
import utils
from model import MTGPopularityPredictor
from trainer import MTGTrainer
from dataset import all_decks, used_card_names, tokenized_dataset
from datetime import datetime
from model import MTGPopularityPredictor
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


small_dataset, json_filepath, preserved_information, ignore_lands, include_side_deck, svd_components, banlist_path = utils.parse_config()

with open(json_filepath, "r", encoding="UTF-8") as file:
    json_data = json.load(file)
    json_data = utils.transform_json(json_data)



with open(banlist_path) as f:
    banlist = [line.rstrip() for line in f.readlines()]

with open("../data/SetList.json", "r", encoding="UTF-8") as file:
    set_data = json.load(file)
    release_dates = dict()
    for expansion in set_data['data']:
        release_dates[expansion['code']] = datetime.strptime(expansion['releaseDate'], "%Y-%m-%d")
    release_dates = dict(sorted(release_dates.items(), key=lambda x: x[1]))
    date2set = {v: k for k, v in release_dates.items()}


with open("../data/SetList.json", "r", encoding="UTF-8") as file:
    set_data = json.load(file)
    release_dates = dict()
    for expansion in set_data['data']:
        release_dates[expansion['code']] = datetime.strptime(expansion['releaseDate'], "%Y-%m-%d")
        release_dates = dict(sorted(release_dates.items(), key=lambda x: x[1]))



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pearson_corr, _ = pearsonr(predictions, labels)
    mse = np.mean((predictions - labels) ** 2)
    mae = (predictions - labels).mean()
    return {
        "pearson_correlation": pearson_corr,
        "mse": mse,
        "mae": mae
    }

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



def train_and_save_best():
    training_args = TrainingArguments(
        **config['training_parameters']
    )

    dataset = tokenized_dataset
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test["train"].train_test_split(test_size=0.125)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_test["test"]

    # 4. Modell initialisieren
    model = MTGPopularityPredictor.from_pretrained('roberta-large')

    # 5. Trainer initialisieren
    trainer = MTGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 6. Training starten
    trainer.train()

    # 7. Bestes Modell speichern
    trainer.save_model("best_mtg_predictor")

    # 8. Finale Evaluation auf Test-Set
    test_results = trainer.evaluate(test_dataset)
    with open("../scores.txt", "a") as file:
        entry = f"Trained on {datetime.now().strftime('%Y-%m-%d_%H-%M')}; Metrics:\n"
        for met, val in test_results.items():
            entry += f"{met}: {round(val, 2)}\n"
        file.write(entry)


def save_label_vs_prediction_plot():


    # GPU oder CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dieselbe Train/Test/Val-Aufteilung wie beim Training
    dataset = tokenized_dataset
    train_test = dataset.train_test_split(test_size=0.2, seed=42)  # Seed für Reproduzierbarkeit
    train_val = train_test["train"].train_test_split(test_size=0.125, seed=42)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_test["test"]

    # Kartennamen aus dem ursprünglichen Dataset extrahieren
    card_names = list(used_card_names)  # Annahme: used_card_names existiert bereits aus dataset.py
    test_card_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else range(len(test_dataset))
    test_card_names = [card_names[i] for i in test_card_indices]

    # Tokenizer und Modell laden
    model = MTGPopularityPredictor.from_pretrained('best_mtg_predictor')
    model = model.to(device)
    model.eval()

    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    predictions = []
    labels = []
    processed_cards = 0  # Zähler für verarbeitete Karten

    for batch in tqdm(test_loader, desc="Generating predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['labels']

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = output["logits"].cpu().numpy()

        predictions.extend(pred)
        labels.extend(label.numpy())
        processed_cards += len(pred)

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Plot erstellen
    plt.figure(figsize=(15, 15))

    # Hauptplot mit Kartennamen
    plt.scatter(labels, predictions, alpha=0.3)

    # Füge Kartennamen für ausgewählte Punkte hinzu
    # Wir fügen nur Namen für Karten hinzu, die besonders interessant sind
    for i, (label, pred, name) in enumerate(zip(labels, predictions, test_card_names)):
        # Füge Namen hinzu für:
        # 1. Die am häufigsten gespielten Karten (höchste Labels)
        # 2. Die am stärksten über- oder unterschätzten Karten
        error = abs(pred - label)
        if error > np.percentile(abs(predictions - labels), 95) or label > np.percentile(labels, 95):
            plt.annotate(name, (label, pred),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)

    # Perfekte Vorhersage-Linie
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--', label='Perfekte Vorhersage')

    plt.xlabel('Tatsächliche Werte (log-transformiert)')
    plt.ylabel('Vorhergesagte Werte (log-transformiert)')
    plt.title('Vorhersagen vs. Tatsächliche Werte')

    # De-logarithmierte Werte auf der zweiten y-Achse
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Vorhergesagte Anzahl (nicht transformiert)')
    yticks = ax2.get_yticks()
    ax2.set_yticklabels([f'{int(np.expm1(y))}' for y in yticks])

    plt.tight_layout()
    plt.savefig('../visual_output/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Metriken ausgeben
    mse = np.mean((predictions - labels) ** 2)
    pearson_corr = np.corrcoef(predictions, labels)[0, 1]
    print(f'MSE: {mse:.4f}')
    print(f'Pearson Korrelation: {pearson_corr:.4f}')


if __name__ == '__main__':
    train_and_save_best()
    #save_label_vs_prediction_plot()





