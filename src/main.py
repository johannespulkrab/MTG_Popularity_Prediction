import json
from datetime import date, datetime
import numpy as np
import yaml
from scipy.stats import pearsonr

from transformers import TrainingArguments, Trainer
import utils
from model import MTGPopularityPredictor
from trainer import MTGTrainer
from dataset import all_decks, used_card_names, tokenized_dataset



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


small_dataset, json_filepath, preserved_information, ignore_lands, include_side_deck, svd_components = utils.parse_config()

with open(json_filepath, "r", encoding="UTF-8") as file:
    json_data = json.load(file)
    json_data = utils.transform_json(json_data)



with open('../data/modern_banlist_2024-10-29.txt') as f:
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
    return {
        "pearson_correlation": pearson_corr,
        "mse": mse
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



training_args = TrainingArguments(
    output_dir="mtg-popularity-predictor",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    fp16=True,
    ddp_find_unused_parameters=False
)




if __name__ == '__main__':
    #save_time_series(total_counts, id2card, card2id, json_data, used_card_names)
    dataset = tokenized_dataset
    #Train/Val/Test Split
    train_test = dataset.train_test_split(test_size=0.2)
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
    print(test_results)


