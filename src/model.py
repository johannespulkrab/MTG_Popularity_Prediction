from transformers import RobertaPreTrainedModel, RobertaModel, TrainingArguments, Trainer, RobertaTokenizer
import torch.nn as nn


class MTGPopularityPredictor(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs[0][:, 0, :]
        predictions = self.regression_head(pooled_output).squeeze(-1)

        output = {"logits": predictions}

        # Berechne den Loss nur, wenn Labels vorhanden sind
        if labels is not None:
            loss_fct = nn.MSELoss()
            output["loss"] = loss_fct(predictions, labels)

        return output