from transformers import Trainer
import logging

class MTGTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % self.args.logging_steps == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Step: {self.state.global_step}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        return loss

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        logging.info(f"Evaluation - Epoch {int(self.state.epoch)}")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")
        return metrics
