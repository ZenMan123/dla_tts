from typing import Any, Dict, Optional

from src.metrics.tracker import MetricTracker
from src.trainer.trainer import Trainer


class GANTrainer(Trainer):
    def __init__(
        self,
        model,
        criterion,
        metrics: Dict[str, Any],
        optimizer_g,
        optimizer_d,
        lr_scheduler_g: Optional[Any] = None,
        lr_scheduler_d: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer_g,
            lr_scheduler=lr_scheduler_g,
            **kwargs
        )

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d

    def process_batch(
        self, batch: Dict[str, Any], metrics: MetricTracker
    ) -> Dict[str, Any]:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

            self.optimizer_d.zero_grad(set_to_none=True)

            outputs = self.model(**batch)
            batch.update(outputs)

            disc_losses = self.criterion.discriminator_loss(batch)
            batch.update(disc_losses)

            loss_d = disc_losses["loss_discriminator"]
            loss_d.backward()
            self._clip_grad_norm()
            self.optimizer_d.step()

            self.optimizer_g.zero_grad(set_to_none=True)

            outputs = self.model(**batch)
            batch.update(outputs)

            gen_losses = self.criterion.generator_loss(batch)
            batch.update(gen_losses)

            loss_g = gen_losses["loss_generator"]
            loss_g.backward()
            self._clip_grad_norm()
            self.optimizer_g.step()

            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            batch["loss"] = loss_g + loss_d

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch
