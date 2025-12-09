import warnings

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer.gan_trainer import GANTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    accelerator = Accelerator()

    if config.trainer.device == "auto":
        device = accelerator.device
    else:
        device = torch.device(config.trainer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    optimizer_g = instantiate(
        config.optimizer_g,
        params=model.generator.parameters(),
    )

    disc_params = list(model.mpd.parameters()) + list(model.msd.parameters())
    optimizer_d = instantiate(
        config.optimizer_d,
        params=disc_params,
    )

    lr_scheduler_g = instantiate(config.lr_scheduler_g, optimizer=optimizer_g)
    lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)

    train_loader = dataloaders["train"]

    prepared = accelerator.prepare(
        model,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g,
        lr_scheduler_d,
        train_loader,
    )

    (
        model,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g,
        lr_scheduler_d,
        train_loader,
    ) = prepared

    dataloaders["train"] = train_loader

    epoch_len = config.trainer.get("epoch_len")

    trainer = GANTrainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        accelerator=accelerator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
