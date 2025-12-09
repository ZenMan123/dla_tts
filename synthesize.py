import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    if config.text:
        import torchaudio
        from huggingface_hub import hf_hub_download

        text_to_mel = instantiate(
            config.transforms.instance_transforms.inference.text_to_mel
        )

        pretrained_path = hf_hub_download(
            repo_id=config.inferencer.get("from_pretrained"), filename="model.pth"
        )

        checkpoint = torch.load(pretrained_path, device, weights_only=False)
        state_dict = (
            checkpoint.get("state_dict")
            if checkpoint.get("state_dict") is not None
            else checkpoint
        )
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            mel = text_to_mel(config.text).to(device)
            audio = model(mel=mel)["audio_pred"]

        torchaudio.save(save_path / "output.wav", audio[0].cpu(), 22050)
        print(f"Saved to {save_path / 'output.wav'}")
    else:
        dataloaders, batch_transforms = get_dataloaders(config, device)

        # get metrics
        metrics = instantiate(config.metrics)

        inferencer = Inferencer(
            model=model,
            config=config,
            device=device,
            dataloaders=dataloaders,
            batch_transforms=batch_transforms,
            save_path=save_path,
            metrics=metrics,
            skip_model_load=False,
        )

        inferencer.run_inference()


if __name__ == "__main__":
    main()
