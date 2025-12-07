from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    audio_list = [item["audio"].squeeze(0) for item in dataset_items]
    result_batch["audio"] = pad_sequence(audio_list, batch_first=True)

    mel_list = [item["mel"].squeeze(0).transpose(0, 1) for item in dataset_items]
    result_batch["mel"] = pad_sequence(mel_list, batch_first=True).transpose(1, 2)

    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    return result_batch
