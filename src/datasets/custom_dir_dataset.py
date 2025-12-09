from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        transcriptions_dir = data_dir / "transcriptions"

        index = []
        for txt_file in sorted(transcriptions_dir.glob("*.txt")):
            utterance_id = txt_file.stem
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            index.append(
                {
                    "text": text,
                    "audio_path": utterance_id,
                }
            )
            print(f"added text {text[:30]} with id {utterance_id}")

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        text = data_dict["text"]
        audio_path = data_dict["audio_path"]

        mel = self.instance_transforms["text_to_mel"](text)

        return {
            "mel": mel,
            "text": text,
            "audio_path": audio_path,
        }

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "text" in entry, "Each item should include 'text'"
            assert "audio_path" in entry, "Each item should include 'audio_path'"
