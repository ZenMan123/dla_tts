import json
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeechDataset(BaseDataset):
    def __init__(self, data_dir=None, limit=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.wavs_dir = data_dir / "wavs"

        index = self._get_or_load_index()
        super().__init__(index, limit=limit, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        metadata_path = self.data_dir / "metadata.csv"

        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Creating LJSpeech index"):
            parts = line.strip().split("|")
            audio_id = parts[0]
            text = parts[1]
            audio_path = self.wavs_dir / f"{audio_id}.wav"
            index.append(
                {
                    "path": str(audio_path.absolute()),
                    "audio_id": audio_id,
                    "text": text,
                }
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "Each item should include 'path'"
            assert "audio_id" in entry, "Each item should include 'audio_id'"
            assert "text" in entry, "Each item should include 'text'"
