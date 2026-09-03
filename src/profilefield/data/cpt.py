"""CPT continuity adapter that preserves GeoVAE-SGS preprocessing behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from profilefield.baselines.geovae_sgs import import_baseline_core

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CPTLayerFold:
    layer_id: int
    train_profiles_ns: FloatArray
    train_profiles_raw: FloatArray
    train_coordinates: FloatArray
    train_ids: tuple[str, ...]
    test_profiles_ns: FloatArray
    test_profiles_raw: FloatArray
    test_coordinates: FloatArray
    test_ids: tuple[str, ...]
    normal_score: Any


class CPTAdapter:
    def __init__(self, project_root: str | Path, profile_points: int = 174) -> None:
        self.project_root = Path(project_root).resolve()
        self.repository = self.project_root / "third_party/geovae-sgs"
        self.profile_points = profile_points
        self.core = import_baseline_core(self.project_root)
        self.core.DEBUG = False
        self.core.USE_AMP = False
        self.core.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta_path = self.repository / "Sample_data/meta_Projekt.csv"
        data_root = self.repository / "Sample_data/csv"
        self.meta, self.interfaces, self.num_layers = self.core.parse_meta(str(meta_path))
        self.cpt_map = self.core.load_all_cpts(str(data_root), self.meta)

    def published_rounds(self) -> list[tuple[int, tuple[str, ...]]]:
        frame = pd.read_csv(
            self.repository / "validations/reviewer_analysis/geovae_gp_reviewer_rounds.csv"
        )
        return [
            (
                int(row["round"]),
                tuple(item.strip() for item in str(row["holdout_ids"]).split(",")),
            )
            for _, row in frame.iterrows()
        ]

    def prepare_fold(self, holdout_ids: tuple[str, ...]) -> dict[int, CPTLayerFold]:
        holdout = set(holdout_ids)
        if not holdout <= set(self.meta):
            raise KeyError(f"Unknown holdout CPT IDs: {sorted(holdout - set(self.meta))}")
        train_meta = {key: value for key, value in self.meta.items() if key not in holdout}
        train_map = {key: value for key, value in self.cpt_map.items() if key not in holdout}
        test_meta = {key: value for key, value in self.meta.items() if key in holdout}
        test_map = {key: value for key, value in self.cpt_map.items() if key in holdout}
        train_ns, train_raw, train_xy, normal_scores, train_ids = self.core.build_layer_datasets(
            train_meta, self.interfaces, self.num_layers, train_map, self.profile_points
        )
        _, test_raw, test_xy, _, test_ids = self.core.build_layer_datasets(
            test_meta, self.interfaces, self.num_layers, test_map, self.profile_points
        )
        layers: dict[int, CPTLayerFold] = {}
        for layer in range(1, self.num_layers + 1):
            if not train_raw[layer] or not test_raw[layer]:
                continue
            raw_test = np.vstack(test_raw[layer]).astype(np.float64)
            layers[layer] = CPTLayerFold(
                layer_id=layer,
                train_profiles_ns=np.asarray(train_ns[layer], dtype=np.float64),
                train_profiles_raw=np.vstack(train_raw[layer]).astype(np.float64),
                train_coordinates=np.asarray(train_xy[layer], dtype=np.float64),
                train_ids=tuple(str(value) for value in train_ids[layer]),
                test_profiles_ns=np.asarray(
                    normal_scores[layer].forward(raw_test), dtype=np.float64
                ),
                test_profiles_raw=raw_test,
                test_coordinates=np.asarray(test_xy[layer], dtype=np.float64),
                test_ids=tuple(str(value) for value in test_ids[layer]),
                normal_score=normal_scores[layer],
            )
        return layers

    def train_baseline_vaes(
        self,
        layers: dict[int, CPTLayerFold],
        latent_dim: int,
        hidden: int,
        epochs: int,
        kl_weight: float,
    ) -> dict[int, Any]:
        profiles = {
            layer: fold.train_profiles_ns.astype(np.float32) for layer, fold in layers.items()
        }
        coordinates = {
            layer: fold.train_coordinates.astype(np.float32) for layer, fold in layers.items()
        }
        hull_points = np.vstack(list(coordinates.values()))
        hull = self.core.convex_hull(hull_points)
        return self.core.train_vaes(
            profiles,
            coordinates,
            self.profile_points,
            latent_dim,
            hidden,
            self.core.LEARNING_RATE,
            epochs,
            kl_weight,
            self.core.EARLY_STOP_PATIENCE,
            hull_poly=hull,
            inner_margin=self.core.HULL_INNER_MARGIN,
        )

    def encode(self, model: Any, profiles_ns: FloatArray) -> FloatArray:
        model.eval()
        with torch.no_grad():
            values = torch.as_tensor(profiles_ns, dtype=torch.float32, device=self.core.device)
            mean, _ = model.encode(values)
        return mean.detach().cpu().numpy().astype(np.float64)

    def decode(self, model: Any, latent: FloatArray, normal_score: Any) -> FloatArray:
        decoded_ns = self.core.decode_profiles(model, latent)
        return np.asarray(normal_score.inverse(decoded_ns), dtype=np.float64)
