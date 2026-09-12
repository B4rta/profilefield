# GeoVAE-Field / ProfileField

Repository: [github.com/B4rta/profilefield](https://github.com/B4rta/profilefield)

ProfileField is a research framework for adaptive functional residual
coregionalization in conditional simulation of spatially distributed environmental profiles.
Its main real-data model combines a multioutput random-forest context mean, a fixed
local-linear residual basis, matched independent and coregionalized sparse variational GPs,
validation-selected spatial-dependence shrinkage, isolated joint calibration, and
constraint-preserving posterior draws.

Version 0.2 supports signed monotone relative-height profiles, block-held-out
training residuals, and exact spatial latent-factor sampling with explicit
observation noise. GEDI configurations in `configs/gedidb_v2/` use these settings.
The random forest is a single multioutput estimator sharing tree splits across
ordinates. Older artifact labels saying “Per-RH random forest” refer to this
same estimator, not to separately fitted forests.

Empirical CRPS and quantile coverage score the delivered ensemble. Gaussian
moment scores are retained as auxiliary diagnostics. Whole-profile coverage of
marginal bands is descriptive; it is not a simultaneous-coverage guarantee.
Repeated optimizer seeds measure computational variability and are not new
independent geographical observations.

Exact sampling factors one location-by-location covariance per latent process;
its cost is cubic in the number of queried locations. Sparse training does not
make this prediction step linear. The `full_covariance=False` argument avoids
returning the full output covariance but still generates joint spatial draws.

The published [GeoVAE-SGS](https://github.com/B4rta/geovae-sgs) implementation is included as
a read-only, commit-pinned Git submodule. ProfileField never edits that scientific baseline.

## Quick start

```powershell
git submodule update --init --recursive
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements/lock-cpu-win-py312.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\profilefield synthetic --config configs/synthetic/smoke.yaml
.venv\Scripts\pytest
```

Run `python scripts/validate_sampler.py` to reproduce the numerical covariance
diagnostic. New runs include SHA-256 checksums for every artifact and a source
file fingerprint. Legacy runs remain readable with their weaker integrity
status explicitly reported.

Before pushing, run `python scripts/check_release.py`. The same privacy check
runs in CI. Manuscript drafts, article figures/tables, private documentation and
publication build utilities are excluded from this repository.

Use the files in `configs/synthetic/` for deterministic experiments and sensitivity checks.
Every run produces a self-contained
directory under `outputs/` with configuration, environment, data and split manifests, run metadata,
metric tables, model states, diagnostics, and PNG/PDF figures.

## Experiments

- `profilefield synthetic`: known correlated latent field; fair independent-SVGP vs LMC-SVGP.
- `profilefield cpt-audit`: verifies the immutable GeoVAE-SGS pin, data hashes, split integrity,
  and recomputes the published continuity metrics from source artifacts.
- `profilefield cpt`: executes the full published-fold continuity comparison, retraining the
  original VAE preprocessing per fold and changing only the latent spatial GP.
- `profilefield biomazon-validate`: validates locally downloaded Biomazon feature/profile files,
  official or geographic splits, checksums, RH ordering, and leakage. It never downloads data.
- `profilefield biomazon`: trains the common model suite on a validated local feature table.
- `profilefield validate-run --path <run>`: checks the complete artifact contract and manifest
  hashes for a finished run.

The `biomazon` command name is retained as the generic real-profile adapter for compatibility;
the completed scientific experiments use the provenance-tracked GEDI/AGBD inputs described
below and do not claim Biomazon results.

## External GEDI and EO data

The data preparation utilities can link
[ETH AGBD_raw](https://huggingface.co/datasets/prs-eth/AGBD_raw) observations to quality-filtered
RH0-RH100 profiles from the public [GFZ gediDB](https://gedidb.readthedocs.io/). The local download,
shard catalog, GEDI query and exact footprint join are driven by:

```powershell
python scripts/download_agbd_raw.py
python scripts/catalog_agbd_raw.py
python -m pip install -r requirements/lock-gedidb-win-py312.txt
python scripts/download_gedidb_profiles.py --output data/gedidb_agbd_join --regions agbd_train agbd_validation agbd_test
python scripts/join_agbd_gedidb.py --gedi-profiles data/gedidb_agbd_join/gedi_profiles.parquet --output data/gedidb_agbd_join/french_guiana_eo_rh.parquet
python scripts/build_profile_cells.py --input data/gedidb_agbd_join/french_guiana_eo_rh.parquet --output data/gedidb_agbd_join/french_guiana_eo_rh_cells_500m.parquet --cell-size-m 500 --min-shots 3
python scripts/build_profile_cells.py --input data/gedidb_agbd_join/french_guiana_eo_rh.parquet --output data/gedidb_agbd_join/french_guiana_eo_rh_cells_1km.parquet --cell-size-m 1000 --min-shots 3
profilefield biomazon-validate --config configs/gedidb/eo_context.yaml
profilefield biomazon --config configs/gedidb/eo_context.yaml
```

The strict join requires proximity within 30 m plus agreement in AGBD and RH98. Official AGBD_raw
partitions are retained, and calibration is carved only from training blocks. The original Biomazon
adapter remains available, but no Biomazon performance value is claimed because its endpoint was
not accessible. Large source data, credentials, caches and generated runs are ignored by Git.

## Frozen independent geographic validation

The external protocol in `configs/gedidb_v2/external_validation.yaml` transfers the
existing French Guiana footprint models to a North Island, New Zealand landscape.
The query cell is chosen from AGBD coordinate counts alone, before accessing its RH
outcomes. Freeze source artifact hashes before querying GEDI:

```powershell
python scripts/freeze_external.py
python scripts/download_gedidb_profiles.py --output data/nz_join --region-spec configs/gedidb_v2/external_region.json
python scripts/join_agbd_gedidb.py --gedi-profiles data/nz_join/gedi_profiles.parquet --output data/nz_join/nz_eo_rh.parquet
python scripts/run_external.py --protocol <sealed-protocol-directory>
```

All eight source comparators retain their fitted model states, feature transforms,
validation-selected weights, and calibration multipliers. No external targets are
used for fitting, tuning, or calibration. Because older source checkpoints did not
save residual covariance matrices, the evaluator reconstructs these from the same
source-only cross-fitting folds and verifies their residual statistics. It does not
retrain production forests, neural networks, or GPs. Trusted checkpoint restoration
is tested against original predictive means, variances, and draws.

External case selection uses sorted IDs and a frozen random seed, never scores.
The evaluator checks feature identity, source checksums, footprint-ID disjointness,
and minimum geographic separation. A second region tests geographic transfer, not
universal biome independence; repeated fitting seeds are still not independent sites.
The query box can contain mixed land cover and is not represented as a forest-only
probability sample. Existing French Guiana results remain separate and unchanged.

## Data-splitting safeguards

All normalization, residual bases, empirical covariances and event thresholds are fit on training
indices only. Split manifests are frozen before fitting. Validation/calibration/test identities
and blocks are checked for overlap, and the test partition is excluded from selection and
calibration. Configuration files define the reproducible experiment settings.

On Windows network shares, executables inside a UNC-hosted `.venv` may be blocked. In that case
create the virtual environment on a local drive and install this repository into it in editable
mode; experiment outputs still remain in the project.
