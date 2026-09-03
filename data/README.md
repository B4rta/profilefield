# External data

No Biomazon raster, HDF5, or generated feature data is committed. The official release is
[Biomazon v1.1](https://doi.org/10.26165/JUELICH-DATA/2YP2PZ), distributed as three large
256 x 256-patch HDF5 files plus split lists, normalization statistics, and loss weights.

The repository includes a resumable downloader that reads the official Juelich DATA manifest,
keeps partial downloads as `.part` files, and verifies every completed file with SHA-256. Its safe
default downloads only the small support files:

```powershell
python scripts/download_biomazon.py --selection support
```

Probe the complete release before allocating storage:

```powershell
python scripts/download_biomazon.py --selection all --probe-only
```

Download only the support files plus the test HDF5, or the complete release, after confirming the
reported sizes and free space:

```powershell
python scripts/download_biomazon.py --selection test --allow-large
python scripts/download_biomazon.py --selection all --allow-large
```

The destination defaults to git-ignored `data/biomazon/`. Set `BIOMAZON_ROOT` to the resulting
`data/biomazon/dataset` directory. A trusted mirror can be supplied with `--base-url` if the
official Datapub endpoint is temporarily unavailable. Never commit the downloaded files.

## Availability incident (checked 2026-09-01)

The Juelich DATA API and the 2,324-byte official checksum manifest are publicly accessible and the
manifest checksum verifies correctly. The linked `https://datapub.fz-juelich.de/biomazon/` data
endpoint, however, currently times out before completing an HTTP/TLS connection. The Datapub root
and another known public Datapub collection fail in the same way, so this is a service-wide
availability problem rather than a bad Biomazon path or a local browser issue.

The arXiv v2 manuscript source (Section "Data availability") still says that the dataset, model
weights, and source code will be made public soon. The DOI record currently stores only the
checksum manifest and links the external Datapub directory. Once that endpoint or a trusted mirror
becomes available, rerunning the downloader will continue any `.part` files and verify all results.
The official dataset contacts listed in the DOI metadata are `sa.mandal@fz-juelich.de` and
`r.sedona@fz-juelich.de`.

## Downloaded fallback: ETH AGBD_raw

Because the Biomazon Datapub endpoint was unavailable, the complete public
[`prs-eth/AGBD_raw`](https://huggingface.co/datasets/prs-eth/AGBD_raw) release is used as the
multimodal fallback. It contains 10 m Sentinel-2 patches, ALOS-2, canopy height, land cover,
topography/DEM, coordinates, GEDI AGBD and uncertainty, RH98, acquisition metadata, and official
train/validation/test splits. The compressed Parquet release is 418,132,172,941 bytes (418.13 GB)
and contains 15,934,561 samples. Its license is CC BY-NC 4.0.

The resumable download is pinned to the current immutable Hugging Face commit. The generated
`download_plan.json` records every shard's expected size and SHA-256:

```powershell
C:\Users\barta\.codex\venvs\profilefield-gedidb-download\Scripts\python.exe `
  scripts/download_agbd_raw.py
```

The data remains under git-ignored `data/agbd_raw/`. The download is complete: all 2,658 shards,
418,132,172,941 bytes and 15,934,561 rows match the pinned release, and every SHA-256 was verified
again by fully rereading the local files. `shard_catalog.parquet` records per-shard bounds and row
counts for spatially selective access.

## Joined GEDI profile data

RH0-RH100 profiles were queried from the public GFZ gediDB v2026.4.30 without duplicating its
global ~20 TB archive. The query enforces sensitivity >= 0.95, full-power beam type, and passing
L2A, L2B, and L4 quality flags. `data/gedidb_amazon/` contains 11,424 valid profiles across Tapajos,
Manaus, Acre, and French Guiana for within-region and leave-region-out experiments.

For the EO-context experiment, three French Guiana cells with upstream AGBD_raw train,
validation, and test membership were queried separately. The exact join retains 3,175 footprints
only when separation is <= 30 m, AGBD matches at stored precision, and RH98 differs by <= 0.05 m.
The resulting table contains 33 selected Sentinel-2/context features and 101 ordered RH targets.
All Parquet files and manifests remain under git-ignored `data/gedidb_agbd_join/`.
