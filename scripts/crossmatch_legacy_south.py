"""Create a lightweight literature × Legacy Survey South match manifest.

Examples:
  python scripts/crossmatch_legacy_south.py match
  python scripts/crossmatch_legacy_south.py top-k \
      outputs/crossmatches/legacy_south_manifest.parquet --k 100
  python scripts/crossmatch_legacy_south.py fetch-images \
      outputs/crossmatches/legacy_south_top_100_objects.parquet

`match` intentionally opens no Legacy image or RGB columns.  The resulting
manifest is the input to a later, targeted heavy-modality retrieval step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lsdb
import pandas as pd

MENTIONS = "hf://datasets/astronolan/galaxy-mentions-hats"
LEGACY_SOUTH = "hf://datasets/hugging-science/mmu_legacysurvey_dr10_south_21"
MENTION_COLUMNS = [
    "mention_id",
    "ra",
    "dec",
    "mention_summary",
    "arxiv_id",
    "wiki_entity_id",
]
LEGACY_COLUMNS = ["object_id", "ra", "dec"]


def require_columns(catalog: object, requested: list[str], label: str) -> None:
    """Raise a useful error before an expensive computation starts."""
    available = set(catalog.columns)  # type: ignore[attr-defined]
    missing = [column for column in requested if column not in available]
    if missing:
        raise ValueError(
            f"{label} is missing columns {missing}. Available columns: "
            f"{sorted(available)}"
        )


def command_match(args: argparse.Namespace) -> None:
    mentions = lsdb.open_catalog(args.mentions, columns=MENTION_COLUMNS)
    legacy = lsdb.open_catalog(args.catalog, columns=LEGACY_COLUMNS)
    require_columns(mentions, MENTION_COLUMNS, "mentions catalog")
    require_columns(legacy, LEGACY_COLUMNS, "Legacy catalog")

    matches = mentions.crossmatch(
        legacy,
        radius_arcsec=args.radius_arcsec,
        n_neighbors=1,
        suffixes=("_mention", "_legacy"),
        suffix_method="all_columns",
    )
    # LSDB returns a NestedFrame.  Convert its lightweight result to a regular
    # pandas DataFrame before writing; NestedFrame.to_parquet has no `index=`
    # parameter, unlike pandas.DataFrame.to_parquet.
    manifest = pd.DataFrame(matches.compute())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    print(f"Wrote {len(manifest):,} matched mentions to {output}")
    print(f"Unique Legacy objects: {manifest['object_id_legacy'].nunique():,}")


def command_top_k(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = pd.read_parquet(manifest_path)
    object_id = "object_id_legacy"
    mention_id = "mention_id_mention"
    required = {object_id, mention_id, "ra_legacy", "dec_legacy"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is not a Legacy manifest; missing {missing}")

    counts = (
        manifest.groupby(object_id)[mention_id]
        .nunique()
        .rename("caption_count")
        .sort_values(ascending=False)
    )
    selected_ids = counts.head(args.k).index
    captions = manifest[manifest[object_id].isin(selected_ids)].copy()
    objects = (
        captions.sort_values("_dist_arcsec")
        .drop_duplicates(object_id)
        [[object_id, "ra_legacy", "dec_legacy"]]
        .merge(counts.rename_axis(object_id).reset_index(), on=object_id)
        .sort_values(["caption_count", object_id], ascending=[False, True])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"legacy_south_top_{args.k}"
    captions_path = output_dir / f"{stem}_captions.parquet"
    objects_path = output_dir / f"{stem}_objects.parquet"
    captions.to_parquet(captions_path, index=False)
    objects.to_parquet(objects_path, index=False)
    print(f"Wrote {len(objects):,} objects to {objects_path}")
    print(f"Wrote {len(captions):,} matched captions to {captions_path}")


def command_fetch_images(args: argparse.Namespace) -> None:
    """Retrieve heavy columns only for objects selected by ``top-k``."""
    objects_path = Path(args.objects)
    objects = pd.read_parquet(objects_path)
    required = {"object_id_legacy", "ra_legacy", "dec_legacy"}
    missing = sorted(required.difference(objects.columns))
    if missing:
        raise ValueError(f"{objects_path} is not a Legacy object subset; missing {missing}")

    targets_df = (
        objects[["object_id_legacy", "ra_legacy", "dec_legacy"]]
        .drop_duplicates("object_id_legacy")
        .rename(
            columns={
                "object_id_legacy": "target_object_id",
                "ra_legacy": "ra",
                "dec_legacy": "dec",
            }
        )
    )
    targets = lsdb.from_dataframe(targets_df, ra_column="ra", dec_column="dec")
    columns = ["object_id", "ra", "dec", *args.columns]
    legacy = lsdb.open_catalog(args.catalog, columns=columns)
    require_columns(legacy, columns, "Legacy catalog")

    retrieved = pd.DataFrame(
        targets.crossmatch(
            legacy,
            radius_arcsec=args.radius_arcsec,
            n_neighbors=1,
            suffixes=("_target", "_legacy"),
            suffix_method="all_columns",
        ).compute()
    )
    if len(retrieved) != len(targets_df):
        raise RuntimeError(
            f"Retrieved {len(retrieved)} of {len(targets_df)} requested objects; "
            "increase --radius-arcsec only if the selected coordinates are not "
            "copied directly from the Legacy manifest."
        )
    if not (
        retrieved["target_object_id_target"].astype(str)
        == retrieved["object_id_legacy"].astype(str)
    ).all():
        raise RuntimeError("Spatial retrieval returned an object ID different from its target ID")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    retrieved.to_parquet(output, index=False)
    print(f"Wrote {len(retrieved):,} image rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="create the lightweight match manifest")
    match.add_argument("--mentions", default=MENTIONS, help="HATS URI for literature mentions")
    match.add_argument("--catalog", default=LEGACY_SOUTH, help="HATS URI for Legacy South")
    match.add_argument("--radius-arcsec", type=float, default=1.0)
    match.add_argument(
        "--output",
        default="outputs/crossmatches/legacy_south_manifest.parquet",
    )
    match.set_defaults(func=command_match)

    top_k = subparsers.add_parser("top-k", help="write the top-K caption-rich subsets")
    top_k.add_argument("manifest", help="manifest created by the match command")
    top_k.add_argument("--k", type=int, default=100)
    top_k.add_argument("--output-dir", default="outputs/crossmatches")
    top_k.set_defaults(func=command_top_k)

    fetch = subparsers.add_parser("fetch-images", help="retrieve images for a selected object subset")
    fetch.add_argument("objects", help="object subset created by the top-k command")
    fetch.add_argument("--catalog", default=LEGACY_SOUTH, help="HATS URI for Legacy South")
    fetch.add_argument(
        "--columns",
        default=["image", "rgb"],
        nargs="+",
        help="heavy Legacy columns to retrieve (default: image rgb)",
    )
    fetch.add_argument("--radius-arcsec", type=float, default=0.1)
    fetch.add_argument(
        "--output",
        default="outputs/crossmatches/legacy_south_top_images.parquet",
    )
    fetch.set_defaults(func=command_fetch_images)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "k", 1) < 1:
        raise ValueError("--k must be at least 1")
    args.func(args)


if __name__ == "__main__":
    main()
