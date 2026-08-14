"""Create a lightweight literature × survey match manifest.

Examples:
  python scripts/crossmatch_legacy_south.py match --survey manga
  python scripts/crossmatch_legacy_south.py top-k \
      outputs/crossmatches/legacy_north_manifest.parquet --survey legacy-north --k 100
  python scripts/crossmatch_legacy_south.py fetch-images \
      outputs/crossmatches/legacy_north_top_100_objects.parquet --survey legacy-north

`match` intentionally opens no image columns. The resulting
manifest is the input to a later, targeted heavy-modality retrieval step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lsdb
import pandas as pd

MENTIONS = "hf://datasets/astronolan/galaxy-mentions-hats"
SURVEYS = {
    "legacy-south": {
        "uri": "hf://datasets/hugging-science/mmu_legacysurvey_dr10_south_21",
        "suffix": "legacy",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": [],
        "image_columns": ["image", "rgb"],
    },
    "legacy-north": {
        "uri": "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
        "suffix": "legacy",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["z_spec"],
        "image_columns": ["image", "z_spec"],
    },
    "manga": {
        "uri": "hf://datasets/hugging-science/mmu_manga",
        "suffix": "manga",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["z"],
        "image_columns": ["images", "z"],
    },
    "hsc": {
        "uri": "hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
        "suffix": "hsc",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": [],
        "image_columns": ["image"],
    },
    "galaxy-zoo": {
        "uri": "hf://datasets/UniverseTBD/mmu_gz10",
        "suffix": "gz10",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["redshift", "gz10_label"],
        "image_columns": ["rgb_image", "redshift", "gz10_label", "rgb_pixel_scale"],
    },
    "des-dr2": {
        "uri": "https://linea.data.lsdb.io/hats/des/des_dr2",
        "suffix": "des",
        "id_column": "COADD_OBJECT_ID",
        "ra_column": "RA",
        "dec_column": "DEC",
        "light_columns": [],
        "image_columns": [],
    },
}
MENTION_COLUMNS = [
    "mention_id",
    "ra",
    "dec",
    "mention_summary",
    "arxiv_id",
    "wiki_entity_id",
]


def require_columns(catalog: object, requested: list[str], label: str) -> None:
    """Raise a useful error before an expensive computation starts."""
    available = set(catalog.columns)  # type: ignore[attr-defined]
    missing = [column for column in requested if column not in available]
    if missing:
        raise ValueError(
            f"{label} is missing columns {missing}. Available columns: "
            f"{sorted(available)}"
        )


def catalog_uri(args: argparse.Namespace) -> str:
    return args.catalog or SURVEYS[args.survey]["uri"]


def selected_object_column(survey: str) -> str:
    return f"object_id_{SURVEYS[survey]['suffix']}"


def catalog_key_columns(survey: str) -> list[str]:
    profile = SURVEYS[survey]
    return [profile["id_column"], profile["ra_column"], profile["dec_column"]]


def normalize_catalog_columns(frame: pd.DataFrame, survey: str) -> pd.DataFrame:
    """Use stable output names even when a source uses uppercase column names."""
    profile = SURVEYS[survey]
    suffix = profile["suffix"]
    return frame.rename(
        columns={
            f"{profile['id_column']}_{suffix}": f"object_id_{suffix}",
            f"{profile['ra_column']}_{suffix}": f"ra_{suffix}",
            f"{profile['dec_column']}_{suffix}": f"dec_{suffix}",
        }
    )


def parse_selection_count(value: str) -> int | None:
    """Parse a positive top-K count, or all for the full intersection."""
    if value.lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--k must be a positive integer or 'all'") from error
    if count < 1:
        raise argparse.ArgumentTypeError("--k must be a positive integer or 'all'")
    return count


def command_match(args: argparse.Namespace) -> None:
    mentions = lsdb.open_catalog(args.mentions, columns=MENTION_COLUMNS)
    catalog_columns = [*catalog_key_columns(args.survey), *SURVEYS[args.survey]["light_columns"]]
    legacy = lsdb.open_catalog(catalog_uri(args), columns=catalog_columns)
    require_columns(mentions, MENTION_COLUMNS, "mentions catalog")
    require_columns(legacy, catalog_columns, f"{args.survey} catalog")

    matches = mentions.crossmatch(
        legacy,
        radius_arcsec=args.radius_arcsec,
        n_neighbors=1,
        suffixes=("_mention", f"_{SURVEYS[args.survey]['suffix']}"),
        suffix_method="all_columns",
    )
    # LSDB returns a NestedFrame.  Convert its lightweight result to a regular
    # pandas DataFrame before writing; NestedFrame.to_parquet has no `index=`
    # parameter, unlike pandas.DataFrame.to_parquet.
    manifest = normalize_catalog_columns(pd.DataFrame(matches.compute()), args.survey)

    output = Path(
        args.output or f"outputs/crossmatches/{args.survey.replace('-', '_')}_manifest.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    print(f"Wrote {len(manifest):,} matched mentions to {output}")
    print(f"Unique {args.survey} objects: {manifest[selected_object_column(args.survey)].nunique():,}")


def command_top_k(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = pd.read_parquet(manifest_path)
    object_id = selected_object_column(args.survey)
    suffix = SURVEYS[args.survey]["suffix"]
    mention_id = "mention_id_mention"
    required = {object_id, mention_id, f"ra_{suffix}", f"dec_{suffix}"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is not a {args.survey} manifest; missing {missing}")

    counts = (
        manifest.groupby(object_id)[mention_id]
        .nunique()
        .rename("caption_count")
        .sort_values(ascending=False)
    )
    selected_ids = counts.index if args.k is None else counts.head(args.k).index
    captions = manifest[manifest[object_id].isin(selected_ids)].copy()
    objects = (
        captions.sort_values("_dist_arcsec")
        .drop_duplicates(object_id)
        [[object_id, f"ra_{suffix}", f"dec_{suffix}"]]
        .merge(counts.rename_axis(object_id).reset_index(), on=object_id)
        .sort_values(["caption_count", object_id], ascending=[False, True])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = "all" if args.k is None else f"top_{args.k}"
    stem = f"{args.survey.replace('-', '_')}_{selection}"
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
    suffix = SURVEYS[args.survey]["suffix"]
    object_id = selected_object_column(args.survey)
    required = {object_id, f"ra_{suffix}", f"dec_{suffix}"}
    missing = sorted(required.difference(objects.columns))
    if missing:
        raise ValueError(f"{objects_path} is not a {args.survey} object subset; missing {missing}")

    targets_df = (
        objects[[object_id, f"ra_{suffix}", f"dec_{suffix}"]]
        .drop_duplicates(object_id)
        .rename(
            columns={
                object_id: "target_object_id",
                f"ra_{suffix}": "ra",
                f"dec_{suffix}": "dec",
            }
        )
    )
    targets = lsdb.from_dataframe(targets_df, ra_column="ra", dec_column="dec")
    image_columns = args.columns or SURVEYS[args.survey]["image_columns"]
    if not image_columns:
        raise ValueError(f"{args.survey} has no image payload; use match/top-k only.")
    columns = [*catalog_key_columns(args.survey), *image_columns]
    legacy = lsdb.open_catalog(catalog_uri(args), columns=columns)
    require_columns(legacy, columns, f"{args.survey} catalog")

    retrieved = normalize_catalog_columns(pd.DataFrame(
        targets.crossmatch(
            legacy,
            radius_arcsec=args.radius_arcsec,
            n_neighbors=1,
            suffixes=("_target", f"_{suffix}"),
            suffix_method="all_columns",
        ).compute()
    ), args.survey)
    if len(retrieved) != len(targets_df):
        raise RuntimeError(
            f"Retrieved {len(retrieved)} of {len(targets_df)} requested objects; "
            "increase --radius-arcsec only if the selected coordinates are not "
            "copied directly from the match manifest."
        )
    if not (
        retrieved["target_object_id_target"].astype(str)
        == retrieved[object_id].astype(str)
    ).all():
        raise RuntimeError("Spatial retrieval returned an object ID different from its target ID")

    output = Path(
        args.output
        or f"outputs/crossmatches/{args.survey.replace('-', '_')}_top_images.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    retrieved.to_parquet(output, index=False)
    print(f"Wrote {len(retrieved):,} image rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="create the lightweight match manifest")
    match.add_argument("--mentions", default=MENTIONS, help="HATS URI for literature mentions")
    match.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    match.add_argument("--catalog", help="override the survey's HATS URI")
    match.add_argument("--radius-arcsec", type=float, default=1.0)
    match.add_argument(
        "--output",
        help="output Parquet path (default is based on --survey)",
    )
    match.set_defaults(func=command_match)

    top_k = subparsers.add_parser("top-k", help="write the top-K caption-rich subsets")
    top_k.add_argument("manifest", help="manifest created by the match command")
    top_k.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    top_k.add_argument(
        "--k",
        type=parse_selection_count,
        default=100,
        help="number of caption-rich objects, or 'all' for the full intersection",
    )
    top_k.add_argument("--output-dir", default="outputs/crossmatches")
    top_k.set_defaults(func=command_top_k)

    fetch = subparsers.add_parser("fetch-images", help="retrieve images for a selected object subset")
    fetch.add_argument("objects", help="object subset created by the top-k command")
    fetch.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    fetch.add_argument("--catalog", help="override the survey's HATS URI")
    fetch.add_argument(
        "--columns",
        default=None,
        nargs="+",
        help="heavy columns to retrieve (default depends on --survey)",
    )
    fetch.add_argument("--radius-arcsec", type=float, default=0.1)
    fetch.add_argument(
        "--output",
        help="output Parquet path (default is based on --survey)",
    )
    fetch.set_defaults(func=command_fetch_images)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
