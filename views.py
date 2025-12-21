# analysis/views.py
import os
import Helpers.DataHelpers as DataHelpers
from datasets import resolve_data_dir
from pipeline import ViewSpec

def genotype_views(spec, *, colors=("C0","C1","C2"), meta_filename="sex_gen.csv"):
    data_dir = resolve_data_dir(spec)
    meta_path = os.path.join(data_dir, meta_filename)

    return [
        ViewSpec("wt",  color=colors[0],
                 selector=lambda d, mp=meta_path: DataHelpers.restrict_subjects(
                     d, mp, genotypes="wt", subject_col="animal", genotype_col="genotype", attach_meta=True)),
        ViewSpec("het", color=colors[1],
                 selector=lambda d, mp=meta_path: DataHelpers.restrict_subjects(
                     d, mp, genotypes="het", subject_col="animal", genotype_col="genotype", attach_meta=True)),
        ViewSpec("hom", color=colors[2],
                 selector=lambda d, mp=meta_path: DataHelpers.restrict_subjects(
                     d, mp, genotypes="hom", subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ]



def single_genotype_view(spec, *, genotype="hom", sex=None, name=None, color="C0", meta_filename="sex_gen.csv"):
    data_dir = resolve_data_dir(spec)
    meta_path = os.path.join(data_dir, meta_filename)

    vname = name if name is not None else (f"{genotype}_{sex}" if sex else genotype)

    return [
        ViewSpec(
            vname,
            color=color,
            selector=lambda d, mp=meta_path, g=genotype, s=sex: DataHelpers.restrict_subjects(
                d, mp,
                genotypes=g,
                sex=s,
                subject_col="animal",
                genotype_col="genotype",
                attach_meta=True,
            ),
        )
    ]



def sex_views(spec, *, genotype = "hom", 
              colors=("#1f77b4", "#e75480"), meta_filename="sex_gen.csv"):
    data_dir = resolve_data_dir(spec)
    meta_path = os.path.join(data_dir, meta_filename)

    return [
        ViewSpec("male", color=colors[0],
                 selector=lambda d, mp=meta_path, g=genotype: DataHelpers.restrict_subjects(
                     d, mp, genotypes=g, sex="male",
                     subject_col="animal", genotype_col="genotype", attach_meta=True)),
        ViewSpec("female", color=colors[1],
                 selector=lambda d, mp=meta_path, g=genotype: DataHelpers.restrict_subjects(
                     d, mp, genotypes=g, sex="female",
                     subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ]




def genotype_sex_views(
    spec,
    *,
    genotype="hom",
    sexes=("male", "female"),
    colors=("#1f77b4", "#e75480"),
    meta_filename="sex_gen.csv",
    name_prefix=None,
    extra_views=None,
):
    """
    Returns views split by sex for a chosen genotype.
    Example output names: hom_male, hom_female

    - genotype: "wt" / "het" / "hom" (or list/tuple if you want)
    - sexes: which sexes to include
    - extra_views: optional list[ViewSpec] to append
    """
    data_dir = resolve_data_dir(spec)
    meta_path = os.path.join(data_dir, meta_filename)

    prefix = name_prefix if name_prefix is not None else str(genotype)

    views = []
    for i, sex in enumerate(sexes):
        color = colors[i % len(colors)]
        views.append(
            ViewSpec(
                name=f"{prefix}_{sex}",
                color=color,
                selector=lambda d, mp=meta_path, g=genotype, s=sex: DataHelpers.restrict_subjects(
                    d, mp,
                    genotypes=g,
                    sex=s,
                    subject_col="animal",
                    genotype_col="genotype",
                    attach_meta=True,
                ),
            )
        )

    if extra_views:
        views.extend(extra_views)

    return views



def stimdur_views(
    *,
    stim_durs=(15, 60, 120, 6000),
    stimdur_col="stim_dur",
    colors=("C9", "C4", "C5", "C7"),
    label_map=None,
):
    """
    Returns one ViewSpec per stim_dur.
    6000 is typically your "RT" condition (labeling handled via label_map).
    """

    if label_map is None:
        label_map = {15: "15 ms", 60: "60 ms", 120: "120 ms", 6000: "RT"}

    views = []
    for i, sd in enumerate(stim_durs):
        name = label_map.get(sd, f"{sd} ms")
        color = colors[i % len(colors)]
        views.append(
            ViewSpec(
                name=name,
                color=color,
                selector=lambda d, sd=sd, col=stimdur_col: d[d[col] == sd].copy(),
            )
        )
    return views