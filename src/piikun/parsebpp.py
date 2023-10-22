#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from piikun import runtime

patterns = {
    "a10-species-delimitation-models": re.compile(r"^Number of species-delimitation models = (\d+).*$"),
    "a10-species-delimitation-model-header": re.compile(r"^ *model +prior +posterior.*$"),
    "a10-species-delimitation-model-row": re.compile(r"^ *(\d+?) +([01]+) +([0-9Ee\-.]+) +([0-9Ee\-.]+).*$"),
}

def _format_error(format_type, message):
    import sys
    runtime.RuntimeClient._logger.error(f"Invalid '{format_type}' format: {message}")
    sys.exit(1)

def check_for_overlapping_labels(labels):
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i != j and label1 in label2:
                return True, (label1, label2)
    return False, None

def parse_species_subsets(
    species_subsets_str,
    lineage_labels,
):
    species_subsets = []
    current_subset = []
    temp_lineage_label = ""
    for cidx, char in enumerate(species_subsets_str):
        if char == ' ':
            if current_subset:
                species_subsets.append(current_subset)
                current_subset = []
        else:
            temp_lineage_label += char
            if temp_lineage_label in lineage_labels:
                current_subset.append(temp_lineage_label)
                temp_lineage_label = ""
            elif cidx == len(species_subsets_str) - 1:
                print(char)
                print(temp_lineage_label)
                assert False
    if current_subset:
        species_subsets.append(current_subset)
    return species_subsets

def parse_guide_tree_with_pp(tree_str, rooting="force-rooted"):
    import dendropy
    tree = dendropy.Tree.get_from_string(
        tree_str,
        schema="newick",
        rooting="force-rooted",
    )
    clade_roots = []
    for nd in tree:
        if nd.label:
            nd.posterior_probability = float(nd.label[1:])
            clade_roots.append(nd)
    clade_roots = sorted(clade_roots, key=lambda x: x.posterior_probability, reverse=True)
    tree.clade_roots = clade_roots
    return tree


def parse_bpp_a10(
    source_stream,
    partition_factory,
):

    n_expected_lineages = None
    n_expected_species_delimitation_models = None
    n_expected_ancestral_nodes = None

    guide_tree_str = None
    lineage_labels = []
    species_delimitation_model_defs = []
    ancestral_node_descs = []

    current_section = "pre"
    is_done_processing_rows = None
    for line_offset, line in enumerate(source_stream):
        line_idx = line_offset + 1
        line_text = line.strip()
        if current_section == "pre":
            assert current_section == "pre", current_section
            # if line_text.startswith("COMPRESSED ALIGNMENTS") and not line_text.startswith("COMPRESSED ALIGNMENTS AFTER"):
            #     current_section = "alignments"
            m = patterns["a10-species-delimitation-models"].match(line_text)
            if m:
                current_section = "species-delimitation-models"
                n_expected_species_delimitation_models = int(m[1])
            continue
        elif current_section == "species-delimitation-models":
            if not line_text:
                continue
            if line_text.startswith("Order of ancestral nodes"):
                current_section = "ancestral-node-definitions"
                continue
            m = patterns["a10-species-delimitation-model-row"].match(line_text)
            if not m:
                if not patterns["a10-species-delimitation-model-header"].match(line_text):
                    _format_error(format_type="bpp-a10", message=f"Expecting model header row: line {line_idx}: '{line_text}'")
                continue
            species_delimitation_model_defs.append({
                "model_id": m[1],
                "model_code": m[2],
                "prior": str(m[3]),
                "posterior": str(m[4]),
            })
            continue
        elif current_section == "ancestral-node-definitions":
            if not line_text:
                current_section = "post-ancestral-node-definitions"
                continue
            n_expected_ancestral_nodes = len(species_delimitation_model_defs[0]["model_code"])
            ancestral_node_label = line_text.strip()
            ancestral_node_descs.append(ancestral_node_label)
            continue
        elif current_section == "post-ancestral-node-definitions":
            if not line_text:
                continue
            if line_text.startswith("Guide tree with posterior probability"):
                current_section = "guide-tree-pp"
                continue
            continue
        elif current_section == "guide-tree-pp":
            if not line_text:
                current_section = "post-guide-tree-pp"
                continue
            guide_tree_str = line_text
            continue
    assert len(species_delimitation_model_defs) == n_expected_species_delimitation_models, f"{len(species_delimitation_model_defs)} != {n_expected_species_delimitation_models}"
    assert len(ancestral_node_descs) == n_expected_ancestral_nodes
    assert guide_tree_str
    guide_tree = parse_guide_tree_with_pp(tree_str=guide_tree_str)
    species_labels = [taxon.label for taxon in guide_tree.taxon_namespace]
    ancestral_node_subsets = []
    for anc_idx, desc in enumerate(ancestral_node_descs):
        subsets = parse_species_subsets(
            species_subsets_str=desc,
            lineage_labels=species_labels,
        )
        ancestral_node_subsets.append(subsets)
    print(ancestral_node_subsets)
    assert len(ancestral_node_subsets) == len(ancestral_node_descs)
    # print(species_delimitation_model_defs)

class BppParser:

    def __init__(self):
        pass

    def parse_bpp_a10(self, source_stream, partition_factory):
        pass

