#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from piikun import runtime

patterns = {
    "a10-species-delimitation-models": re.compile(r"^Number of species-delimitation models = (\d+).*$"),
    "a10-species-delimitation-model-header": re.compile(r"^ *model +prior +posterior.*$"),
    "a10-species-delimitation-model-row": re.compile(r"^ *(\d+?) +([01]+) +([0-9Ee\-.]+) +([0-9Ee\-.]+).*$"),

    "alignment-ntax-nchar": re.compile(r"^\s*(\d+)\s*(\d+)\s*.*$"),
    "alignment-row": re.compile(r"^(.*?)?\^(\S+).*$"),
    "alignment-end": re.compile(r"^[0-9 ]+$"),

    "a11-section-b": re.compile(r"\(B\)\s*(\d+) species delimitations .*$"),
    # "a11-species-delimitation-desc": re.compile(r"^(\d+\)\s+([0-9.Ee\-]+)\s+(\d+)\s+\(.*)"),
    "a11-species-delimitation-desc": re.compile(r"^(\d+)\s+([0-9.Ee\-]+)\s+(\d+)\s+\(.*\)"),
    # "a11-species-delimitation-desc": re.compile(r"^74 .*"),
    "a11-section-c": re.compile(r"\(C\)\s*(\d+) delimited species .*$"),
    # "spd_pattern_end": re.compile(r"\(C\)\s*", re.MULTILINE),
    # "a11_treemodel_entry": re.compile(r"^(\d+) ([0-9.]+) ([0-9.]+) (\d+) (.*;).*$"),
    # "strip_tree_tokens": re.compile(r"[\(\),\s;]+"),
}

def _format_error(format_type, message):
    import sys
    runtime.RuntimeClient._logger._runtime_handler.show_path = True
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
    assert False
    if current_subset:
        species_subsets.append(current_subset)
    return species_subsets

def parse_guide_tree_with_pp(
    tree_str,
    rooting="force-rooted",
    ancestral_node_label_id_map=None,
):
    import dendropy
    tree = dendropy.Tree.get_from_string(
        tree_str,
        schema="newick",
        rooting="force-rooted",
    )
    tree.bpp_ancestor_label_node_map = {}
    tree.bpp_ancestor_index_node_map = {}
    for nd in tree:
        nd._is_collapsed = None
        if nd.label and nd.label.startswith("#"):
            nd.posterior_probability = float(nd.label[1:])
        if not nd.is_leaf():
            nd.leaf_node_labels = []
            for leaf_nd in nd.leaf_iter():
                nd.leaf_node_labels.append(leaf_nd.taxon.label)
            nd.bpp_ancestor_label = "".join(nd.leaf_node_labels)
            tree.bpp_ancestor_label_node_map[nd.bpp_ancestor_label] = nd
            if ancestral_node_label_id_map:
                tree.bpp_ancestor_index_node_map[ancestral_node_label_id_map[nd.bpp_ancestor_label]] = nd
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
    ancestral_node_label_id_map = {}

    current_section = "pre"
    is_done_processing_rows = None
    for line_offset, line in enumerate(source_stream):
        line_idx = line_offset + 1
        line_text = line.strip()
        if current_section == "pre":
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
                "prior_probability": str(m[3]),
                "posterior_probability": str(m[4]),
            })
            continue
        elif current_section == "ancestral-node-definitions":
            if not line_text:
                current_section = "post-ancestral-node-definitions"
                continue
            n_expected_ancestral_nodes = len(species_delimitation_model_defs[0]["model_code"])
            ancestral_node_label = line_text.strip()
            ancestral_node_label_id_map[ancestral_node_label] = len(ancestral_node_label_id_map)
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
    assert len(ancestral_node_label_id_map) == n_expected_ancestral_nodes

    assert guide_tree_str
    guide_tree = parse_guide_tree_with_pp(
        tree_str=guide_tree_str,
        ancestral_node_label_id_map=ancestral_node_label_id_map,
    )
    for anc_label, anc_nd in guide_tree.bpp_ancestor_label_node_map.items():
        anc_idx = ancestral_node_label_id_map[anc_label]
        assert guide_tree.bpp_ancestor_index_node_map[anc_idx] == anc_nd

    for spdm_idx, model_def in enumerate(species_delimitation_model_defs):

        # flag for open/closed
        for anc_idx, anc_code in enumerate(model_def["model_code"]):
            anc_nd = guide_tree.bpp_ancestor_index_node_map[anc_idx]
            assert ancestral_node_label_id_map[anc_nd.bpp_ancestor_label] == anc_idx
            if anc_code == "0":
                anc_nd._is_collapsed = True
            elif anc_code == "1":
                anc_nd._is_collapsed = False
            else:
                raise ValueError(anc_code)

        # collect subsets from closed parents
        subsets = []
        for nd in guide_tree:
            if nd._parent_node and nd._parent_node._is_collapsed:
                nd._is_collapsed = True
                continue
            if nd.is_leaf():
                # leaf with uncollapsed parent: singleton subset
                subsets.append([nd.taxon.label])
                continue
            if nd._is_collapsed:
                # top-most collapsed internal node in this subtree
                subsets.append(list(nd.leaf_node_labels))
                continue

        metadata_d = {
            "support": float(model_def["posterior_probability"]),
            "prior_probability": float(model_def["prior_probability"]),
            "posterior_probability": float(model_def["posterior_probability"]),
        }
        kwargs = {
            "metadata_d": metadata_d,
            "subsets": subsets,
        }
        partition = partition_factory(**kwargs)
        partition._origin_size = len(species_delimitation_model_defs)
        partition._origin_offset = spdm_idx
        yield partition

def parse_bpp_a11(
    source_stream,
    partition_factory,
):
    current_section = "pre"
    lineage_labels = []
    partition_info = []
    line = None
    line_idx = 0
    n_expected_lineages = None
    n_expected_lineages_set_on_line = None
    n_partitions_expected = None
    for line_offset, line in enumerate(source_stream):
        line_idx = line_offset + 1
        line_text = line.strip()
        if current_section == "pre":
            if line_text.startswith("COMPRESSED ALIGNMENTS"):
                current_section = "alignments1"
                continue
            if line_text.startswith("COMPRESSED ALIGNMENTS AFTER"):
                current_section = "post-alignments1"
                continue
            continue
        elif current_section == "alignments1":
            # if lineage_labels and n_expected_lineages and len(lineage_labels) == n_expected_lineages:
            #     continue
            # if line.startswith("COMPRESSED ALIGNMENTS"):
            #     continue
            if not line_text:
                continue
            m = patterns["alignment-ntax-nchar"].match(line_text)
            if m:
                if n_expected_lineages:
                    _format_error(format_type="bpp-a11", message=f"Unexpected alignment label and character description (already set to {n_expected_lineages} on {n_expected_lineages_set_on_line}): line {line_idx}: '{line_text}'")
                n_expected_lineages = int(m[1])
                n_expected_lineages_set_on_line = line_idx
                current_section = "alignment-row"
                continue
            _format_error(format_type="bpp", message=f"Missing alignment label and character description: line {line_idx}: '{line_text}'")
            # continue
        elif current_section == "alignment-row":
            # breakout pattern
            if not n_expected_lineages:
                _format_error(format_type="bpp", message=f"Number of expected lineages not set before parsing alignment: line {line_idx}: '{line_text}'")
                # continue
            m = patterns["alignment-end"].match(line_text)
            if m:
                current_section = "post-alignments1"
                continue
            if not line_text:
                continue
            m = patterns["alignment-row"].match(line_text)
            if m:
                if len(lineage_labels) == n_expected_lineages:
                    _format_error(format_type="bpp-a11", message=f"Unexpected sequence definition ({n_expected_lineages} labels already parsed): line {line_idx}: '{line_text}'")
                    # continue
                lineage_labels.append(m[2])
                continue
            _format_error(format_type="bpp-a11", message=f"Expected sequence data: line {line_idx}: '{line_text}'")
            # continue
            continue
        elif current_section == "post-alignments1":
            assert n_expected_lineages
            if len(lineage_labels) != n_expected_lineages:
                _format_error(format_type="bpp-a11", message=f"{n_expected_lineages} lineages expected but {len(lineage_labels)} lineages identified ({lineage_labels}): line {line_idx}: '{line_text}'")
                # continue
            m = patterns["a11-section-b"].match(line_text)
            if m:
                current_section = "a11-section-b"
                n_partitions_expected = int(m[1])
                continue
            # sink all till next section of interest
            continue
        elif current_section == "a11-section-b":
            assert n_expected_lineages
            if len(lineage_labels) != n_expected_lineages:
                _format_error(format_type="bpp-a11", message=f"{n_expected_lineages} lineages expected but {len(lineage_labels)} lineages identified ({lineage_labels}): line {line_idx}: '{line_text}'")
                # continue
            if not n_partitions_expected:
                _format_error(format_type="bpp-a11", message=f"Number of expected models not set before parsing models: line {line_idx}: '{line_text}'")
                # continue
            # if not line_text:
            #     current_section = "a11-section-c":
            m = patterns["a11-section-c"].match(line_text)
            if m:
                current_section = "a11-section-c"
                continue
            if not line_text:
                current_section = "a11-section-c"
                continue
            m = patterns["a11-species-delimitation-desc"].match(line_text)
            if m:
                parts = line_text.strip().split()
                frequency = float(parts[1])
                num_subsets = int(parts[2])
                species_subsets_str = " ".join(parts[3:]).strip("()")
                continue
            _format_error(format_type="bpp-a11", message=f"Expecting species delimitation model description: line {line_idx}: '{line_text}'")

            # species_subsets = []
            # current_subset = []
            # temp_lineage_label = ""
            # for char in species_subsets_str:
            #     if char == ' ':
            #         if current_subset:
            #             species_subsets.append(current_subset)
            #             current_subset = []
            #     else:
            #         temp_lineage_label += char
            #         if temp_lineage_label in lineage_labels:
            #             current_subset.append(temp_lineage_label)
            #             temp_lineage_label = ""

            species_subsets = parse_species_subsets(
                species_subsets_str=species_subsets_str,
                lineage_labels=lineage_labels,
            )

            partition_d = {
                "frequency": frequency,
                "n_subsets": num_subsets,
                "subsets": species_subsets
            }
            # runtime.logger.info(f"Partition {len(partition_info)+1} of {n_partitions_expected}: {num_subsets} clusters, probability = {frequency}")
            partition_info.append(partition_d)
        elif current_section == "a11-section-c":
            break
        else:
            _format_error(format_type="bpp", message=f"Unhandled line: {line_idx}: '{line_text}'")
            # continue
    if not partition_info:
        runtime.terminate_error("No species delimitation partitions parsed from source", exit_code=1)
    assert len(partition_info) == n_partitions_expected
    for ptn_idx, ptn_info in enumerate(partition_info):
        metadata_d = {
            "support": ptn_info["frequency"],
        }
        kwargs = {
            "metadata_d": metadata_d,
        }
        kwargs["subsets"] = ptn_info["subsets"]
        partition = partition_factory(**kwargs)
        partition._origin_size = len(partition_info)
        partition._origin_offset = ptn_idx
        yield partition


