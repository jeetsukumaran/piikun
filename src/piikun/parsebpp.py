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

    # species_labels = [taxon.label for taxon in guide_tree.taxon_namespace]
    # ancestral_node_subsets_map = {}
    # for anc_idx, desc in enumerate(ancestral_node_label_id_map):
    #     subsets = parse_species_subsets(
    #         species_subsets_str=desc,
    #         lineage_labels=species_labels,
    #     )
    #     ancestral_node_subsets_map[anc_idx] = subsets
    # print(ancestral_node_subsets_map)
    # assert len(ancestral_node_subsets_map) == len(ancestral_node_label_id_map)
    # for desc, subsets in zip(ancestral_node_label_id_map, ancestral_node_subsets_map.values()):
    #     print(f"{desc} --> {subsets}")
    # print(species_delimitation_model_defs)

# def find_terminal_population_clades(tree):
#     """
#     Takes a tree with nodes *annotated" to indicate whether or not they should
#     be collapsed and identifies the set of 'terminal population clades'.

#     'Terminal population clades' are substrees descending from nodes with no
#     ancestors in a collapsed state AND either: (a) no child nodes [i.e., a
#     leaf] or (b) no descendent nodes in a non-collapsed or open states.
#     Nodes corresponding to terminal populations themselves are, by definition,
#     need to in a collapsed state unless they are leaves.

#     Returns a dictionary mapping terminal population nodes to leaf nodes descending
#     from the terminal population nodes.
#     """
#     # ensure if parent is collapsed, all children are collapsed
#     for gnd in tree.preorder_node_iter():
#         if gnd.parent_node is not None and gnd.parent_node.annotations["is_collapsed"].value:
#             gnd.annotations["is_collapsed"] = True
#         else:
#             gnd.annotations["is_collapsed"] = False
#     # identify the lowest nodes (closest to the tips) that are open, and
#     # add its children if the children are (a) leaves; or (b) themselves
#     # are closed
#     terminal_population_clades = {}
#     lineage_population_clade_map = {}
#     for nd in tree.postorder_node_iter():
#         if nd.annotations["is_collapsed"].value:
#             continue
#         for child in nd.child_node_iter():
#             if child.is_leaf():
#                 terminal_population_clades[child] = set([child])
#                 lineage_population_clade_map[child] = child
#             elif child.annotations["is_collapsed"].value:
#                 terminal_population_clades[child] = set([desc for desc in child.leaf_iter()])
#                 for lnd in terminal_population_clades[child]:
#                     lineage_population_clade_map[lnd] = child
#     for nd in tree:
#         if nd not in terminal_population_clades:
#             nd.annotations["population_id"] = "0"
#             continue
#         if nd.is_leaf():
#             nd.annotations["population_id"] = nd.taxon.label
#         else:
#             nd.annotations["population_id"] = "+".join(desc.taxon.label for desc in nd.leaf_iter())
#         # print("{}: {}".format(nd.annotations["population"], len(terminal_population_clades[nd])))
#     return terminal_population_clades, lineage_population_clade_map

def find_terminal_population_clades(tree):
    """
    Takes a tree with nodes *annotated" to indicate whether or not they should
    be collapsed and identifies the set of 'terminal population clades'.

    'Terminal population clades' are substrees descending from nodes with no
    ancestors in a collapsed state AND either: (a) no child nodes [i.e., a
    leaf] or (b) no descendent nodes in a non-collapsed or open states.
    Nodes corresponding to terminal populations themselves are, by definition,
    need to in a collapsed state unless they are leaves.

    Returns a dictionary mapping terminal population nodes to leaf nodes descending
    from the terminal population nodes.
    """
    # ensure if parent is collapsed, all children are collapsed
    for gnd in tree.preorder_node_iter():
        if gnd.parent_node is not None and gnd.parent_node.annotations["is_collapsed"].value:
            gnd.annotations["is_collapsed"] = True
        else:
            gnd.annotations["is_collapsed"] = False
    # identify the lowest nodes (closest to the tips) that are open, and
    # add its children if the children are (a) leaves; or (b) themselves
    # are closed
    terminal_population_clades = {}
    lineage_population_clade_map = {}
    for nd in tree.postorder_node_iter():
        if nd.annotations["is_collapsed"].value:
            continue
        for child in nd.child_node_iter():
            if child.is_leaf():
                terminal_population_clades[child] = set([child])
                lineage_population_clade_map[child] = child
            elif child.annotations["is_collapsed"].value:
                terminal_population_clades[child] = set([desc for desc in child.leaf_iter()])
                for lnd in terminal_population_clades[child]:
                    lineage_population_clade_map[lnd] = child
    for nd in tree:
        if nd not in terminal_population_clades:
            nd.annotations["population_id"] = "0"
            continue
        if nd.is_leaf():
            nd.annotations["population_id"] = nd.taxon.label
        else:
            nd.annotations["population_id"] = "+".join(desc.taxon.label for desc in nd.leaf_iter())
        # print("{}: {}".format(nd.annotations["population"], len(terminal_population_clades[nd])))
    return terminal_population_clades, lineage_population_clade_map

# def calculate_bpp_full_species_tree(
#         src_tree_string,
#         guide_tree,
#         population_probability_threshold=0.95):
#     import dendropy
#     from dendropy.calculate import treecompare
#     # Logic:
#     # - Any internal node label is assumed to be a bpp annotation in the
#     #   form of "#<float>" indicating the posterior probability of the node.
#     # - If not found, assigned posterior probability of 0 if internal node or 1 if leaf.
#     # - In post-order, collapse all nodes with less than threshold pp
#     # - Revisit tree, assign taxon labels
#     tree0 = dendropy.Tree.get(
#             data=src_tree_string,
#             schema="newick",
#             rooting="force-rooted",
#             suppress_external_node_taxa=False,
#             suppress_internal_node_taxa=True,
#             taxon_namespace=guide_tree.taxon_namespace,
#             )
#     tree0.encode_bipartitions()
#     guide_tree.encode_bipartitions()
#     try:
#         diff = treecompare.symmetric_difference(tree0, guide_tree)
#         assert diff == 0
#     except AssertionError:
#         print("[BPP tree]{}".format(tree0.as_string("newick")))
#         print("[Guide tree]{}".format(guide_tree.as_string("newick")))
#         print(f"RF = {diff}")
#         raise
#     for nd in tree0:
#         edge_len = guide_tree.bipartition_edge_map[nd.edge.bipartition].length
#         if edge_len is not None and nd.edge.length is not None:
#             nd.edge.length = edge_len
#         if nd.is_leaf():
#             nd.pp = 1.0
#             nd.label = nd.taxon.label
#         elif nd.label:
#             nd.pp = float(nd.label[1:])
#             nd.label = None
#         else:
#             nd.pp = 0.0

#     tree1 = dendropy.Tree(tree0)
#     tree1.taxon_namespace = dendropy.TaxonNamespace()

#     nodes_to_process = [tree1.seed_node]
#     while nodes_to_process:
#         nd = nodes_to_process.pop(0)
#         if nd.is_leaf():
#             pass
#         elif nd.pp < population_probability_threshold:
#             desc_tips = []
#             for sub_nd in nd.leaf_iter():
#                 desc_tips.append(sub_nd)
#             # nd.set_child_nodes(new_children)
#             len_to_add = 0.0
#             subnode = desc_tips[0]
#             while subnode is not nd:
#                 if subnode.edge.length is not None:
#                     len_to_add += subnode.edge.length
#                 subnode = subnode.parent_node
#             child_labels = [c.label for c in desc_tips]
#             nd.label = "+".join(child_labels)
#             if nd.edge.length is not None and len_to_add is not None:
#                 nd.edge.length += len_to_add
#             nd.child_labels = child_labels
#             nd.clear_child_nodes()
#             guide_tree_edge = guide_tree.bipartition_edge_map[nd.edge.bipartition]
#             guide_tree_edge.head_node.is_collapsed = True
#         else:
#             for sub_nd in nd.child_node_iter():
#                 nodes_to_process.append(sub_nd)
#     collapsed_nodes = set()
#     for gnd in guide_tree.postorder_node_iter():
#         if getattr(gnd, "is_collapsed", False):
#             len_to_add = 0.0
#             desc_nd = gnd
#             while True:
#                 try:
#                     desc_nd = desc_nd._child_nodes[0]
#                     if desc_nd.edge.length is not None and len_to_add is not None:
#                         len_to_add += desc_nd.edge.length
#                 except IndexError:
#                     break
#             if gnd.edge.length is not None and len_to_add is not None:
#                 gnd.edge.length += len_to_add
#             for xnd in gnd.postorder_iter():
#                 if xnd is gnd:
#                     continue
#                 xnd.edge.length = 1e-10
#             gnd.annotations["is_collapsed"] = True
#         else:
#             gnd.annotations["is_collapsed"] = False
#     for gnd in guide_tree.preorder_node_iter():
#         if gnd.parent_node is not None and gnd.parent_node.annotations["is_collapsed"].value:
#             gnd.annotations["is_collapsed"] = True
#     for nd in tree1.leaf_node_iter():
#         nd.taxon = tree1.taxon_namespace.require_taxon(label=nd.label)
#         nd.label = None

#     return guide_tree, tree1

class BppParser:

    def __init__(self):
        pass

    def parse_bpp_a10(self, source_stream, partition_factory):
        pass

