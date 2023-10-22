#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re

patterns = {
    "alignment_ntax_nchar": re.compile("^\s*(\d+)\s*(\d+)\s*.*$", re.MULTILINE),
    "alignment_sequence": re.compile("^(.*?\^)?(\S+).*$", re.MULTILINE),
    "a11_section_b": re.compile(r"\(B\)\s*(\d+) species delimitations .*$", re.MULTILINE),
    "spd_pattern_end": re.compile(r"\(C\)\s*", re.MULTILINE),
    "a11_treemodel_entry": re.compile(r"^(\d+) ([0-9.]+) ([0-9.]+) (\d+) (.*;).*$"),
    "strip_tree_tokens": re.compile(r"[\(\),\s;]+"),
    "a10-species-delimitation-models": re.compile(r"^Number of species-delimitation models = (\d+).*$"),
    "a10-species-delimitation-model-row": re.compile(r"^ ?(\d+) +([01]+) +([0-9Ee\-.])+ +([0-9Ee\-.])"),
    "a10-species-delimitation-model-header": re.compile(r"^ *model +prior +posterior.*$"),
    "a10-species-delimitation-model-row": re.compile(r"^ *(\d+?) +([01]+) +([0-9Ee\-.]+) +([0-9Ee\-.]+).*$"),
}
# a11_section_b = re.compile("^\(B\).*$")
# a11_treemodel_entry = re.compile(r"^\s*(\d+)\s+([0-9.])\s+([1-9.])\s+(\d)\s+(.*;)\s+(.*)$")
# a11_treemodel_entry = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(.*;)\s+(.*)$")

# def decompose_model_subset(model_subset_str: str, original_labels: list) -> list:
#     """Decompose a model subset definition string into its constituent labels."""
#     # Sort the original labels by length in descending order
#     sorted_labels = sorted(original_labels, key=len, reverse=True)
#     # Initialize an empty list to store the decomposed labels
#     decomposed_labels = []
#     # Iterate over the sorted labels to match and decompose the model subset string
#     for label in sorted_labels:
#         while label in model_subset_str:
#             decomposed_labels.append(label)
#             model_subset_str = model_subset_str.replace(label, '', 1)
#     return decomposed_labels

# def decompose_model_subset(model_subset_str: str, original_labels: list) -> list:
#     """Decompose a model subset definition string into its constituent labels."""

#     # Sort the original labels by length in descending order
#     sorted_labels = sorted(original_labels, key=len, reverse=True)

#     # Initialize an empty list to store the decomposed labels
#     decomposed_labels = []

#     # Iterate over the sorted labels to match and decompose the model subset string
#     for label in sorted_labels:
#         while label in model_subset_str:
#             decomposed_labels.append(label)
#             model_subset_str = model_subset_str.replace(label, '', 1)

#     return decomposed_labels

def parse_model_subset_str(model_subset_str: str, original_labels: list) -> list:
    """Decompose a model subset definition string into its constituent labels."""

    # Sort the original labels by length in descending order
    sorted_labels = sorted(original_labels, key=len, reverse=True)

    # Initialize an empty list to store the decomposed labels
    decomposed_labels = []

    # Iterate over the sorted labels to match and decompose the model subset string
    for label in sorted_labels:
        # print("----")
        # print(label)
        # print(model_subset_str)
        # print(label in model_subset_str)
        if label in model_subset_str:
            decomposed_labels.append(label)
            model_subset_str = model_subset_str.replace(label, '', 1)

    return decomposed_labels

def process_bpp_a10_results(file_contents: str) -> list:
    """Process the BP&P A10 analysis results to extract and expand species delimitation models."""

    # Split the file contents into lines
    lines = file_contents.split("\n")

    # Find the start and end of the "Order of ancestral nodes:" section
    node_order_start = lines.index("Order of ancestral nodes:") + 1
    newick_tree_start = next(i for i, line in enumerate(lines[node_order_start:], start=node_order_start) if line.startswith("("))

    # Extract the labels of the leaves for each node
    node_leaf_labels = lines[node_order_start:newick_tree_start]

    # Define a helper function to expand the model encoding into subsets
    def expand_model_encoding(encoding):
        """Expand the model encoding into subsets based on the node leaf labels."""
        subsets = []
        idx = 0
        while idx < len(encoding):
            # If a '1' is found, it indicates the node splits the leaf set
            if encoding[idx] == '1':
                idx += 1
                continue
            # If a '0' is found, all of the leaves form a single subset
            elif encoding[idx] == '0':
                # subsets.append(list(node_leaf_labels[idx].strip().split()))
                subsets.append(node_leaf_labels[idx].strip())
                idx += 1
            # If neither '1' nor '0', it's a leaf label
            else:
                # subsets.append([encoding[idx]])
                subsets.append(encoding[idx])
                idx += 1
        return subsets

    # Extract the table with posterior probabilities
    posterior_table_start = lines.index('     model    prior    posterior') + 1

    # Initialize an empty list to store the dictionaries
    model_data = []

    # Iterate over the lines starting from the identified start of the table
    for line in lines[posterior_table_start:]:
        # Break the loop if an empty line is encountered (assumed end of table)
        if not line.strip():
            break

        # Split the line into its parts and extract the data
        parts = line.split()
        model = {
            "model_id": int(parts[0]),
            "model_encoding": parts[1],
            "prior_probability": float(parts[2]),
            "posterior_probability": float(parts[3]),
            "model_subsets_str": expand_model_encoding(parts[1])
        }
        model_data.append(model)

    return model_data

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

def check_for_overlapping_labels(labels):
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i != j and label1 in label2:
                return True, (label1, label2)
    return False, None

# def parse_species_delimitation_data(labels, src_data):
#     labels = sorted(labels, key=len, reverse=True)
#     in_relevant_section = False
#     parsed_data = []
#     n_partitions_expected = None
#     for line in src_data.split("\n"):
#         line = line[:-1].strip()
#         if not line:
#             continue
#         m = a11_section_b.match(line)
#         if m:
#             in_relevant_section = True
#             n_partitions_expected = int(m[1])
#             continue
#         if in_relevant_section and spd_pattern_end.match(line):
#             break
#         # Parse lines within the relevant section.
#         if in_relevant_section:
#             assert n_partitions_expected
#             parts = line.strip().split()
#             frequency = float(parts[1])
#             num_subsets = int(parts[2])
#             species_subsets_str = " ".join(parts[3:]).strip("()")
#             species_subsets = []
#             current_subset = []
#             temp_label = ""
#             for char in species_subsets_str:
#                 if char == ' ':
#                     if current_subset:
#                         species_subsets.append(current_subset)
#                         current_subset = []
#                 else:
#                     temp_label += char
#                     if temp_label in labels:
#                         current_subset.append(temp_label)
#                         temp_label = ""
#             if current_subset:
#                 species_subsets.append(current_subset)
#             assert len(species_subsets) == num_subsets
#             parsed_data.append({
#                 "frequency": frequency,
#                 "n_subsets": num_subsets,
#                 "subsets": species_subsets
#             })
#     assert len(parsed_data) == n_partitions_expected
#     return parsed_data

# test_contents = [
#     "(VARIABLE)",
#     "(B)  8 species delimitations & their posterior probabilities",
#     "  99180   0.99180   8 (Ge Ta Tr Al El MX CA SA)",
#     "    801   0.00801   1 (GeTaTrAlElMXCASA)",
#     "      6   0.00006   6 (Ge Ta Tr Al El MXCASA)",
#     "      4   0.00004   3 (GeTa Tr AlElMXCASA)",
#     "      3   0.00003   5 (Ge Ta Tr AlMXCASA El)",
#     "      2   0.00002   2 (GeTaTr AlElMXCASA)",
#     "      2   0.00002   4 (Ge Ta Tr AlElMXCASA)",
#     "      2   0.00002   7 (Ge Ta Tr Al El MX CASA)",
#     "(C)  delimited species & their posterior probabilities",
#     "(VARIABLE)"
# ]

def parse_bpp_ctl(control_filepath):
    species_info = {}
    with open(control_filepath, 'r') as f:
        content = f.read()
    # Remove comments and condense into a list of non-empty, non-comment lines
    lines = [line.split('*')[0].strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('*')]
    # Find the line index containing 'species&tree'
    species_tree_line_index = next((i for i, line in enumerate(lines) if 'species&tree' in line), None)
    if species_tree_line_index is not None:
        # Extract the number of labels and the labels themselves
        n_and_labels_match = re.search(r'species&tree\s*=\s*(\d+)\s+([\w\s]+)', lines[species_tree_line_index])
        if n_and_labels_match:
            n = int(n_and_labels_match.group(1))
            labels = n_and_labels_match.group(2).split()[:n]
            # Validation: n should match the length of the labels list
            if n != len(labels):
                raise ValueError("The number 'n' does not match the length of the labels list.")
            species_info['n'] = n
            species_info['labels'] = labels
            # Extract sizes from the next non-blank line
            sizes_str = lines[species_tree_line_index + 1]
            sizes = list(map(int, sizes_str.split()))
            species_info['sizes'] = sizes
            # Extract the tree string from the second next non-blank line
            tree_str = lines[species_tree_line_index + 2]
            species_info['tree_string'] = tree_str
    return species_info

# def read_bpp_a11(
#     control_filepath,
#     bpp_output_filepath,
# ):
#     species_info = parse_bpp_ctl(control_filepath=control_filepath)
#     with open(bpp_output_filepath) as src:
#         src_data = src.read()
#     partition_info = parse_species_delimitation_data(
#         labels=species_info["labels"],
#         src_data=src_data,
#     )
#     return species_info, partition_info
# def read_bpp_a11(
#     bpp_output_filepath,
# ):
#     with open(bpp_output_filepath) as src:
#         return parse_bpp_a11(src)
#   return species_info, partition_info

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

def extract_bpp_output_posterior_guide_tree_string(source_text):
    # pretty dumb logic for now: just locates the last non-blank line
    # works with: bp&p Version 3.1, April 2015, in "10" mode, i.e. infer species delimitation with guide tree.
    lines = source_text.split("\n")
    result = None
    for idx, line in enumerate(lines[-1::-1]):
        if line:
            result = line
            break
    return result


# read_bpp_a11(
#     control_filepath="/home/jeetsukumaran/site/storage/workspaces/code/research/projects/20230715-species-delimitation-model-distances/piikun-paper-figures/20230819-diagnosis/20230829-01/data/bpp/chambers-and-hillis-2019/Unguided_BPP/unguided.bpp.ctl",
#     bpp_output_filepath="/home/jeetsukumaran/site/storage/workspaces/code/research/projects/20230715-species-delimitation-model-distances/piikun-paper-figures/20230819-diagnosis/20230829-01/data/bpp/chambers-and-hillis-2019/Unguided_BPP/unguided_out.txt",
# )
