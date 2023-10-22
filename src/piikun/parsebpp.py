#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re

patterns = {
    "alignment_ntax_nchar": re.compile("^\s*(\d+)\s*(\d+)\s*$", re.MULTILINE),
    "alignment_sequence": re.compile("^\^(\S+).*$", re.MULTILINE),
    "a11_section_b": re.compile(r"\(B\)\s*(\d+) species delimitations .*$", re.MULTILINE),
    "spd_pattern_end": re.compile(r"\(C\)\s*", re.MULTILINE),
    "a11_treemodel_entry": re.compile(r"^(\d+) ([0-9.]+) ([0-9.]+) (\d+) (.*;).*$"),
    "strip_tree_tokens": re.compile(r"[\(\),\s;]+"),
}
# a11_section_b = re.compile("^\(B\).*$")
# a11_treemodel_entry = re.compile(r"^\s*(\d+)\s+([0-9.])\s+([1-9.])\s+(\d)\s+(.*;)\s+(.*)$")
# a11_treemodel_entry = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(.*;)\s+(.*)$")


def check_for_overlapping_labels(labels):
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i != j and label1 in label2:
                return True, (label1, label2)
    return False, None

def parse_species_delimitation_data(labels, src_data):
    labels = sorted(labels, key=len, reverse=True)
    in_relevant_section = False
    parsed_data = []
    n_partitions_expected = None
    for line in src_data.split("\n"):
        line = line[:-1].strip()
        if not line:
            continue
        m = a11_section_b.match(line)
        if m:
            in_relevant_section = True
            n_partitions_expected = int(m[1])
            continue
        if in_relevant_section and spd_pattern_end.match(line):
            break
        # Parse lines within the relevant section.
        if in_relevant_section:
            assert n_partitions_expected
            parts = line.strip().split()
            frequency = float(parts[1])
            num_subsets = int(parts[2])
            species_subsets_str = " ".join(parts[3:]).strip("()")
            species_subsets = []
            current_subset = []
            temp_label = ""
            for char in species_subsets_str:
                if char == ' ':
                    if current_subset:
                        species_subsets.append(current_subset)
                        current_subset = []
                else:
                    temp_label += char
                    if temp_label in labels:
                        current_subset.append(temp_label)
                        temp_label = ""
            if current_subset:
                species_subsets.append(current_subset)
            assert len(species_subsets) == num_subsets
            parsed_data.append({
                "frequency": frequency,
                "n_subsets": num_subsets,
                "subsets": species_subsets
            })
    assert len(parsed_data) == n_partitions_expected
    return parsed_data

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

    return species_info, partition_info

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
