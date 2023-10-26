## Introduction

`piikun` is a Python package for the analysis and visualization of species delimitation models in an information theoretic framework that provides a true distance or metric space for these models based on the variance of information criterion of [(Meila, 2007)]().
The name ``piikun`` is from a [Kumeyaay](https://en.wikipedia.org/wiki/Kumeyaay_language) ([Ipai](https://en.wikipedia.org/wiki/Ipai_language)) word for "[sparrowhawk](https://livingdictionaries.app/iipay-aa/entry/JIZpvX7ajl8gxzwVbCL5)", in homage to the indigenous people of Southern California, on whose land I live and work and has become my home.

The species delimitation models being analyzed may be generated by any inference package, such as [BP&P](flouri-2018-species-tree), [SNAPP](https://www.beast2.org/snapp/), [DELINEATE](https://github.com/jsukumaran/delineate) etc., or constructed based on taxonomies or classifications based on conceptual descriptions in literature, geography, folk taxonomies, etc.
Regardless of source or basis, each species delimitation model can be considered a *partition* of taxa or lineages and thus can be represented in a dedicated and widely-supported data exchange format, ["`SPART-XML`"](@miralles-2022-spart-versatile), which `piikun` takes as one of its input formats, in addition to DELINEATE.

For every collection of species delimitation models, `piikun` generates a set of partition profiles, partition comparison tables, and a suite of graphical plots visualizing data in these tables.
The partition profiles report unitary information theoretic and other statistics for each of the species delimitation partition, including the probability and entropy of each partition following [@meila-2007-comparing-clusterings].

The partition comparison tables, on the other hand, provide a range of bivariate statistics for every distinct pair of partitions, including the mutual information, joint entropy, etc., as well as a information theoretic distance statistics are true metrics on the space of species distribution models: the variance of information [@meila-2007-comparing-clusterings] and the normalized joint variance of information distance [@vinh-2010-information-theoretic].

## Installation

### Installing from the GitHub Repositories

We recommend that you install directly from the main GitHub repository using pip (which works with an Anaconda environment as well):

```
$ python3 -m pip install --user --upgrade git+https://github.com/jeetsukumaran/piikun.git
```

or

```
$ python3 -m pip install --user --upgrade git+git://github.com/jeetsukumaran/piikun.git
```

## Usage

Following the generation, definition or conceptualization of the species delimitation partitions (see below), a typical analyses would consist of:

- **Compiling** the partition definitions and associated information from the species delimitation model analyses results.

- **Analyzing** the partition data to calculate the various measures of information for each partition and associated distances between each distinct pair of partitions.

- **Visualizing** the results.

### Generating the Partitions

A typical `piikun` analysis starts *after* generating or otherwise conceptualizing the source or sources species delimitation models to be analyzed.
This can be the results of one or more DELINEATE, BPP, or some other species delimitation analyses run on the a dataset that, while needing to be invariant in terms of lineage/population concepts, may vary based on the statistical datak
So, for example, one may imagine a data set consisting 40 samples that we are organizing into higher level units (e.g., a sample of individuals into demes in a BPP A11 analysis, or a sample of putative demes into species units in a BPP A10 or DELINEATE analysis).
We may use various different kinds of data to represent these lineages in these analysis and other related ones.
We might run multiple different BPP analyses using different markers, or we may be comparing our work on one set of markers to another that used a different set (or ran under different constraints).

We can compare the species delimitation results across all these analyses because the basic concepts being organized into deme or species "blocks" or "subsets" are consistent across all of them.
This is why we can also include and compare the results from, not just across different genetic datasets of the same samples, but also across analyses that integrate morphological data, or take into account arrangements described in literature, speculatively, etc.
As long as the the "elements" of the subsets of the partitions map to the same concept, the resulting partitions can be compared, regardless of method of conceptualization or identification.

### Quick Start: Single-Step Analysis

The program ``piikun-analyze`` connects together three separate programs (discussed individually below): ``piikun-compile``, ``piikun-evaluate``, and ``piikun-visualize``.
You can run the entire pipeline on one of the provided example datasets by:

```
$ cd examples/
$ piikun-analyze -f delineate data/maddison-2020/lionepha-p090-hky.mcct-mean-age.delimitation-results.json
```


### Detailed Toolchain Pipelines

### Compiling the Partitions: Extracting the Partition Data from the Species Delimitation Model Sources

``piikun-compile`` is a command-line program that parses and formats data about species delimitation models from various sources and concats them in a common ``.json``-formatted datastore.
``piikun-compile`` takes as its input a collection of partitions in one of the following data formats, specified using the ``-f`` or ``--format`` options:

-   "``delineate``": [DELINEATE](https://github.com/jsukumaran/delineate) results

    This specifies the primary ``.json`` results files produces by DELINEATE as sources.

    ```bash
    $ piikun-compile -f delineate delineate-results.json
    $ piikun-compile --format delineate delineate-results.json
    ```

-   "``bpp-a11``: BPP (A11 mode) format

    This specifies the output log files from BPP as sources.

    ```bash
    $ piikun-compile -f bpp-a11 output.txt
    $ piikun-compile --format bpp-a11 output.txt
    ```

-   "``spart-xml``": SPART-XML

    This specifies the "SPART"-format XML as sources.

    ```bash
    $ piikun-compile -f spart-xml data.xml
    $ piikun-compile --format spart-xml data.xml
    ```

- "``json-dict``": Generic JSON dictionary

    This specifies the sources will be dictionaries in JSON format, with a specific set of keys/elements (see below for details).

    ```bash
    $ piikun-compile -f json-dict data.json
    $ piikun-compile --format json-dict data.json
    ```

- "``json-list``": Generic JSON list (of lists)

    This specifies the sources will be lists of lists in JSON format (see below for details).

    ```bash
    $ piikun-compile -f json-dict data.json
    $ piikun-compile --format json-dict data.json
    ```

The output file names and paths can be specified by using the ``-o``/``--output-title`` and ``-O``/``--output-directory`` options.

Additional information can be added using the "``--set-property``" flag.
For example, the following adds information regarding the source that can be referenced in visualizations and analysis downstream:

```bash
$ piikun-compile \
    -f delineate delineate-results.json \
    --set-property n_genes:143 \
    --set-property hypothesis:geographical \
```

See ``--help`` for details on this and other options.

### Collating and Combining Multiple Sources

The data files produced by ``piikun-compule`` can be analyzed by ``piikun-evaluate`` individually directly.
To analyze data from multiple source formats you will use ``piikun-compile`` on each source *type* separately, generating a ``pikun`` partition JSON data file, for each one, and then run ``piikun-compile`` on all of these results to generate a single dataset.

```bash
# Produces: ``delineate1-results.partitions.json``, ``delineate2-results.partitions.json``
$ piikun-compile -f delineate delineate1-results.json delineate2-results.json

# Produces: ``bpp1.out.partitions.json``, ``bpp2.out.partitions.json``
$ piikun-compile -f bpp-a11 bpp1.out.txt bpp2.out.txt

# Produces unified dataset for analysis: "``concated-data.partitions.json``"
$ piikun-compile \
    delineate1-results.partitions.json \
    delineate2-results.partitions.json \
    bpp1.out.partitions.json \
    bpp2.out.partitions.json \
    -o concated-data
```

See ``--help`` for details on this and other options, such as setting the output file names and paths using the ``-o``/``--output-title`` and ``-O``/``--output-directory``, etc.


### ``piikun-evaluate``: Calculate Statistics and Distances

This command carries out the main calculations of this package.
It takes as its input the ``.partitions.json`` data file produced by ``piikun-compile``.

```bash
# Produces: ``delineate1-results.partitions.json``, ``delineate2-results.partitions.json``
$ piikun-compile -f delineate delineate1-results.json delineate2-results.json

# Produces: ``bpp1.out.partitions.json``, ``bpp2.out.partitions.json``
$ piikun-compile -f bpp-a11 bpp1.out.txt bpp2.out.txt

# Independent/separate comparative analysis of species
# delimitation models from multiple sources
$ piikun-evaluate delineate1-results.partitions.json
$ piikun-evaluate delineate2-results.partitions.json
$ piikun-evaluate bpp1.out.partitions.json
$ piikun-evaluate bpp2.out.partitions.json

# Single joint analysis of species delimitation models
# from multiple sources

# Combine species delimitation models from multiple sources
# into single data file: ``concated-data.partitions.json``
$ piikun-compile \
    delineate1-results.partitions.json \
    delineate2-results.partitions.json \
    bpp1.out.partitions.json \
    bpp2.out.partitions.json \
    -o concated-data

# Joint/single comparative analysis of species
# delimitation models from multiple sources
$ piikun-evaluate concated-data.partitions.json


```

```bash
$ piikun-evaluate \
    -o project42 \
    -O analysis_dir \
    data.partitions.json
$ piikun-evaluate \
    --output-title project42 \
    --output-directory analysis_dir \
    data.partitions.json
```

See ``--help`` for details on this and other options, such as setting the output file names and paths using the ``-o``/``--output-title`` and ``-O``/``--output-directory``, etc.

-   The number of partitions can are read from the input set can be restricted to the first $n$ partitions using the ``--limit-partitions`` option:

    ```bash
    $ piikun-evaluate \
        --format delineate \
        --output-title project42 \
        --output-directory analysis_dir \
        --limit-partitions 10 \
        delineate-results.json
    ```

    This is option is particularly useful when the number of partitions in the input is large and/or most of the partitions in the input set may not be of interest.
    For e.g., a typical [DELINEATE](https://github.com/jsukumaran/delineate) analysis may generate hundreds if not thousands of partitions, and most of these are low-probability ones of not much practical interest.
    Using the ``--limit`` flag will focus on just the subset of interest, which will help with computation time and resources.

#### Output

``piikun-evaluate`` will generate two data files (named and located based on the ``-o``/``--output-title`` and ``-O``/``--output-directory`` options):

- ``output-directory/output-title-profiles.tsv``
- ``output-directory/output-title-comparisons.tsv``

These files provide univariate and a mix of univariate and bivariate statistics, respectively, for the partitions.

Both of these files can be directly loaded as a PANDAS data frame for more detailed analysis:

```bash
>>> import pandas as pd
>>> df1 = pd.read_json(
...     "output-directory/output-title-comparisons.json",
... )
```

The ``__distances`` file includes the variance of information distance statistics: ``vi_distance`` and ``vi_normalized_kraskov``.

## Reference



### Standard Workflow Tool Chain

| Command              | Input                       | Output                                |
|----------------------|-----------------------------|---------------------------------------|
| ``piikun-compile``   | (Various)                   | ``<title>__partitions.json``          |
| ``piikun-evaluate``  | ``<title>-partitions.json`` | ``<title>__profiles.json``            |
|                      |                             | ``<title>__distances.json``           |
| ``piikun-visualize`` | ``<title>-distances.json``  | ``<title>__<visualization-name>.html`` |
|                      |                             | ``<title>__<visualization-name>.jpg``  |
|                      |                             | ``<title>__<visualization-name>.pdf``  |
|                      |                             | ``<title>__<visualization-name>.png``  |



### Internal Data Formats


#### ``piikun`` or ``json-dicts``


#### ``nested-lists``

``` json
[
    [["pop1", "pop2", "pop3", "pop4"]],
    [["pop1"], ["pop2", "pop3", "pop4"]],
    [["pop1", "pop2"], ["pop3", "pop4"]],
    [["pop2"], ["pop1", "pop3", "pop4"]],
    [["pop1"], ["pop2"], ["pop3", "pop4"]],
    [["pop1", "pop2", "pop3"], ["pop4"]],
    [["pop2", "pop3"], ["pop1", "pop4"]],
    [["pop1"], ["pop2", "pop3"], ["pop4"]],
    [["pop1", "pop3"], ["pop2", "pop4"]],
    [["pop3"], ["pop1", "pop2", "pop4"]],
    [["pop1"], ["pop3"], ["pop2", "pop4"]],
    [["pop1", "pop2"], ["pop3"], ["pop4"]],
    [["pop2"], ["pop1", "pop3"], ["pop4"]],
    [["pop2"], ["pop3"], ["pop1", "pop4"]],
    [["pop1"], ["pop2"], ["pop3"], ["pop4"]]
]
```













