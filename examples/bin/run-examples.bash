#! /bin/bash

## Basic tutorials on using Piikun

## OK, let's get started!
## First, let's checkout the project in new workspace

## !!! show walkthroughs of using virtualenv and conda on how to install
## Activity should start with the student in their home direcotry
## and end with student being in a ``examples/var`` subdirectory of the project
## as cloned from "git@github.com:jeetsukumaran/piikun.git",
## directory eg., `~/workspaces/code/piikun/examples/var/`, with the var directory created
## if needed safely ("mkdir -p").


export DATABANK_ROOT=../data/

piikun-compile --format nested-lists ${DATABANK_ROOT}/nested-lists/ex1.json
piikun-evaluate ex1__partitions.json


