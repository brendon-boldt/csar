#!/usr/bin/env bash

function _run {
  micromamba -n csar run python -m src $@
}

set -xe


function _main {
  ### Environment Setup ###
  # Feel free to use conda instead of micromamba (even if it is slower).
  micromamba env create -yf env.yml


  ### Synthetic Langauge Experiments ###
  _run analysis collect -n3
  # Add -o if you want to overwrite previously collected data
  

  ### Human Language Experiments ###
  _run script morpho-challenge -w
  # Image captions
  _run script coco -w
  _run script mt -w


  ### Emergent Language Experiments ###
  # The code to re-generate these emergent language datasets is not packaged in
  # the supplemental material but could be made avaialble upon request.
  _run script ec -w data/ec-vector/av.json
  # "bom" corresponds to sparse meanings
  _run script ec -w data/ec-vector/sparse.json
  for variant in ref setref concept; do
    _run script ec-mu -w data/ec-shapeworld/${variant}.jsonl
  done
  # Run the following to view the output of one of the scripts.
  # _run read save/scripts/coco.pkl


  ### Analysis ###
  _run analysis report
}


_main

# To run the morpheme induction algorithm on arbitrary data, see README.md
