#!/usr/bin/env zsh

set -x
rm -rf {software,data}.tgz
git ls-files \
  | tar caf software.tgz -T-
find data \
  | tar caf data.tgz -T-

check_tar() {
  tar xf $1 -O | grep -i '\(boldt\|brendon\|cmu\|carnegie\)'
}

check_tar software.tgz
check_tar data.tgz
