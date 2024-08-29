#!/bin/bash

set -xe

name=$(date +%s)
mkdir -p .backups
gzip store.sqlite
mv -i store.sqlite.gz .backups/$name.sqlite.gz
