#!/usr/bin/env bash

# Download, extract
dir="$1"
url="$2"
full="${url##*/}"
file="${full%%.*}"
dest="$dir/$file.mtx"
if [ ! -f "$dest" ]; then
  echo "Fetching $url ..."
  wget -q "$url"
  tar -xzf "$full"
  cp -r "$file"/* "$dir"/
  rm -rf "$file"
fi
