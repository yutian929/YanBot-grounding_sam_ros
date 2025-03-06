#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define the URLs for the checkpoints
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"
swinb_cogcoor_url="${BASE_URL}v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

# Function to download a file if it doesn't already exist
download_if_not_exists() {
    local url=$1
    local filename=$(basename $url)
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename checkpoint..."
        wget $url || { echo "Failed to download checkpoint from $url"; exit 1; }
    else
        echo "$filename already exists, skipping download."
    fi
}

# Download each of the checkpoints
download_if_not_exists $swint_ogc_url
download_if_not_exists $swinb_cogcoor_url

echo "All groundingdino checkpoints are downloaded successfully."

# Define the URLs for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything/"
sam_h_url="${BASE_URL}sam_vit_h_4b8939.pth"
sam_l_url="${BASE_URL}sam_vit_l_0b3195.pth"
sam_b_url="${BASE_URL}sam_vit_b_01ec64.pth"

# Download each of the checkpoints
download_if_not_exists $sam_h_url
download_if_not_exists $sam_l_url
download_if_not_exists $sam_b_url

echo "All sam checkpoints are downloaded successfully."