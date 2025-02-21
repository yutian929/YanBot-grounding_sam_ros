#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"
swinb_cogcoor_url="${BASE_URL}v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"



# Download each of the four checkpoints using wget
echo "Downloading groundingdino_swint_ogc.pth checkpoint..."
wget $swint_ogc_url || { echo "Failed to download checkpoint from $swint_ogc_url"; exit 1; }

echo "Downloading groundingdino_swinb_cogcoor.pth checkpoint..."
wget $swinb_cogcoor_url || { echo "Failed to download checkpoint from $swinb_cogcoor_url"; exit 1; }

echo "All groundingdino checkpoints are downloaded successfully."

# Define the URLs for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything/"
sam_h_url="${BASE_URL}sam_vit_h_4b8939.pth"
sam_l_url="${BASE_URL}sam_vit_l_0b3195.pth"
sam_b_url="${BASE_URL}sam_vit_b_01ec64.pth"

# Download each of the four checkpoints using wget
echo "Downloading sam_vit_h_4b8939.pth checkpoint..."
wegt $sam_h_url || { echo "Failed to download checkpoint from $sam_h_url"; exit 1; }

echo "Downloading sam_vit_l_0b3195.pth checkpoint..."
wget $sam_l_url || { echo "Failed to download checkpoint from $sam_l_url"; exit 1; }

echo "Downloading sam_vit_b_01ec64.pth checkpoint..."
wget $sam_b_url || { echo "Failed to download checkpoint from $sam_b_url"; exit 1; }

echo "All sam checkpoints are downloaded successfully."