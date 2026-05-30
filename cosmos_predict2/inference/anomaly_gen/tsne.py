# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cosmos_predict2.models.ag_modules.mask_encoder import build_mask_encoder
from sklearn.manifold import TSNE
import torch
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from imaginaire.utils import log

SEED = 1

config = {
    'pool_kernel': 1,
    'init_cfg': {
        'checkpoint': 'checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt'
    },
    'learnable_head': False,
}

def normalize(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32) # -1~1
    image = torch.from_numpy(image[np.newaxis, :, :, :])
    image = torch.permute(image, (0, 3, 1, 2))
    image = torchvision.transforms.Resize((518, 518))(image)
    return image

def get_feats(input_data, model):
    with torch.inference_mode():
        feat = model(input_data.to("cuda"))
    return feat.detach().to("cpu")

def  sample_by_tsne(img_filenames):
    # Build NVDINOV2
    NVDINOV2_model = build_mask_encoder(
        encoder_type = "nvdinov2",
        encoder_config = config,
        freeze = True
    ).to("cuda")

    # Load OK data
    train_full_OK_images = img_filenames
    
    # Normalize
    train_full_OK_images_inputs = [normalize(Image.open(file)) for file in train_full_OK_images]

    # Get features
    train_full_OK_images_feats = [get_feats(input_data, NVDINOV2_model) for input_data in tqdm(train_full_OK_images_inputs)]

    #  t-SNE
    real_train_OK_data_embeds = torch.concatenate(train_full_OK_images_feats)
    B, L, C = real_train_OK_data_embeds.shape
    X = real_train_OK_data_embeds.numpy().reshape(B, -1)
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', 
                    perplexity=10,
                    random_state=SEED).fit_transform(X)
    
    # DBSCAN Clustering
    clustering = DBSCAN(eps=2, min_samples=1).fit(X_embedded)
    total_clusters = clustering.labels_.max()
    log.info(f"Total clusters: {total_clusters} in {len(train_full_OK_images)} OK images")

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=X_embedded[:, 0], y=X_embedded[:, 1],
                            mode='markers',
                            name='Full train OK data',
                            text=train_full_OK_images,
                            marker=dict(size=3,color=clustering.labels_)))

    fig.update_layout(
        title=dict(text="Clustering result for OK data's t-SNE result", font=dict(size=12))
    )

    fig.write_html(f"tsne_analysis.html")

    # Categorize images by cluster labels
    train_full_OK_images_by_cluster = defaultdict(list)
    for image, label in zip(train_full_OK_images, clustering.labels_):
        train_full_OK_images_by_cluster[label].append(image)

    return train_full_OK_images_by_cluster
