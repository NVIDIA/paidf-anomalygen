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

import torch
import torch.nn as nn
from torch.nn import functional as F


class StructureLoss(nn.Module):
    """Compute the weighted IoU loss and BCE loss for the global and local (pixel-level) restriction.

    References:
        [1] https://arxiv.org/abs/2006.11392
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() == 3:
            input = input.unsqueeze(1)
        target = target.to(dtype=input.dtype)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        wbce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        input = torch.sigmoid(input)
        inter = ((input * target) * weit).sum(dim=(2, 3))
        union = ((input + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class DualMiLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def compute_relation(self, feat, mask):
        dot_product = feat * mask
        norm_feat = torch.norm(feat, dim=-1, keepdim=True)
        norm_mask = torch.norm(mask, dim=-1, keepdim=True)
        normalized_dot_product = dot_product / (norm_feat * norm_mask)
        relation = torch.sum(normalized_dot_product, dim=-1)
        relation = F.relu(relation)
        return relation

    def compute_relation_dot(self, feat, mask):
        batch_size = feat.shape[0]
        relation = feat @ mask.transpose(1, 2)  # (bsz, h*w)
        relation_norm = F.normalize(relation.view(batch_size, -1))
        return relation_norm

    def compute_log(self, G_s):
        frobenius_norm_s = torch.norm(G_s, p="fro")  # ||G_s||_F
        frobenius_norm_s_squared = frobenius_norm_s**2  # ||G_s||_F^2
        log_frobenius_s = torch.log2(frobenius_norm_s_squared)  # log2(||G_s||_F^2)
        L_mi = log_frobenius_s

        return L_mi

    def compute_relation_loss(self, z1, z2, norm_f):
        # normlize
        batch_size = z1.shape[0]
        norm_z1 = F.normalize(z1.reshape(batch_size, -1))
        norm_z2 = F.normalize(z2.reshape(batch_size, -1))

        # compute gram matrix of z1, z2,
        G_z1 = torch.einsum("bx,dx->bd", norm_z1, norm_z1)
        G_z2 = torch.einsum("bx,dx->bd", norm_z2, norm_z2)
        G_f = torch.einsum("bx,dx->bd", norm_f, norm_f)
        G_tri = G_z1 * G_z2 * G_f

        # Norm gram matrice
        G_f = G_f / torch.trace(G_f)
        G_tri = G_tri / torch.trace(G_tri)

        # compute log loss
        loss_f, loss_tri = self.compute_log(G_f), self.compute_log(G_tri)
        loss_r = -loss_f + loss_tri
        return loss_r

    def compute_distill_loss(self, norm_f_t, norm_f_s):
        # compute gram matrix of z1, z2,
        G_t = torch.einsum("bx,dx->bd", norm_f_t, norm_f_t)
        G_s = torch.einsum("bx,dx->bd", norm_f_s, norm_f_s)
        G_ts = G_t * G_s

        # Norm gram matrice
        G_s = G_s / torch.trace(G_s)
        G_t = G_t / torch.trace(G_t)
        G_ts = G_ts / torch.trace(G_ts)

        # compute log loss
        loss_s, loss_t, loss_ts = (
            self.compute_log(G_s),
            self.compute_log(G_t),
            self.compute_log(G_ts),
        )
        loss_d = loss_s + loss_t - loss_ts
        return loss_d

    def forward(self, student, teacher, relation_model=None):
        """
        z_stu: size [batch_size, s_dim, h, w]
        z_tea: size [batch_size, t_dim, h, w]
        """
        feat_s, mask_s, _ = student
        feat_t, mask_t, _ = teacher
        relation_t = relation_model(feat_t, mask_t)  # [bsz, h*w]
        relation_s = relation_model(feat_s, mask_s)  # [bsz, h*w]

        loss_r = self.compute_relation_loss(z1=feat_t, z2=mask_t, norm_f=relation_t)
        loss_d = self.compute_distill_loss(norm_f_t=relation_t, norm_f_s=relation_s)

        loss = self.alpha * loss_r + self.beta * loss_d
        return loss
