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

import numpy as np


class MeanIoUMeter:
    """Computes and stores the average and current value."""

    def __init__(self, n_class: int, ignore_index: int = 255):
        """init."""
        self.initialized = False
        self.area_intersect = None
        self.area_union = None
        self.area_pred_label = None
        self.area_label = None
        self.count = None
        self.n_class = n_class
        self.ignore_index = ignore_index

    def initialize(
        self,
        area_intersect: np.ndarray,
        area_union: np.ndarray,
        area_pred_label: np.ndarray,
        area_label: np.ndarray,
    ):
        """Initialize counter variables for metric calculation."""
        self.area_intersect = area_intersect
        self.area_union = area_union
        self.area_pred_label = area_pred_label
        self.area_label = area_label
        self.count = 1
        self.initialized = True

    def add(
        self,
        area_intersect: np.ndarray,
        area_union: np.ndarray,
        area_pred_label: np.ndarray,
        area_label: np.ndarray,
    ):
        """Update MeanIoUMeter with new value and weight."""
        self.area_intersect += area_intersect
        self.area_union += area_union
        self.area_pred_label += area_pred_label
        self.area_label += area_label

    def update(
        self,
        area_intersect: np.ndarray,
        area_union: np.ndarray,
        area_pred_label: np.ndarray,
        area_label: np.ndarray,
    ):
        """Update MeanIoUMeter with new value and weight."""
        if not self.initialized:
            self.initialize(area_intersect, area_union, area_pred_label, area_label)
        else:
            self.add(area_intersect, area_union, area_pred_label, area_label)

    def clear(self):
        """Clear the initialized status."""
        self.initialized = False

    @staticmethod
    def intersect_and_union(
        pred_label: np.ndarray,
        label: np.ndarray,
        num_classes: int,
        ignore_index: int,
    ):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        # Values outside [0, num_classes) make np.bincount(minlength=num_classes)
        # return an array longer than num_classes; area_pred + area_label -
        # area_intersect then fails to broadcast mid-accumulation. Surface a
        # clear error instead of a cryptic broadcast crash.
        for name, arr in (("pred_label", pred_label), ("label", label)):
            if arr.size and (arr.min() < 0 or arr.max() >= num_classes):
                raise ValueError(
                    f"{name} contains values outside [0, {num_classes}) "
                    f"(min={int(arr.min())}, max={int(arr.max())}); "
                    "clip/remap labels or increase num_classes."
                )

        intersect = pred_label[pred_label == label]
        area_intersect = np.bincount(intersect, minlength=num_classes)
        area_pred_label = np.bincount(pred_label, minlength=num_classes)
        area_label = np.bincount(label, minlength=num_classes)
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        n_class: int,
    ):
        """Calculate evaluation metrics.

        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
        """
        total_area_intersect = total_area_intersect.astype(np.float32)
        total_area_union = total_area_union.astype(np.float32)
        total_area_pred_label = total_area_pred_label.astype(np.float32)
        total_area_label = total_area_label.astype(np.float32)

        # Avoid division by zero.
        iou = np.divide(
            total_area_intersect,
            total_area_union,
            out=np.zeros_like(total_area_intersect),
            where=total_area_union != 0,
        )
        precision = np.divide(
            total_area_intersect,
            total_area_pred_label,
            out=np.zeros_like(total_area_intersect),
            where=total_area_pred_label != 0,
        )
        recall = np.divide(
            total_area_intersect,
            total_area_label,
            out=np.zeros_like(total_area_intersect),
            where=total_area_label != 0,
        )
        f1 = np.divide(
            2 * recall * precision,
            recall + precision,
            out=np.zeros_like(recall),
            where=(recall + precision) != 0,
        )

        cls_iou = {}
        cls_precision = {}
        cls_recall = {}
        cls_f1 = {}
        for i in range(n_class):
            cls_iou["iou_" + str(i)] = np.nan_to_num(iou[i])
            cls_precision["precision_" + str(i)] = np.nan_to_num(precision[i])
            cls_recall["recall_" + str(i)] = np.nan_to_num(recall[i])
            cls_f1["f1_" + str(i)] = np.nan_to_num(f1[i])

        score_dict = {}
        score_dict.update(cls_iou)
        score_dict.update(cls_f1)
        score_dict.update(cls_precision)
        score_dict.update(cls_recall)
        return score_dict

    def update_cm(self, pr: np.ndarray, gt: np.ndarray):
        """Get the current confusion matrix, calculate the current F1 score, and update the confusion matrix"""
        gt = np.squeeze(gt, axis=1) if len(gt.shape) == 4 else gt
        for lt, lp in zip(gt, pr):
            area_intersect, area_union, area_pred_label, area_label = (
                self.intersect_and_union(
                    pred_label=lp,
                    label=lt,
                    num_classes=self.n_class,
                    ignore_index=self.ignore_index,
                )
            )
            self.update(area_intersect, area_union, area_pred_label, area_label)

    def get_scores(self):
        """get scores from confusion matrix"""
        return self.total_area_to_metrics(
            self.area_intersect,
            self.area_union,
            self.area_pred_label,
            self.area_label,
            n_class=self.n_class,
        )
