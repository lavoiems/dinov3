# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import torch
from torch import nn


class CDLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        k=5,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.k = k
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def sample_synthetic(head):
        raise NotImplementedError("TODO")

    def init_weights(self):
        pass

    def energy(data_logits, synthetic_logits):
        raise NotImplementedError("TODO")

    def forward(self, data_logits, head, ignore_diagonal=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops, batch, prototypes]
        teacher_probs:  [teacher crops, batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        synthetic_sample = self.sample_synthetic(head)
        synthetic_logits = head(synthetic_sample)
        loss = self.energy(data_logits, synthetic_logits)
        return loss
