# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict

from transformers import EvalPrediction

from swift.utils import Serializer, get_logger
from .base import EvalMetrics
from .utils import MeanMetric

logger = get_logger()


class TableMetrics(EvalMetrics):
    """Evaluation metrics for table structure recognition using TEDS and GriTS.

    This wraps the existing TEDS/GriTS ORM implementations for use during
    SFT validation. Computes:
    - teds: Full TEDS score (content + structure)
    - teds_structure: Structure-only TEDS
    - grits_con: GriTS content similarity
    - grits_top: GriTS topology similarity
    - grits_loc: GriTS location similarity

    All metrics are percentages (0-100).
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        from swift.rewards.orm import TEDS, GriTS

        self.teds = TEDS(structure_only=False)
        self.teds_structure = TEDS(structure_only=True)
        self.grits_con = GriTS(metric_type='con')
        self.grits_top = GriTS(metric_type='top')
        self.grits_loc = GriTS(metric_type='loc')

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        """Compute TEDS and GriTS metrics for table predictions.

        Args:
            eval_prediction: Contains predictions and label_ids as tensors

        Returns:
            Dictionary of metric scores (percentages 0-100)
        """
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids

        # Decode tensors to strings (same as NlgMetrics)
        pred_strs = []
        label_strs = []
        for i in range(preds.shape[0]):
            pred_strs.append(Serializer.from_tensor(preds[i]))
            label_strs.append(Serializer.from_tensor(labels[i]))

        # Initialize metric accumulators
        score_dict = {
            'teds': MeanMetric(),
            'teds_structure': MeanMetric(),
            'grits_con': MeanMetric(),
            'grits_top': MeanMetric(),
            'grits_loc': MeanMetric(),
        }

        # Compute metrics for each sample
        # ORM classes expect completions and html_table as arguments
        teds_scores = self.teds(completions=pred_strs, html_table=label_strs)
        teds_structure_scores = self.teds_structure(completions=pred_strs, html_table=label_strs)
        grits_con_scores = self.grits_con(completions=pred_strs, html_table=label_strs)
        grits_top_scores = self.grits_top(completions=pred_strs, html_table=label_strs)
        grits_loc_scores = self.grits_loc(completions=pred_strs, html_table=label_strs)

        # Update accumulators
        for i in range(len(pred_strs)):
            score_dict['teds'].update(teds_scores[i])
            score_dict['teds_structure'].update(teds_structure_scores[i])
            score_dict['grits_con'].update(grits_con_scores[i])
            score_dict['grits_top'].update(grits_top_scores[i])
            score_dict['grits_loc'].update(grits_loc_scores[i])

        # Compute averages and convert to percentages (0-100)
        return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}
