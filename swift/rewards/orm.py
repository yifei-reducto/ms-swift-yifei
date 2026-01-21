# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome Reward Model (ORM) implementations for GRPO training.

import os
import re
from typing import Dict, List, Union

import json

from swift.infer_engine import InferRequest


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(max(min(-exceed_len / self.soft_cache_length, 0), -0.5))
        return rewards


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class TEDS(ORM):
    """Tree Edit Distance-based Similarity (TEDS) reward for table structure recognition.

    TEDS measures the similarity between predicted and ground truth HTML tables
    by computing tree edit distance on their DOM tree representations.

    Reference: https://arxiv.org/abs/1911.10683
    """

    def __init__(self, structure_only: bool = False):
        """Initialize TEDS reward.

        Args:
            structure_only: If True, only compare table structure without text content.
        """
        self.structure_only = structure_only
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                'The lxml package is required but not installed. '
                "Please install it using 'pip install lxml'.") from e

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text for comparison."""
        if not text:
            return []
        return text.strip().split()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by stripping whitespace."""
        if text is None:
            return ''
        return ' '.join(text.split())

    def tree_to_list(self, root, structure_only: bool = False) -> List[str]:
        """Convert HTML tree to a list representation for edit distance."""
        from lxml import etree

        def traverse(node, depth=0):
            result = []
            if isinstance(node.tag, str):
                tag = node.tag.lower()
                if tag in ('table', 'thead', 'tbody', 'tr', 'td', 'th'):
                    # Include colspan and rowspan attributes
                    attrs = []
                    colspan = node.get('colspan', '1')
                    rowspan = node.get('rowspan', '1')
                    if colspan != '1':
                        attrs.append(f'colspan={colspan}')
                    if rowspan != '1':
                        attrs.append(f'rowspan={rowspan}')
                    tag_str = f'<{tag} {" ".join(attrs)}>' if attrs else f'<{tag}>'
                    result.append(tag_str)

                    if not structure_only and tag in ('td', 'th'):
                        text = self.normalize_text(node.text or '')
                        tail_text = ''
                        for child in node:
                            if child.tail:
                                tail_text += ' ' + self.normalize_text(child.tail)
                            text += ' ' + self.normalize_text(
                                etree.tostring(child, method='text', encoding='unicode') or '')
                        text = self.normalize_text(text + tail_text)
                        if text:
                            result.append(f'[TEXT:{text}]')

                    for child in node:
                        result.extend(traverse(child, depth + 1))
                    result.append(f'</{tag}>')
            return result

        return traverse(root)

    def parse_html_table(self, html_str: str):
        """Parse HTML string and extract table element."""
        from lxml import etree

        try:
            html_str = html_str.strip()
            if not html_str.startswith('<'):
                return None

            # Try parsing as HTML fragment
            try:
                parser = etree.HTMLParser()
                tree = etree.fromstring(f'<html><body>{html_str}</body></html>', parser)
                tables = tree.xpath('//table')
                if tables:
                    return tables[0]
            except Exception:
                pass

            # Try parsing as XML
            try:
                root = etree.fromstring(html_str.encode('utf-8'))
                if root.tag.lower() == 'table':
                    return root
                tables = root.xpath('//table')
                if tables:
                    return tables[0]
            except Exception:
                pass

            return None
        except Exception:
            return None

    def compute_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute Levenshtein edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def compute_teds(self, pred_html: str, gt_html: str) -> float:
        """Compute TEDS score between predicted and ground truth HTML tables."""
        pred_tree = self.parse_html_table(pred_html)
        gt_tree = self.parse_html_table(gt_html)

        if pred_tree is None and gt_tree is None:
            return 1.0
        if pred_tree is None or gt_tree is None:
            return 0.0

        pred_list = self.tree_to_list(pred_tree, self.structure_only)
        gt_list = self.tree_to_list(gt_tree, self.structure_only)

        if not pred_list and not gt_list:
            return 1.0
        if not pred_list or not gt_list:
            return 0.0

        edit_dist = self.compute_edit_distance(pred_list, gt_list)
        max_len = max(len(pred_list), len(gt_list))
        teds = 1.0 - edit_dist / max_len

        return max(0.0, teds)

    def extract_html_from_completion(self, completion: str) -> str:
        """Extract HTML table from model completion.

        Handles thinking models by removing <think>...</think> content first,
        so only the final output is used for reward calculation.
        """
        # Remove thinking tokens (for models like Qwen3-VL-Thinking)
        # First try to remove complete <think>...</think> blocks
        completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL | re.IGNORECASE)
        # Handle unclosed thinking tags - take content after </think>
        if '</think>' in completion.lower():
            parts = re.split(r'</think>', completion, flags=re.IGNORECASE)
            completion = parts[-1] if len(parts) > 1 else completion
        # Handle case where only <think> exists (content before it or discard thinking content)
        if '<think>' in completion.lower():
            parts = re.split(r'<think>', completion, flags=re.IGNORECASE)
            completion = parts[0] if parts[0].strip() else (parts[-1] if len(parts) > 1 else completion)

        # Look for table within code blocks
        code_block_match = re.search(r'```(?:html)?\s*(.*?)```', completion, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            completion = code_block_match.group(1)

        # Look for <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if answer_match:
            completion = answer_match.group(1)

        # Find table element
        table_match = re.search(r'(<table.*?</table>)', completion, re.DOTALL | re.IGNORECASE)
        if table_match:
            return table_match.group(1)

        return completion.strip()

    def __call__(self, completions, html_table, **kwargs) -> List[float]:
        """Compute TEDS rewards for completions.

        Args:
            completions: List of model completions (predicted HTML tables)
            html_table: List of ground truth HTML tables from the dataset
        """
        rewards = []
        for completion, gt in zip(completions, html_table):
            pred_html = self.extract_html_from_completion(completion)
            teds_score = self.compute_teds(pred_html, gt)
            rewards.append(teds_score)
        return rewards


class TEDSStructure(TEDS):
    """TEDS reward that only compares table structure without text content."""

    def __init__(self):
        super().__init__(structure_only=True)


class GriTS(ORM):
    """Grid Table Similarity (GriTS) reward for table structure recognition.

    GriTS measures table similarity by comparing cell contents at grid positions,
    accounting for cell spanning (colspan/rowspan).

    Reference: https://arxiv.org/abs/2203.12555
    """

    def __init__(self, metric_type: str = 'con'):
        """Initialize GriTS reward.

        Args:
            metric_type: Type of GriTS metric - 'con' (content), 'top' (topology),
                        or 'loc' (location). Default is 'con'.
        """
        self.metric_type = metric_type
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                'The lxml package is required but not installed. '
                "Please install it using 'pip install lxml'.") from e

    def parse_html_to_grid(self, html_str: str) -> List[List[str]]:
        """Parse HTML table to a 2D grid representation."""
        from lxml import etree

        try:
            html_str = html_str.strip()
            if not html_str.startswith('<'):
                return []

            # Parse HTML
            try:
                parser = etree.HTMLParser()
                tree = etree.fromstring(f'<html><body>{html_str}</body></html>', parser)
                tables = tree.xpath('//table')
                if not tables:
                    return []
                table = tables[0]
            except Exception:
                try:
                    table = etree.fromstring(html_str.encode('utf-8'))
                    if table.tag.lower() != 'table':
                        tables = table.xpath('//table')
                        if not tables:
                            return []
                        table = tables[0]
                except Exception:
                    return []

            # Extract rows
            rows = table.xpath('.//tr')
            if not rows:
                return []

            # Determine grid dimensions
            num_rows = len(rows)
            num_cols = 0
            for row in rows:
                cols = 0
                for cell in row.xpath('.//td|.//th'):
                    colspan = int(cell.get('colspan', 1))
                    cols += colspan
                num_cols = max(num_cols, cols)

            if num_cols == 0:
                return []

            # Initialize grid
            grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
            occupied = [[False for _ in range(num_cols)] for _ in range(num_rows)]

            # Fill grid
            for row_idx, row in enumerate(rows):
                col_idx = 0
                for cell in row.xpath('.//td|.//th'):
                    # Skip occupied cells
                    while col_idx < num_cols and occupied[row_idx][col_idx]:
                        col_idx += 1
                    if col_idx >= num_cols:
                        break

                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))

                    # Get cell text
                    text = etree.tostring(cell, method='text', encoding='unicode') or ''
                    text = ' '.join(text.split())

                    # Fill grid cells
                    for r in range(rowspan):
                        for c in range(colspan):
                            if row_idx + r < num_rows and col_idx + c < num_cols:
                                if self.metric_type == 'top':
                                    grid[row_idx + r][col_idx + c] = f'cell_{row_idx}_{col_idx}'
                                elif self.metric_type == 'loc':
                                    grid[row_idx + r][col_idx + c] = f'{row_idx}_{col_idx}'
                                else:
                                    grid[row_idx + r][col_idx + c] = text
                                occupied[row_idx + r][col_idx + c] = True

                    col_idx += colspan

            return grid
        except Exception:
            return []

    def compute_cell_similarity(self, pred_cell: str, gt_cell: str) -> float:
        """Compute similarity between two cells."""
        if self.metric_type in ('top', 'loc'):
            return 1.0 if pred_cell == gt_cell else 0.0
        else:
            # Content comparison using token overlap
            pred_tokens = set(pred_cell.lower().split())
            gt_tokens = set(gt_cell.lower().split())

            if not pred_tokens and not gt_tokens:
                return 1.0
            if not pred_tokens or not gt_tokens:
                return 0.0

            intersection = len(pred_tokens & gt_tokens)
            precision = intersection / len(pred_tokens) if pred_tokens else 0
            recall = intersection / len(gt_tokens) if gt_tokens else 0

            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

    def compute_grits(self, pred_html: str, gt_html: str) -> float:
        """Compute GriTS score between predicted and ground truth HTML tables."""
        pred_grid = self.parse_html_to_grid(pred_html)
        gt_grid = self.parse_html_to_grid(gt_html)

        if not pred_grid and not gt_grid:
            return 1.0
        if not pred_grid or not gt_grid:
            return 0.0

        # Align grids to same dimensions
        pred_rows, pred_cols = len(pred_grid), len(pred_grid[0]) if pred_grid else 0
        gt_rows, gt_cols = len(gt_grid), len(gt_grid[0]) if gt_grid else 0

        max_rows = max(pred_rows, gt_rows)
        max_cols = max(pred_cols, gt_cols)

        # Pad grids
        for grid, rows, cols in [(pred_grid, pred_rows, pred_cols), (gt_grid, gt_rows, gt_cols)]:
            for row in grid:
                row.extend([''] * (max_cols - len(row)))
            for _ in range(max_rows - len(grid)):
                grid.append([''] * max_cols)

        # Compute cell-wise similarity
        total_sim = 0.0
        total_cells = max_rows * max_cols

        for i in range(max_rows):
            for j in range(max_cols):
                pred_cell = pred_grid[i][j] if i < len(pred_grid) and j < len(pred_grid[i]) else ''
                gt_cell = gt_grid[i][j] if i < len(gt_grid) and j < len(gt_grid[i]) else ''
                total_sim += self.compute_cell_similarity(pred_cell, gt_cell)

        return total_sim / total_cells if total_cells > 0 else 0.0

    def extract_html_from_completion(self, completion: str) -> str:
        """Extract HTML table from model completion.

        Handles thinking models by removing <think>...</think> content first,
        so only the final output is used for reward calculation.
        """
        # Remove thinking tokens (for models like Qwen3-VL-Thinking)
        # First try to remove complete <think>...</think> blocks
        completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL | re.IGNORECASE)
        # Handle unclosed thinking tags - take content after </think>
        if '</think>' in completion.lower():
            parts = re.split(r'</think>', completion, flags=re.IGNORECASE)
            completion = parts[-1] if len(parts) > 1 else completion
        # Handle case where only <think> exists (content before it or discard thinking content)
        if '<think>' in completion.lower():
            parts = re.split(r'<think>', completion, flags=re.IGNORECASE)
            completion = parts[0] if parts[0].strip() else (parts[-1] if len(parts) > 1 else completion)

        code_block_match = re.search(r'```(?:html)?\s*(.*?)```', completion, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            completion = code_block_match.group(1)

        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if answer_match:
            completion = answer_match.group(1)

        table_match = re.search(r'(<table.*?</table>)', completion, re.DOTALL | re.IGNORECASE)
        if table_match:
            return table_match.group(1)

        return completion.strip()

    def __call__(self, completions, html_table, **kwargs) -> List[float]:
        """Compute GriTS rewards for completions.

        Args:
            completions: List of model completions (predicted HTML tables)
            html_table: List of ground truth HTML tables from the dataset
        """
        rewards = []
        for completion, gt in zip(completions, html_table):
            pred_html = self.extract_html_from_completion(completion)
            grits_score = self.compute_grits(pred_html, gt)
            rewards.append(grits_score)
        return rewards


class GriTSTop(GriTS):
    """GriTS topology metric - compares table structure only."""

    def __init__(self):
        super().__init__(metric_type='top')


class GriTSLoc(GriTS):
    """GriTS location metric - compares cell locations."""

    def __init__(self):
        super().__init__(metric_type='loc')


class TableFormat(ORM):
    """Reward function that checks if the completion contains valid HTML table format."""

    def __call__(self, completions, **kwargs) -> List[float]:
        """Check if completions contain valid HTML table structure.

        Handles thinking models by removing <think>...</think> content first.
        """
        rewards = []
        for completion in completions:
            # Remove thinking tokens (for models like Qwen3-VL-Thinking)
            completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL | re.IGNORECASE)
            if '</think>' in completion.lower():
                parts = re.split(r'</think>', completion, flags=re.IGNORECASE)
                completion = parts[-1] if len(parts) > 1 else completion

            # Check for basic HTML table structure
            has_table = bool(re.search(r'<table.*?>.*?</table>', completion, re.DOTALL | re.IGNORECASE))
            has_rows = bool(re.search(r'<tr.*?>.*?</tr>', completion, re.DOTALL | re.IGNORECASE))
            has_cells = bool(re.search(r'<t[dh].*?>.*?</t[dh]>', completion, re.DOTALL | re.IGNORECASE))

            if has_table and has_rows and has_cells:
                rewards.append(1.0)
            elif has_table and has_rows:
                rewards.append(0.5)
            elif has_table:
                rewards.append(0.25)
            else:
                rewards.append(0.0)
        return rewards


class ThinkingLengthPenalty(ORM):
    """Penalty if thinking content is longer than the actual table output.

    Applies a -0.5 penalty when the <think>...</think> content is longer
    than the actual output (content after thinking).
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            # Extract thinking content length
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL | re.IGNORECASE)
            think_len = len(think_match.group(1)) if think_match else 0

            # Extract output length (content after </think> or whole completion if no thinking)
            if '</think>' in completion.lower():
                output = re.split(r'</think>', completion, flags=re.IGNORECASE)[-1]
            else:
                output = completion
            output_len = len(output.strip())

            # Apply penalty if thinking is longer than output
            if think_len > output_len and output_len > 0:
                rewards.append(-0.5)
            else:
                rewards.append(0.0)
        return rewards


class TableLengthPenalty(ORM):
    """Penalty for output table being longer than groundtruth.

    - No penalty if output is ≤10% longer than groundtruth
    - -0.1 penalty for every 10% beyond the 10% threshold

    Example: 35% longer → -0.1 * (35-10)/10 = -0.25
    """

    def extract_table(self, completion: str) -> str:
        """Extract table content from completion, removing thinking tokens."""
        # Remove thinking tokens
        completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL | re.IGNORECASE)
        if '</think>' in completion.lower():
            parts = re.split(r'</think>', completion, flags=re.IGNORECASE)
            completion = parts[-1] if len(parts) > 1 else completion

        # Extract table element
        table_match = re.search(r'(<table.*?</table>)', completion, re.DOTALL | re.IGNORECASE)
        if table_match:
            return table_match.group(1)
        return completion.strip()

    def __call__(self, completions, html_table, **kwargs) -> List[float]:
        rewards = []
        for completion, gt in zip(completions, html_table):
            # Extract table from completion
            pred_table = self.extract_table(completion)

            pred_len = len(pred_table)
            gt_len = len(gt)

            if gt_len == 0:
                rewards.append(0.0)
                continue

            # Calculate percentage longer
            extra_percentage = ((pred_len - gt_len) / gt_len) * 100

            # Apply penalty: -0.1 for every 10% beyond 10% threshold
            if extra_percentage > 10:
                penalty = -0.1 * (extra_percentage - 10) / 10
                rewards.append(max(penalty, -0.5))
            else:
                rewards.append(0.0)
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'teds': TEDS,
    'teds_structure': TEDSStructure,
    'grits': GriTS,
    'grits_top': GriTSTop,
    'grits_loc': GriTSLoc,
    'table_format': TableFormat,
    'thinking_length_penalty': ThinkingLengthPenalty,
    'table_length_penalty': TableLengthPenalty,
}
