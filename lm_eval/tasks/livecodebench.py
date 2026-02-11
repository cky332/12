"""LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code
https://livecodebench.github.io/

LiveCodeBench is a benchmark for evaluating LLMs on code generation, collected from competitive
programming platforms (LeetCode, Codeforces, AtCoder) after model training cutoffs. Problems
include both functional (call-based) and stdin/stdout formats.

Dataset: https://huggingface.co/livecodebench
"""

import json
import os
import sys
import re

from lm_eval.base import Task

# Add LiveCodeBench to path so we can import its utilities
_LCB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LiveCodeBench")
if _LCB_DIR not in sys.path:
    sys.path.insert(0, _LCB_DIR)

from lcb_runner.benchmarks.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


class LiveCodeBench(Task):
    """LiveCodeBench code generation task.

    Uses the livecodebench/code_generation_lite dataset from HuggingFace.
    Problems are competitive programming tasks with both functional (LeetCode-style)
    and stdin/stdout formats.
    """

    DATASET_PATH = None  # We load via LiveCodeBench's own loader
    DATASET_NAME = None

    def __init__(self, release_version="release_v1", start_date=None, end_date=None):
        self.release_version = release_version
        self.start_date = start_date
        self.end_date = end_date
        self.stop_words_list = ["\n```", "\nclass Solution", "\n# Example", "\n# Test", "\nif __name__"]
        # Do NOT call super().__init__() first since it tries load_dataset with DATASET_PATH=None.
        # Instead replicate needed setup manually.
        self.stop_words = self.stop_words_list
        self.requires_execution = True
        # Load dataset via LiveCodeBench's loader
        self._dataset = load_code_generation_dataset(
            release_version=self.release_version,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def get_dataset(self):
        """Returns the list of CodeGenerationProblem objects."""
        return self._dataset

    def get_prompt(self, doc):
        """Builds a prompt suitable for base code generation models (e.g., StarCoder).

        For functional problems (with starter_code): provides the problem description
        as a docstring/comment, then the starter code for the model to complete.

        For stdin/stdout problems: provides the problem description as a comment
        block, then a code header for the model to continue.
        """
        question_content = doc.question_content

        if doc.starter_code:
            # Functional / call-based problem (typically LeetCode)
            prompt = f'"""\n{question_content}\n"""\n{doc.starter_code}\n'
        else:
            # Stdin/stdout problem (typically Codeforces/AtCoder)
            prompt = f'"""\n{question_content}\n"""\nimport sys\ninput = sys.stdin.readline\n\n'

        return prompt

    def get_reference(self, doc):
        """Returns the evaluation sample dict (input_output JSON) for the problem.

        This format is what LiveCodeBench's testing_util.run_test expects:
        {"input_output": json.dumps({"inputs": [...], "outputs": [...], "fn_name": ...})}
        """
        return doc.get_evaluation_sample()

    def get_solutions(self, doc):
        """LiveCodeBench does not provide canonical solutions in the dataset.
        Return None to indicate no solution is available.
        """
        return None

    def get_full_data(self, doc):
        """For human code detection. LiveCodeBench does not ship canonical solutions,
        so we cannot provide human code for watermark false-positive testing.
        Returns None to skip this problem in human code detection.
        """
        return None

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token."""
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    @staticmethod
    def _extract_code_from_fenced(text):
        """Extract code from markdown code fences if present."""
        # Look for ```python ... ``` blocks
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return text

    def postprocess_generation(self, generation, idx):
        """Post-process a generation: strip prompt prefix and stop at stop tokens.

        The generation includes the prompt as prefix. We strip it, then stop at
        the first stop token, then re-prepend the prompt so the full code is
        available for watermark detection (which needs the prompt prefix for
        tokenization alignment).
        """
        doc = self._dataset[idx]
        prompt = self.get_prompt(doc)
        # Strip prompt prefix from generation
        gen_only = generation[len(prompt):]
        gen_only = self._stop_at_stop_token(gen_only, self.stop_words)
        return prompt + gen_only

    def process_results(self, generations, references):
        """Evaluate generated code using LiveCodeBench's evaluation framework.

        :param generations: list(list(str)) - list of lists containing code generations
        :param references: list(dict) - list of dicts with 'input_output' key
        :return: (dict, list) - metrics dict and pass info
        """
        n_problems = len(generations)
        n_samples = len(generations[0]) if generations else 0

        # Strip prompts from generations to get just the code
        stripped_generations = []
        for idx, gens in enumerate(generations):
            doc = self._dataset[idx]
            prompt = self.get_prompt(doc)
            stripped = []
            for gen in gens:
                code = gen[len(prompt):] if gen.startswith(prompt) else gen
                stripped.append(code)
            stripped_generations.append(stripped)

        # Build samples_list in the format expected by codegen_metrics
        # Each sample needs {"input_output": json_string}
        samples_list = references  # already in correct format from get_reference()

        # k_list for pass@k metrics
        k_list = [k for k in [1, 5, 10, 20, 40] if k <= n_samples]
        if not k_list:
            k_list = [1]

        metrics_result, results, metadata = codegen_metrics(
            samples_list=samples_list,
            generations_list=stripped_generations,
            k_list=k_list,
            num_process_evaluate=16,
            timeout=6,
            debug=False,
        )

        # Build pass_info compatible with the framework's expected format
        # pass_info should be a list of per-problem results
        pass_info = []
        for idx in range(n_problems):
            if idx in results:
                problem_results = results[idx]
                # Each entry is a list of test results per generation
                passed = [all(r > 0 for r in gen_result) if gen_result else False
                          for gen_result in problem_results]
                pass_info.append(passed)
            else:
                pass_info.append([False] * n_samples)

        return metrics_result, pass_info

    def strip_prompt(self, code_gens):
        """Strips the prompt from the code generations."""
        stripped_code_gens = [[] for _ in code_gens]
        for idx, code_gen in enumerate(code_gens):
            doc = self._dataset[idx]
            prompt = self.get_prompt(doc)
            for sample in code_gen:
                stripped_code_gens[idx].append(sample[len(prompt):])
        return stripped_code_gens
