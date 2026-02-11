import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def _load_dataset_compat(repo_id, split="test", **kwargs):
    """Load dataset with compatibility for both old and new versions of the datasets library.
    datasets >= 3.0 no longer supports custom loading scripts, so we fall back to
    downloading the repo and loading data files directly."""
    import os
    try:
        return load_dataset(repo_id, split=split, **kwargs)
    except (RuntimeError, TypeError, ValueError):
        pass

    # Fallback: download the repo and find data files
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id, repo_type="dataset")

    # Search for data files in common formats
    data_extensions = {
        "json": "json", "jsonl": "json",
        "parquet": "parquet",
        "csv": "csv",
        "arrow": "arrow",
    }
    # Look for data files in root and data/ subdirectory
    search_dirs = [local_dir, os.path.join(local_dir, "data")]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for fname in sorted(os.listdir(search_dir)):
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in data_extensions:
                fpath = os.path.join(search_dir, fname)
                fmt = data_extensions[ext]
                try:
                    return load_dataset(fmt, data_files=fpath, split=split)
                except Exception:
                    continue

    raise RuntimeError(
        f"Cannot load dataset '{repo_id}' with datasets >= 3.0.\n"
        f"This dataset only has a loading script and no standard data files.\n"
        f"Please downgrade the datasets library:\n"
        f"    pip install 'datasets<3.0'\n"
        f"For example: pip install datasets==2.21.0"
    )


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    try:
        # datasets < 3.0: supports trust_remote_code and version_tag
        dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
    except (RuntimeError, TypeError, ValueError):
        # datasets >= 3.0: loading scripts no longer supported, load parquet directly
        dataset = _load_dataset_compat("livecodebench/code_generation_lite", split="test")
        # Filter by version_tag manually since custom script params are unavailable
        version_matched = [p for p in dataset if p.get("version_tag", None) == release_version or release_version is None]
        dataset = version_matched if version_matched else list(dataset)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
