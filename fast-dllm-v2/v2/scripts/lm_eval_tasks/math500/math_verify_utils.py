"""MATH-500 `math500_dd` task: `math_verify` filter + grader (discrete-diffusion style).

`utils.py` still owns **prompts** and **process_docs** (boxed instruction + normalized
targets). This module only implements the `lm_eval` custom filter and `process_results`
that run ``parse`` / ``verify`` on generations and the gold `doc[\"answer\"]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from math_verify import parse, verify
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": "$" + str(doc["answer"]) + "$",
        }
        return out_doc

    return dataset.map(_process_doc)


def extract_answer(
    resps: List[List[str]], docs: List[Dict]
) -> List[List[Any]]:
    res = []
    for resp_group in resps:
        group_res = []
        for resp in resp_group:
            group_res.append(parse(resp))
        res.append(group_res)
    return res


def process_results(
    doc: dict, results: List[List[Any]]
) -> Dict[str, Union[float, List[float]]]:
    parsed_res = results[0]
    ans = parse(doc["answer"])
    correct = verify(ans, parsed_res[0])
    return {"exact_match": float(correct)}
