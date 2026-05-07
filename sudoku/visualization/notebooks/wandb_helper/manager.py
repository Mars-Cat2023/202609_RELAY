from typing import Any, Callable, Dict, List, Tuple, Union
from copy import deepcopy
import os
import re
import pandas as pd
import json
from typing import Optional
import wandb
import hydra

Number = Union[int, float]
Value = Union[int, float, bool, str, None]

DEFAULT_QUERY = {
    "$and": [
        {"state": {"$in": ["finished"]}},
        {"tags": "exp3"},
        {"tags": {"$nin": ["discard"]}},
    ]
}



class RunsManager:
    def __init__(
        self,
        name: str,
        wandb_entity: str,
        wandb_project: str,
        fields_of_interest: List[str],
        summary_fields_of_interest: List[str],
        run_filter: Optional[Dict[str, Any]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        tags_delimiter: List[str] = ["=", "__"],
        value_processor: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize the runs manager.

        Args:
            name: Name of the runs manager. Will be used in the local cache file names.
            wandb_entity: Wandb entity.
            wandb_project: Wandb project.
            fields_of_interest: List of regex patterns to match against the keys in the config of the runs.
                The whole config can be big, and we might only be interested in a few key hyperparameters. This list is used
                to filter the config to only keep the keys that match the patterns.
            summary_fields_of_interest: List of regex patterns to match against the keys in the summary of the runs.
                Same as fields_of_interest_patterns, but for the summary fields.
            run_filter: MongoDB query to use to  query the runs. Eg:
                {
                    "$and": [
                        {"state": {"$in": ["finished"]}},
                        {"tags": "exp3"},
                        {"tags": {"$nin": ["discard"]}},
                    ]
                }
            rename_map: Map of keys to rename. Before plotting you may want to rename fields, e.g. config.predictor.max_steps to sampling_steps, etc. Use this dictionary for this mapping.
            tags_delimiter: Delimiter to split the tags.
            value_processor: Function to process the values.
        """
        self.name = name
        self.run_filter = run_filter if run_filter is not None else DEFAULT_QUERY
        self.wandb_entity = wandb_entity or os.getenv("WANDB_ENTITY")
        self.wandb_project = wandb_project or os.getenv("WANDB_PROJECT")
        self.fields_of_interest_patterns = fields_of_interest
        self.summary_fields_of_interest = summary_fields_of_interest
        self.runs: Optional[List[Dict[str, Any]]] = None
        self._df: Optional[pd.DataFrame] = None
        self.rename_map = rename_map 
        self.tags_delimiter = tags_delimiter
        self.value_processor = value_processor if value_processor is not None else (lambda x: x)
        self.wandb_api = wandb.Api()
    
    @classmethod
    def from_yaml(cls, config: str) -> "RunsManager":
        """Create manager from yaml config."""
        with open(config, "r") as f:
            pass

    def get_runs(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get matching runs from wandb.

        Args:
            force_refresh: If True, fetch the runs from wandb again even if there is a local cache runs.
        Returns:
            List of raw runs.
        """
        if force_refresh or not os.path.exists(f"{self.name}_raw_runs.json"):
            raw_runs = []
            print(
                f"Getting runs from {self.wandb_entity}/{self.wandb_project}"
            )
            for run in self.wandb_api.runs(
                f"{self.wandb_entity}/{self.wandb_project}",
                filters=self.run_filter,
            ):
                raw_run = {
                    "run_id": run.id,
                    "config": flatten_dict(dict(run.config)),
                    "summary": dict(run.summary),
                    "tags": run.tags,
                }
                raw_runs.append(raw_run)
            # save raw
            with open(f"{self.name}_raw_runs.json", "w") as f:
                json.dump(raw_runs, f, indent=4, default=str)
        raw_runs = json.load(open(f"{self.name}_raw_runs.json"))
        self.runs = raw_runs
        return raw_runs
    
    def refresh(self, force_refresh_from_wandb: bool = False) -> pd.DataFrame:
        """Refresh in memory runs from wandb or local cache."""
        return self.df(force_refresh_from_wandb=force_refresh_from_wandb, force_refresh_from_local=True)

    def df(
        self,
        force_refresh_from_wandb: bool = False,
        force_refresh_from_local: bool = False,
    ):
        """Prepare a pandas dataframe with the runs. Download the runs from wandb if needed, and cache the dataframe locally.

        Steps:
            1. Return cached in-memory dataframe if present and no refresh requested.
            2. Load from local ``{name}_runs.json`` if it exists and no refresh requested.
            3. Fetch runs from wandb when needed (runs is None or force_refresh_from_wandb).
            4. For each run: filter config and summary by fields-of-interest patterns; parse tags
               (key=value) into tag columns.
            5. Build dataframe, rename columns per rename_map, keep only selected columns.
            6. Apply value_processor, then cache in memory and save to ``{name}_runs.json``.
        
        Note: 
            Before renaming is done, the column names will contain `config.`, `summary.` and `tags.` prefixes. So the keys of 
            rename_map should not contain these prefixes.

        Args:
            force_refresh_from_wandb: If True, force refresh the runs from wandb even if there is a local cache runs.
            force_refresh_from_local: If True, refresh the in memory runs from the local cache.
        Returns:
            Pandas dataframe with the runs.

        """
        # check for local copy
        if (
            not force_refresh_from_local
            and not force_refresh_from_wandb
            and self._df is not None
        ):
            return self._df
        if (
            not force_refresh_from_local
            and not force_refresh_from_wandb
            and os.path.exists(f"{self.name}_runs.json")
        ):
            print("Loading local copy of df from runs.json")
            self._df = pd.read_json(f"{self.name}_runs.json", orient="records")
            return self._df
        print("Attempting to load runs from local raw runs")

        cols_present = set()
        permanent_cols = ["run_id"]

        def _renamer(key: str) -> str:
            has_explicit_rename = key in self.rename_map
            if not has_explicit_rename:
                if key.startswith("tags."):  # keep all tags
                    cols_present.add(key)
                    return key
            renamed_key = self.rename_map.get(key, key)
            if key in cols_present:
                raise ValueError(f"Column {key} already exists")
            if has_explicit_rename:
                cols_present.add(renamed_key)
            return renamed_key

        if self.runs is None or force_refresh_from_wandb:
            self.get_runs(force_refresh=force_refresh_from_wandb)

        # process the runs
        processed_runs = []
        for run in self.runs:
            # filter to keep only the fields of interest in config and flatten the names
            run.update(
                {
                    f"config.{k}": v
                    for k, v in filter_dict_by_patterns(
                        run["config"], self.fields_of_interest_patterns
                    ).items()
                }
            )
            # filter to keep only the fields of interest in summary and flatten the names
            run.update(
                {
                    f"summary.{k}": v
                    for k, v in filter_dict_by_patterns(
                        run["summary"], self.summary_fields_of_interest
                    ).items()
                }
            )
            # tags
            _tags_tuples = [split_tag(t, self.tags_delimiter) for t in run["tags"]]
            run.update({f"tags.{k}": v for k, v in _tags_tuples})
            processed_runs.append(run)
        _df = pd.DataFrame(processed_runs)
        _df = _df.rename(columns=_renamer)
        # keep only the columns that are present in the rename_map
        _df = _df[permanent_cols + list(cols_present)]
        _df = self.value_processor(_df)
        self._df = _df
        self._df.to_json(f"{self.name}_runs.json", orient="records")
        return self._df


def split_tag(tag: str, delimiter: Optional[List[str]] = ["=", "__"]) -> Tuple[str, Optional[str]]:
    if delimiter is None:
        return tag, None
    for d in delimiter:
        split_tag = tag.split(d)
        if len(split_tag) == 2:
            break
    if len(split_tag) == 1:
        return split_tag[0], None
    elif len(split_tag) == 2:
        return split_tag[0], split_tag[1]
    else:
        raise ValueError(f"Invalid tag {tag} with delimiter {delimiter}")


# flatten the configs
def flatten_dict(
    params: Dict[str, Any], delimiter: str = "."
) -> Dict[str, Value]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a.b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'.'``.
    Returns:
        Flattened dict.
    """
    output: Dict[str, Union[str, Number]] = {}

    def populate(
        inp: Union[Dict[str, Any], List, str, Number, bool], prefix: List[str]
    ) -> None:

        if isinstance(inp, dict):
            for k, v in inp.items():
                populate(v, deepcopy(prefix) + [k])

        elif isinstance(inp, list):
            for i, val in enumerate(inp):
                populate(val, deepcopy(prefix) + [str(i)])
        elif isinstance(inp, (str, float, int, bool)) or (inp is None):
            output[delimiter.join(prefix)] = inp
        else:  # unsupported type
            raise ValueError(
                f"Unsuported type {type(inp)} at {delimiter.join(prefix)} for flattening."
            )

    populate(params, [])

    return output


def filter_dict_by_patterns(
    data: Dict[str, Any], patterns: List[str]
) -> Dict[str, Any]:
    """
    Filter a dictionary by a list of regex patterns.
    Args:
        data: Dictionary to filter.
        patterns: List of regex patterns to match against the keys.
    Returns:
        Filtered dictionary.
    """
    filtered_data = {}
    for key, value in data.items():
        if any(re.search(pattern, key) for pattern in patterns):
            filtered_data[key] = value
    return filtered_data