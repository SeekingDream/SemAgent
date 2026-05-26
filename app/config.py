"""
Values of global configuration variables.
"""

# Overall output directory for results
output_dir: str = ""

# Max number of times context retrieval and all is tried
overall_retry_limit: int = 3

# upper bound of the number of conversation rounds for the agent
conv_round_limit: int = 15

# whether to perform layered search
enable_layered: bool = True

# whether to do angelic debugging
enable_angelic: bool = False

# whether to do perfect angelic debugging
enable_perfect_angelic: bool = False

# A special mode to only save SBFL result and exit
only_save_sbfl_result: bool = False

# A special mode to only generate reproducer tests and exit
only_reproduce: bool = False

# A special mode to only evaluate a reproducer test
only_eval_reproducer: bool = False

# timeout for test cmd execution, currently set to 5 min
test_exec_timeout: int = 300

# A special mode to only collect fix locations and exit
disable_patch_generation: bool = False

# Used with disable_patch_generation - constrains or extends the amount of context retrieval rounds
context_generation_limit: int = -1

oracle_format_2: bool = False

# Use original ACR patch format
patch_generation_mode: str = "acr"

# Use Debug Config
debug_config: bool = False

models: list[str] = []

backup_model = ["gpt-4o-2024-05-13"]

disable_angelic: bool = False

### Configs for call chain fixer and reproducer second loop

# use call chain fixer:
use_call_chain_fix: bool = True

# enable the reviewer:
use_reviewer: bool = True

# use cached results
use_cached_results: bool = True

# cached results path
cached_results_path: str = "" #<path to empty json file>

# use extra reproducer loop 2:
use_reproducer_output_for_localization: bool = True

### Adding Unit Test Patches

use_unit_test_patch_diffs: bool = False

path_to_unit_test_patch_diffs: str = ""

### other imp configs

# whether to perform sbfl
enable_sbfl: bool = True # Just for help in localization

# Experimental mode to add reproducer and reviewer into the workflow
reproduce_and_review: bool = True

# whether to perform our own validation
enable_validation: bool = True

### Patch selection from multiple runs mode

patch_selection_from_multiple_runs: bool = False

list_of_run_patches_descriptions_and_paths: list[dict[str, str]] = []

### VertexAI configs

vertexai_creds_path: str = ""
vertex_project: str = ""

### Give an initial patch

is_initial_patch_given: bool = True
path_to_patches: str = "" # Path to patches in json format



