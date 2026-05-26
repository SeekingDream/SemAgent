"""
The main driver.
"""

import json
import logging
import platform
import shutil
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
from itertools import chain
from multiprocessing import set_start_method
import os
from os import PathLike
from os.path import abspath
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import config, inference, log, result_analysis, task_counter
from app import utils as apputils
from app.manage import ProjectApiManager
from app.model import common
from app.model.register import register_all_models
from app.post_process import (
    extract_organize_and_form_input,
    get_final_patch_path,
    organize_and_form_input,
    reextract_organize_and_form_inputs,
)
from app.raw_tasks import RawGithubTask, RawLocalTask, RawSweTask, RawTask
from app.task import SweTask, Task


def main():
    register_all_models()
    parser = ArgumentParser()

    subparser_dest_attr_name = "command"
    subparsers = parser.add_subparsers(dest=subparser_dest_attr_name)

    swe_parser = subparsers.add_parser(
        "swe-bench", help="Run one or multiple swe-bench tasks"
    )
    set_swe_parser_args(swe_parser)

    github_parser = subparsers.add_parser(
        "github-issue",
        help="Run an online github issue",
    )
    set_github_parser_args(github_parser)

    local_parser = subparsers.add_parser("local-issue", help="Run a local issue.")
    set_local_parser_args(local_parser)

    extract_patches_parser = subparsers.add_parser(
        "extract-patches", help="Only extract patches from the raw results dir"
    )
    extract_patches_parser.add_argument("experiment_dir", type=str)
    add_task_related_args(extract_patches_parser)

    re_extract_patches_parser = subparsers.add_parser(
        "re-extract-patches",
        help=(
            "same as extract-patches, except that individual dirs"
            " are moved out of their categories first"
        ),
    )
    re_extract_patches_parser.add_argument("experiment_dir", type=str)
    add_task_related_args(re_extract_patches_parser)

    args = parser.parse_args()

    ## common options
    config.output_dir = args.output_dir
    if config.output_dir is not None:
        config.output_dir = abspath(config.output_dir)
    num_processes: int = int(args.num_processes)
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout

    # model related
    config.models = list(chain.from_iterable(args.model))
    if not config.models:
        config.models.append("gpt-3.5-turbo-0125")
    common.set_model(config.models[0])

    # FIXME: make temperature part of the Model class
    common.MODEL_TEMP = args.model_temperature

    # acr related

    # can be superseded in the the config.py file
    config.conv_round_limit = args.conv_round_limit if config.conv_round_limit is None else config.conv_round_limit
    config.enable_sbfl = args.enable_sbfl if config.enable_sbfl is None else config.enable_sbfl
    config.enable_validation = args.enable_validation if config.enable_validation is None else config.enable_validation
    config.enable_angelic = args.enable_angelic if config.enable_angelic is None else config.enable_angelic
    config.enable_perfect_angelic = args.enable_perfect_angelic if config.enable_perfect_angelic is None else config.enable_perfect_angelic
    config.only_save_sbfl_result = args.save_sbfl_result if config.only_save_sbfl_result is None else config.only_save_sbfl_result
    config.disable_patch_generation = args.output_fix_locs if config.disable_patch_generation is None else config.disable_patch_generation
    config.context_generation_limit = args.output_fix_limit if config.context_generation_limit is None else config.context_generation_limit
    config.oracle_format_2 = args.oracle_format_2 if config.oracle_format_2 is None else config.oracle_format_2
    config.patch_generation_mode = args.patch_generation_mode if config.patch_generation_mode is None else config.patch_generation_mode
    config.only_reproduce = args.reproduce if config.only_reproduce is None else config.only_reproduce

    # needs to be in the cli args
    config.patch_generation_only = args.patch_generation_only
    config.patch_input_dir = args.patch_input_dir
    config.task_input_dir = args.patch_input_dir
    config.validation_only = args.validation_only
    config.validate_gold = args.validate_gold
    config.run_tests = args.run_tests
    config.hypothesis_cand = args.hypothesis_cand
    config.pre_hypothesis = args.pre_hypothesis
    config.debug = args.debug
    if args.debug_config is not None: args = parse_debug_config(args)

    if config.output_dir is not None:
        config.output_dir = abspath(config.output_dir)

    subcommand = getattr(args, subparser_dest_attr_name)
    if subcommand == "swe-bench":

        """
        if args.result_analysis:
            # do analysis and exit
            result_analysis.analyze(config.output_dir)
            exit(0)
        """

        ## If in oracle mode, use the oracle tasks map
        if args.enable_oracle:
            task_map_path = args.tasks_map_oracle
        else:
            task_map_path = args.tasks_map

        tasks = make_swe_tasks(
            args.task, args.task_list_file, args.setup_map, task_map_path, args.oracle_format_2
        )

        config.only_eval_reproducer = args.eval_reproducer if config.only_eval_reproducer is None else config.only_eval_reproducer
        config.reproduce_and_review = args.reproduce_and_review if config.reproduce_and_review is None else config.reproduce_and_review

        """
        # check for validation only mode, otherwise run tasks
        if globals.validate_gold or globals.validation_only:
            # input to validate: list[RawSweTask], ...
            validate(tasks, globals.output_dir, globals.validate_gold)
        elif globals.run_tests:
            validate(tasks, globals.output_dir, init_only=globals.run_tests)
        else:
        """

        groups = group_swe_tasks_by_env(tasks)
        run_task_groups(groups, num_processes, organize_output=True)
    elif subcommand == "github-issue":
        setup_dir = args.setup_dir
        if setup_dir is not None:
            setup_dir = abspath(setup_dir)

        task = RawGithubTask(
            args.task_id,
            args.clone_link,
            args.commit_hash,
            args.issue_link,
            setup_dir,
        )
        groups = {"github": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "local-issue":
        local_repo = args.local_repo
        if local_repo is not None:
            local_repo = abspath(local_repo)
        issue_file = args.issue_file
        if issue_file is not None:
            issue_file = abspath(issue_file)
        task = RawLocalTask(
            args.task_id,
            local_repo,
            issue_file,
        )
        groups = {"local": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "extract-patches":
        extract_organize_and_form_input(args.experiment_dir)
    elif subcommand == "re-extract-patches":
        reextract_organize_and_form_inputs(args.experiment_dir)

def parse_debug_config(args):

    assert os.path.isfile(args.debug_config), f"File {args.debug_config} not found"
    globals.debug_config = args.debug_config

    with open(args.debug_config, "r") as f:
        debug_config = json.load(f)
    
    debug_keys = debug_config.keys()
    if "model" in debug_keys and debug_config["model"] != "": common.set_model(debug_config["model"])
    if "model_temperature" in debug_keys and debug_config["model_temperature"] != "": common.MODEL_TEMP = debug_config["model_temperature"]
    if "output_fix_limit" in debug_keys and debug_config["output_fix_limit"] != "": globals.context_generation_limit = debug_config["output_fix_limit"]
    if "patch_input_dir" in debug_keys and debug_config["patch_input_dir"] != "": globals.task_input_dir = args.patch_input_dir
    config_opts = ["output_dir", "conv_round_limit", "output_fix_locs", "oracle_format_2", "patch_generation_only", "patch_input_dir", "patch_generation_mode", "hypothesis_cand", "pre_hypothesis"]
    for config_opt in config_opts:
        if config_opt in debug_keys and debug_config[config_opt] != "": setattr(globals, config_opt, debug_config[config_opt])
    
    # update args if necessary
    if "enable_oracle" in debug_keys and debug_config["enable_oracle"]:
        args.enable_oracle = True
    if "task" in debug_keys:
        args.task = debug_config["task"]
    if "task_list_file" in debug_keys:
        args.task_list_file = debug_config["task_list_file"]
    if "setup_map" in debug_keys:
        args.setup_map = debug_config["setup_map"]
    if "tasks_map" in debug_keys:
        args.tasks_map = debug_config["tasks_map"]
    if "tasks_map_oracle" in debug_keys:
        args.tasks_map_oracle = debug_config["tasks_map_oracle"]

    return args

def set_swe_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)

    parser.add_argument(
        "--setup-map",
        type=str,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks-map",
        type=str,
        help="Path to json file that contains the tasks information.",
    )
    parser.add_argument(
        "--task-list-file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    
    parser.add_argument("--task", type=str, help="Task id to be run.")

    parser.add_argument(
        "--eval-reproducer",
        action="store_true",
        default=False,
        help="Only check if reproducer.py is a correct test",
    )
    parser.add_argument(
        "--reproduce-and-review",
        action="store_true",
        default=True,
        help="Experimental: for swe-bench tasks, reproduce and review the generated patch",
    )
    parser.add_argument(
        "--result-analysis",
        action="store_true",
        default=False,
        help="Perform some analysis on the experiment result and exit.",
    )

    ##Added for oracle debugging setting:
    parser.add_argument(
        "--tasks-map-oracle",
        type=str,
        help="Path to json file that contains the tasks information for oracle setting.",
    )
    parser.add_argument(
        "--enable-oracle",
        action="store_true",
        default=False,
        help="Enable oracle setting, where the file to modify is given to ACR",
    )
    # Alternative oracle setting
    parser.add_argument(
        "--oracle-format-2",
        action="store_true",
        default=False,
        help="Use oracle setting, but only give ACR the oracle file name instead of file content",
    )
    # Flag for running patch generation only from proposed locations
    parser.add_argument(
        "--patch-generation-only",
        action="store_true",
        default=False,
        help="Do not perform context retrieval, only patch generation from previously proposed fix locations"
    )
    parser.add_argument(
        "--patch-input-dir",
        type=str,
        help="For patch generation only mode, provide path to directory with task fix locations"
    )
    # Add options for different patch generation strategies/modes
    parser.add_argument(
        "--patch-generation-mode",
        type=str,
        default="acr",
        choices=["acr", "cot", "hypothesis", "hypothesis-reflect", "hypothesis-diversify", "hypothesis-manual"],
        help="Modes/strategies of patch generation, like using `cot` for edit-level reasoning during patch generation or `acr` for original format"
    )
    parser.add_argument(
        "--pre-hypothesis",
        action="store_true",
        default=False,
        help="Get initial hypothesis before code search"
    )
    # Add option for DEBUG mode
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Use debug mode: replace LLM query with hardcoded value"
    )
    parser.add_argument(
        "--debug-config",
        type=str,
        help="Path to debug config json file with modes and hardcoded debug values"
    )
    # Add options for patch validation only
    parser.add_argument(
        "--validation-only",
        action="store_true",
        default=False,
        help="Perform only patch validation for previously generated patches"
    )
    parser.add_argument(
        "--validate-gold",
        action="store_true",
        default=False,
        help="Run testsuite validation on developer gold patches"
    )
    parser.add_argument(
        "--hypothesis-cand",
        type=int,
        default=5,
        help="Number of hypothesis candidates to generate"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        default=False,
        help="Run testsuite validation without applying patch"
    )



def set_github_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id",
        type=str,
        help="Assign an id to the current fresh issue task.",
    )
    parser.add_argument(
        "--clone-link",
        type=str,
        help="The link to the repository to clone.",
    )
    parser.add_argument("--commit-hash", type=str, help="The commit hash to checkout.")
    parser.add_argument("--issue-link", type=str, help="The link to the issue.")
    parser.add_argument(
        "--setup-dir",
        type=str,
        help="The directory where repositories should be cloned to.",
    )


def set_local_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current local issue task."
    )
    parser.add_argument(
        "--local-repo", type=str, help="Path to a local copy of the target repo."
    )
    parser.add_argument("--issue-file", type=str, help="Path to a local issue file.")


def add_task_related_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory that stores the run results.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Do not print most messages to stdout.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(common.MODEL_HUB.keys()),
        nargs="+",
        action="append",
        help="The model to use. Currently only OpenAI models are supported.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.0,
        help="The model temperature to use, for OpenAI models.",
    )
    parser.add_argument(
        "--conv-round-limit",
        type=int,
        default=15,
        help="Conversation round limit for the main agent.",
    )
    parser.add_argument(
        "--enable-layered",
        action="store_true",
        default=True,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--enable-sbfl", action="store_true", default=False, help="Enable SBFL."
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable validation in our workflow.",
    )
    parser.add_argument(
        "--enable-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable angelic debugging",
    )
    parser.add_argument(
        "--enable-perfect-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable perfect angelic debugging; overrides --enable-angelic",
    )
    parser.add_argument(
        "--save-sbfl-result",
        action="store_true",
        default=False,
        help="Special mode to only save SBFL results for future runs.",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        default=False,
        help="Special mode to only generate reproducer tests",
    )
    parser.add_argument(
        "--num-processes",
        type=str,
        default=1,
        help="Number of processes to run the tasks in parallel.",
    )

    ### Extra
    parser.add_argument(
        "--output-fix-locs",
        action="store_true",
        required=False,
        default=False,
        help="Output fix locations to file and do not repair.",
    )
    parser.add_argument(
        "--output-fix-limit",
        type=int,
        required=False,
        default=10,
        help="Limit output of content retrieval rounds",
    )


def make_swe_tasks(
    task_id: str | None,
    task_list_file: str | None,
    setup_map_file: str,
    tasks_map_file: str,
    oracle_file: bool
) -> list[RawSweTask]:
    if task_id is not None and task_list_file is not None:
        raise ValueError("Cannot specify both task and task-list.")

    all_task_ids = []
    if task_list_file is not None:
        all_task_ids = parse_task_list_file(task_list_file)
    if task_id is not None:
        all_task_ids = [task_id]
    if len(all_task_ids) == 0:
        raise ValueError("No task ids to run.")

    with open(setup_map_file) as f:
        setup_map = json.load(f)
    with open(tasks_map_file) as f:
        tasks_map = json.load(f)

    if oracle_file:
        for sample in tasks_map:
            sample_patch = tasks_map[sample]["patch"]
            oracle_file_prompt = extract_file_from_patch(sample_patch)
            tasks_map[sample]["problem_statement"] = tasks_map[sample]["problem_statement"] + oracle_file_prompt

    # if globals.debug_config and os.path.isfile(globals.debug_config):
    #     with open(globals.debug_config, "r") as f:
    #         config = json.load(f)
    #     if "issue_prompt" in config.keys():
    #         add_prompt = "\n" + config["issue_prompt"]
    #         for sample in tasks_map:
    #             tasks_map[sample]["problem_statement"] = tasks_map[sample]["problem_statement"] + add_prompt


    # Check if all task ids are in the setup and tasks map
    # This allows failing safely if some tasks are not set up properly
    missing_task_ids = [
        x for x in all_task_ids if not (x in setup_map and x in tasks_map)
    ]
    if missing_task_ids:
        # Log the tasks that are not in the setup or tasks map
        for task_id in sorted(missing_task_ids):
            log.print_with_time(
                f"Skipping task {task_id} which was not found in setup or tasks map."
            )
        # And drop them from the list of all task ids
        all_task_ids = filter(lambda x: x not in missing_task_ids, all_task_ids)

    all_task_ids = sorted(all_task_ids)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    for task_id in all_task_ids:
        setup_info = setup_map[task_id]
        task_info = tasks_map[task_id]
        task = RawSweTask(task_id, setup_info, task_info)
        all_tasks.append(task)
    return all_tasks

def extract_file_from_patch(patch_string):
    """
    From a gold patch, extract the changed files.
    """
    gold_files = []
    loc = patch_string.split("\n")
    loc = [l for l in loc if l.startswith("---")]
    for line in loc:
        line = line.removeprefix("--- a/").rstrip()
        gold_files.append(line)
    if gold_files is not None:
        return f"\nIt is known that the bug is located in the following file(s): {gold_files}.\n"
    else:
        return ""


def parse_task_list_file(task_list_file: str) -> list[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file) as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


def group_swe_tasks_by_env(tasks: list[RawSweTask]) -> dict[str, list[RawSweTask]]:
    groups = {}
    for task in tasks:
        key = task.setup_info["env_name"]
        if key not in groups:
            groups[key] = []
        groups[key].append(task)
    return groups


def run_task_groups(
    task_groups: Mapping[str, Sequence[RawTask]],
    num_processes: int,
    organize_output: bool = False,
):
    """
    Main entry for running tasks.
    """
    all_tasks = list(chain.from_iterable(task_groups.values()))
    num_tasks = len(all_tasks)

    task_counter.init_total_num_tasks(num_tasks)

    # print some info about task
    log.print_with_time(f"Total number of tasks: {num_tasks}")
    log.print_with_time(f"Total number of processes: {num_processes}")
    log.print_with_time(f"Task group info: (number of groups: {len(task_groups)})")
    for key, tasks in task_groups.items():
        log.print_with_time(f"\t{key}: {len(tasks)} tasks")

    # if patch selection mode for multiple runs
    if config.patch_selection_from_multiple_runs:
        config.map_of_run_description_to_map_of_issue_patch = {}
        for description_and_path in config.list_of_run_patches_descriptions_and_paths:
            description = description_and_path["description"]
            path = description_and_path["path"]
            config.map_of_run_description_to_map_of_issue_patch[description] = apputils.read_file_and_create_map_between_issue_and_patch(path)

    # single process mode
    if num_processes == 1:
        log.print_with_time("Running in single process mode.")
        run_tasks_serial(all_tasks)
        log.print_with_time("Finished all tasks sequentially.")
    else:
        run_task_groups_parallel(task_groups, num_processes)

    if config.only_save_sbfl_result:
        log.print_with_time("Only saving SBFL results. Exiting.")
        return
    
    if config.patch_selection_from_multiple_runs:
        log.print_with_time("Only patch selection mode, creating swebench selections.")

    if organize_output:
        # post-process completed experiments to get input file to SWE-bench
        log.print_with_time("Post-processing completed experiment results.")
        swe_input_file = organize_and_form_input(config.output_dir)
        log.print_with_time(f"SWE-Bench input file created: {swe_input_file}")


def run_tasks_serial(tasks: list[RawTask]) -> None:
    for task in tasks:
        run_task_in_subprocess(task)


def run_task_groups_parallel(
    task_groups: Mapping[str, Sequence[RawTask]],
    num_processes: int,
):
    num_task_groups = len(task_groups)
    task_counter.init_total_num_task_groups(num_task_groups)
    num_processes = min(num_processes, num_task_groups)

    task_group_ids_items = sorted(
        task_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    log.print_with_time(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
    try:
        # Use ProcessPoolExecutor instead of multiprocessing.Pool,
        # to support nested sub-processing

        group_ids, group_tasks = zip(*task_group_ids_items)
        with ProcessPoolExecutor(num_processes) as executor:
            executor.map(run_task_group, group_ids, group_tasks)
    finally:
        log.print_with_time("Finishing all tasks in the pool.")


def run_task_group(task_group_id: str, task_group_items: list[RawTask]) -> None:
    """
    Run all tasks in a task group sequentially.
    Main entry to parallel processing.
    """
    log.print_with_time(
        f"Starting process for task group {task_group_id}. Number of tasks: {len(task_group_items)}."
    )
    for task in task_group_items:
        # within a group, the runs are always sequential
        run_task_in_subprocess(task)
        log.print_with_time(task_counter.incre_task_return_msg())

    log.print_with_time(
        f"{task_counter.incre_task_group_return_msg()} Finished task group {task_group_id}."
    )


def run_task_in_subprocess(task: RawTask) -> None:
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(run_raw_task, task)


def run_raw_task(task: RawTask) -> bool:
    """
    High-level entry for running one task.

    Args:
        - task: The Task instance to run.

    Returns:
        Whether the task completed successfully.
    """
    if config.only_eval_reproducer:
        assert isinstance(task, RawSweTask)
        evaluate_swe_issue_reproducers(task)
        return True

    task_id = task.task_id
    start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_output_dir = pjoin(config.output_dir, f"{task_id}_{start_time_s}")

    # if globals.patch_generation_only:

    #     try:
    #         # find existing directory with fix locations
    #         dir_prefix = pjoin(globals.patch_input_dir.removesuffix("no_patch"), "no_patch")
    #         task_dirs = [x for x in os.listdir(dir_prefix) if x.startswith(f"{task_id}_")]
    #         if not task_dirs:
    #             print("\nError finding existing directory with fix locations\n")
    #             return False
    #         else:
    #             task_input_dir = task_dirs[-1]
    #             for task_dir in task_dirs:
    #                 if os.path.isfile(pjoin(task_dir, "agent_fix_locations_1.json")):
    #                     task_input_dir = task_dir
    #             # # copy existing files to new task output directory
    #             # print(f"\nCopying directory {pjoin(dir_prefix, task_input_dir)}\n")
    #             # shutil.copytree(pjoin(dir_prefix, task_input_dir), task_output_dir)
    #             # save specific task input directory
    #             globals.task_input_dir = pjoin(dir_prefix, task_input_dir)

    #     except Exception as e:
    #         logger.exepction(e)
    #         return False
        
        
    apputils.create_dir_if_not_exists(task_output_dir)

    task.dump_meta_data(task_output_dir)

    log.log_and_always_print(
        f"============= Running task {task_id} =============",
    )

    run_ok = False

    try:
        run_ok = do_inference(task.to_task(), task_output_dir)

        if run_ok:
            run_status_message = f"Task {task_id} completed successfully."
        else:
            run_status_message = f"Task {task_id} failed without exception."
    except Exception as e:
        logger.exception(e)
        run_status_message = f"Task {task_id} failed with exception: {e}."

    log.log_and_always_print(run_status_message)

    final_patch_path = get_final_patch_path(task_output_dir)
    if final_patch_path is not None:
        log.log_and_always_print(
            f"Please find the generated patch at: {final_patch_path}"
        )
        if isinstance(task, RawSweTask):
            log.log_and_always_print(
                "[SWE-bench mode] Note that the patch may be move to other paths in SWE-bench mode. "
                "Please check the SWE-bench input file containing generated patches for all tasks."
            )
    else:
        log.log_and_always_print("No patch generated. You can try running ACR again.")

    return run_ok


def evaluate_swe_issue_reproducers(raw_task: RawSweTask) -> None:
    swe_task = raw_task.to_task()
    swe_task.setup_project()

    reproducer_files = glob(
        pjoin(
            config.output_dir, "**", f"*{swe_task.task_id}*", "**", "reproducer_*.py"
        ),
        recursive=True,
    )
    for reproducer_file in reproducer_files:
        evaluate_swe_issue_reproducer(swe_task, reproducer_file)


def evaluate_swe_issue_reproducer(
    task: SweTask, reproducer_file: str | PathLike
) -> None:
    reproducer_file = Path(reproducer_file)

    individual_expr_dir = reproducer_file.parent

    developer_patch_file = individual_expr_dir.joinpath("developer_patch.diff")
    if not developer_patch_file.exists():
        individual_expr_dir = individual_expr_dir.parent
        developer_patch_file = individual_expr_dir.joinpath("developer_patch.diff")
    assert developer_patch_file.exists()

    report_dir = individual_expr_dir.joinpath(
        "reproducer-eval",
        *reproducer_file.relative_to(individual_expr_dir).with_suffix("").parts,
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    task.evaluate_reproducer(reproducer_file, developer_patch_file, report_dir)


def do_inference(python_task: Task, task_output_dir: str) -> bool:
    apputils.create_dir_if_not_exists(task_output_dir)

    log_file_name = "info.log"

    # Logger format

    logger.add(
        pjoin(task_output_dir, log_file_name),
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
            " | <level>{message}</level>"
        ),
    )

    # logger.add(
    #     pjoin(task_output_dir, log_file_name),
    #     level="DEBUG",
    #     format="| <level>{level: <8}</level> | <level>{message}</level>"
    # )

    start_time = datetime.now()

    python_task.setup_project()

    try:
        if config.only_save_sbfl_result:
            api_manager = ProjectApiManager(python_task, task_output_dir)
            _, _, run_ok = api_manager.fault_localization()

        elif config.only_reproduce:
            api_manager = ProjectApiManager(python_task, task_output_dir)
            _, _, run_ok = api_manager.reproduce()

        elif config.patch_selection_from_multiple_runs:
            api_manager = ProjectApiManager(python_task, task_output_dir)
            _, _, run_ok = api_manager.select_patch(config.map_of_run_description_to_map_of_issue_patch)

        else:
            # normal mode - actually running the task

            try:
                run_ok = inference.run_one_task(
                    python_task, task_output_dir, config.models
                )

            except common.ClaudeContentPolicyViolation:
                log.log_and_always_print(
                    "Content policy violation. Retry with backup model."
                )

                # retry with backup model
                python_task.setup_project()

                # remove everything other than the info.log file, and
                # also some meta data file dumped by RawTask
                log.log_and_always_print(
                    "Removing all files except info.log and meta files."
                )

                for f in Path(task_output_dir).iterdir():
                    if f.is_file() and f.name not in [
                        log_file_name,
                        "meta.json",
                        "problem_statement.txt",
                        "developer_patch.diff",
                    ]:
                        f.unlink()
                    if f.is_dir():
                        shutil.rmtree(str(f))

                run_ok = inference.run_one_task(
                    python_task, task_output_dir, config.backup_model
                )

            end_time = datetime.now()
            with apputils.cd(python_task.project_path):
                dump_cost(start_time, end_time, task_output_dir)
    finally:
        python_task.reset_project()

    return run_ok


def dump_cost(
    start_time: datetime,
    end_time: datetime,
    task_output_dir: str,
):
    model_stats = common.SELECTED_MODEL.get_overall_exec_stats()
    stats = {
        "commit": apputils.get_current_commit_hash(),
        "start_epoch": start_time.timestamp(),
        "end_epoch": end_time.timestamp(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }
    stats.update(model_stats)

    with open(pjoin(task_output_dir, "cost.json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        # Macos specific requirement for Multi-Processing
        set_start_method("fork", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.remove()
    main()
