import os
import shutil
import json
import filecmp

from app.raw_tasks import RawSweTask
from app.task import SweTask
from app.api.eval_helper import (
    ResolvedStatus,
    get_eval_report,
    get_logs_eval,
    get_resolution_status,
)

DEBUG = False

def validate(tasks: list[RawSweTask], tasks_dir: str, gold_only: bool=False, init_only: bool=False):
    """
    Run test suites to validate previously generated patches for a list of SWE-bench tasks. 
    """

    for task in tasks:
        if gold_only:
            # validate gold patches
            validate_gold(task, tasks_dir)
        elif init_only:
            validate_init(task, tasks_dir)
        else:               
            # set up task
            task = task.to_task()
            task.setup_project()

            # validate existing generated patches
            validate_tasks(task, tasks_dir)


def validate_tasks(task: SweTask, tasks_dir: str):
    """
    Run test suite to validate previously generated patches for a single SWE-bench task. 
    """

    tasks_dir = tasks_dir.split("/applicable_patch")[0]

    # 1. Loop over existing sub-directories with applicable patch(es) for given task
    with os.scandir(os.path.join(tasks_dir, "applicable_patch")) as it:
        task_dirs = [entry.path for entry in it if entry.name.startswith(task.task_id) and entry.is_dir()]

    for task_dir in task_dirs:
        
        # 2. Find existing applicable patch diff(s) by parsing extract_status file
        assert os.path.isfile(os.path.join(task_dir, "extract_status.json")), f"Task directory {task_dir} does not have file extract_status.json"
        with open(os.path.join(task_dir, "extract_status.json"), "r") as f:
            extract_json = json.load(f)

            
        # Iterate over applicable patches
        for i, status in enumerate(extract_json["extract_status"]):
            if status == "APPLICABLE_PATCH":
                
                diff_file = os.path.join(task_dir, f"extracted_patch_{i+1}.diff")
                assert os.path.isfile(diff_file), f"Patch {diff_file} does not exist for task {task.task_id}"

                # 3. Validate patch
                patch_is_correct, err_message, log_file = task.validate(diff_file)
                shutil.move(log_file, os.path.join(task_dir, f"post_validate_patch_{i+1}.log"))

                # Check current validation results match any existing previous validation results
                if DEBUG and os.path.isfile(os.path.join(task_dir, f"run_test_suite_{i+1}.log")):
                    # post-validation file exists
                    post_val_file = os.path.join(task_dir, f"post_validate_patch_{i+1}.log")
                    assert os.path.isfile(post_val_file), f"File does not exist: {post_val_file}"
                    
                    # post-validation logged results equivalent to previous validation logged results
                    init_val_file = os.path.join(task_dir, f"run_test_suite_{i+1}.log")
                    eval_status_post, parse_ok_post = get_logs_eval(task.repo_name, post_val_file)
                    eval_status_init, parse_ok_init = get_logs_eval(task.repo_name, init_val_file)
                    if parse_ok_post and parse_ok_init:
                        eval_ref = {
                            "FAIL_TO_PASS": task.testcases_failing,
                            "PASS_TO_PASS": task.testcases_passing,
                        }
                        eval_result_post, eval_result_init = get_eval_report(eval_status_post, eval_ref), get_eval_report(eval_status_init, eval_ref)
                        assert eval_result_post == eval_result_init, f"Evaluation results not equivalent: {post_val_file}, {init_val_file}"
                        print("check")


def validate_gold(task: RawSweTask, tasks_dir: str):
    """
    Validate provided developer gold patch for a given SWE-bench task.
    """

    # Added for debugging
    # print(f"\n\ntest_cmd:\t{task.setup_info["test_cmd"]}\npre_install_cmd: {task.setup_info["pre_install"]}\ninstall_cmd:\t{task.setup_info["install"]}\nrepo_path:\t{task.setup_info["repo_path"]}\ncommit:\t\t{task.task_info["base_commit"]}\n\n")

    # Make inidivual task directory if needed, and add provided meta data
    task_dir = os.path.join(tasks_dir, task.task_id)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    task.dump_meta_data(task_dir)

    # Find developer patch diff
    diff_file = os.path.join(task_dir, "developer_patch.diff")
    assert os.path.isfile(diff_file), f"Developer gold diff not found at {diff_file}"

    # Validate developer gold patch
    task = task.to_task()
    task.setup_project()
    patch_is_correct, err_message, log_file = task.validate(diff_file)
    shutil.move(log_file, os.path.join(task_dir, "run_test_suite.log"))

    # Added for debugging
    # print(f"\n\ntest_cmd:\t{task.test_cmd}\n\n")

def validate_init(task: RawSweTask, tasks_dir: str):
    """
    Run test suite without applying apatch for a given SWE-bench task.
    """

    print("Starting call to run test suite wihtout applying patch.")

    # Make inidivual task directory if needed, and add provided meta data
    task_dir = os.path.join(tasks_dir, task.task_id)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    task.dump_meta_data(task_dir)

    # Run testsuite
    print(f"\ntest_cmd:\t{task.setup_info['test_cmd']}")
    # test_cmd:       ./tests/runtests.py --verbosity 2 migrations.test_commands
    task = task.to_task()
    task.setup_project()
    print(f"fail_to_pass tests: {task.testcases_failing}")
    print(f"project_path: {task.project_path}")

    patch_is_correct, err_message, log_file = task.validate(None)
    shutil.move(log_file, os.path.join(task_dir, "run_test_suite_init.log"))

    eval_status, parse_ok = get_logs_eval(task.repo_name, os.path.join(task_dir, "run_test_suite_init.log"))
    eval_ref = {
            "FAIL_TO_PASS": task.testcases_failing,
            "PASS_TO_PASS": task.testcases_passing,
        }
    eval_result = get_eval_report(eval_status, eval_ref)
    print(f"\n\neval_status: {eval_result}")
    print(f"\n\nfail_to_pass tests: {eval_result['FAIL_TO_PASS']}")

