import re
import argparse
import subprocess
import sys

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"  # https://docs.python.org/3/library/re.html#simulating-scanf

def parse_output(content, timeout):
    props = {}
    if "Solution 0" not in content:
        # Timeout
        props["time"] = timeout + 1
        props["train_mse"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    # STreeD
    txt_pattern = {
        "terminal_calls": (r"Terminal calls: (\d+)", int),
        "solution": (r"Solution 0:\s+(.*)", str),
        "time": (r"CLOCKS FOR SOLVE: (" + float_pattern + ")", float),
    }

    matches = {}
    for i in txt_pattern:
        matches[i] = txt_pattern[i][1](
            re.search(txt_pattern[i][0], content, re.M).group(1)
        )
    
    # depth, branching nodes, train, test, avg. path length
    solution_vals = matches["solution"].split()

    props["terminal_calls"] = matches["terminal_calls"]
    props["time"] = matches["time"]
    props["leaves"] = int(solution_vals[1]) + 1  # Solution prints branching nodes, but want leaf nodes
    props["train_mse"] = float(solution_vals[2])
    props["test_mse"] = float(solution_vals[3])
    return props


def run_streed(
    exe,
    timeout,
    depth,
    train_data,
    test_data,
    cp,
    use_kmeans,
    use_task_bound,
    use_lower_bound,
    use_d2, # TODO: add CLI param to streed to toggle terminal solver
):
    try:
        result = subprocess.check_output(
            ["timeout", str(timeout), exe,
             "-task", "cost-complex-regression",
             "-file", train_data,
             "-test-file", test_data,
             "-max-depth", str(depth),
             "-max-num-nodes", str(2**depth - 1),
             "-time", str(timeout + 10),
             "-use-lower-bound", "1" if use_lower_bound else "0",
             "-use-task-lower-bound", "1" if use_task_bound else "0",
             "-regression-bound", "kmeans" if use_kmeans else "equivalent",
             "-cost-complexity", str(cp)], timeout=timeout)
        output = result.decode()
        # print(output)
        parsed = parse_output(output, timeout)
        return parsed
    except subprocess.TimeoutExpired as e:
        # print(e.stdout.decode())
        return parse_output("", timeout)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode(), file=sys.stderr, flush=True)
        return {
            "time": -1,
            "train_mse": -1,
            "leaves": -1,
            "terminal_calls": -1
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="STreeD Runner", description="Runs, parses, and saves STreeD output"
    )
    parser.add_argument("exe", help="STreeD executable", required=True)
    parser.add_argument(
        "--id",
        help="Additional string to identify this run, besides the other attributes",
        default="",
    )
    parser.add_argument(
        "--timeout", type=int, help="Timeout for finding the tree", default=60
    )
    parser.add_argument("--depth", type=int, help="Maximum tree depth", required=True)
    parser.add_argument("--train-data", help="Training dataset", required=True)
    parser.add_argument("--test-data", help="Test dataset", required=True)
    parser.add_argument(
        "--complexity-penalty",
        type=float,
        help="Weight for penalizing trees",
        required=True,
    )
    parser.add_argument(
        "--no-kmeans", action="store_true", help="Use attribute to disable kmeans bound"
    )
    parser.add_argument(
        "--no-task-bound",
        action="store_true",
        help="Use attribute to disable task lower bound",
    )
    parser.add_argument(
        "--no-lower-bound",
        action="store_true",
        help="Use attribute to disable lower bound",
    )
    parser.add_argument(
        "--no-d2", action="store_true", help="Use attribute to disable depth-2 solver"
    )

    args = parser.parse_args()
    result = run_streed(
        args.exe,
        args.timeout,
        args.depth,
        args.train_data,
        args.test_data,
        args.complexity_penalty,
        not args.no_kmeans,
        not args.no_task_bound,
        not args.no_lower_bound,
        not args.no_d2,
    )

    for i in result:
        print(f"Output for STreeD ({args.id}) run:")
        print(f"{i}={result[i]}")