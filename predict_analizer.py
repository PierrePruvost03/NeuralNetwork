import os
import sys


class bcolors:
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def print_color(nb):
    if nb > 0.70:
        print(bcolors.OKGREEN, end="")
        return
    if nb > 0.50:
        print(bcolors.WARNING, end="")
        return
    print(bcolors.FAIL, end="")


def reset_color():
    print(bcolors.ENDC, end="")


if len(sys.argv) < 2:
    print("expected dataset")
    exit(1)

dataset = open(sys.argv[1])


global_total = 0
global_correct = 0
global_correct_with_color = 0

total_nothing = 0
nothing_correct = 0

total_check = 0
check_correct = 0
check_correct_with_color = 0

total_checkmate = 0
checkmate_correct = 0
checkmate_correct_with_color = 0

total_check_and_mate = 0
check_and_mate_correct = 0
check_and_mate_correct_with_color = 0


def print_result(total, count):
    if total != 0:
        print_color(count / total)
        print("%.3f%%" % ((count / total) * 100.0))
        reset_color()
    else:
        print("----")


def print_colored(name, total, count, colored_count):
    print(name)
    print("Without color: ", end="")
    print_result(total, count)
    print("With color:    ", end="")
    print_result(total, colored_count)
    print("")


def print_all_result():
    print_colored("Global:", global_total, global_correct, global_correct_with_color)

    print("Nothing:       ", end="")
    print_result(total_nothing, nothing_correct)
    print("")

    print_colored("Check:", total_check, check_correct, check_correct_with_color)
    print_colored(
        "Checkmate:", total_checkmate, checkmate_correct, checkmate_correct_with_color
    )
    print_colored(
        "Checkmate and Check combined:",
        total_check_and_mate,
        check_and_mate_correct,
        check_and_mate_correct_with_color,
    )


line_in_file = sum(1 for _ in open(sys.argv[1]))
line_count = 0

for line in sys.stdin:
    expected_result = dataset.readline()[0:-1].split(" ")[6:8]
    got_result = line[0:-1].split(" ")

    def update(total, correct, with_color):
        total += 1
        if expected_result[0] == got_result[0]:
            correct += 1
            if (
                len(expected_result) > 1
                and len(got_result) > 1
                and [1] == got_result[1]
            ):
                with_color += 1
        return total, correct, with_color

    global_total, global_correct, global_correct_with_color = update(
        global_total, global_correct, global_correct_with_color
    )

    match expected_result[0]:
        case "Nothing":
            total_nothing += 1
            if expected_result[0] == got_result[0]:
                nothing_correct += 1

        case "Check":
            total_check, check_correct, check_correct_with_color = update(
                total_check, check_correct, check_correct_with_color
            )

        case "Checkmate":
            total_checkmate, checkmate_correct, checkmate_correct_with_color = update(
                total_checkmate, checkmate_correct, checkmate_correct_with_color
            )
    if (expected_result[0][0:5] == "Check") and (got_result[0][0:5] == "Check"):
        (
            total_check_and_mate,
            check_and_mate_correct,
            check_and_mate_correct_with_color,
        ) = update(
            total_check_and_mate,
            check_and_mate_correct,
            check_and_mate_correct_with_color,
        )

    if line_count != 0:
        print("\033[H\033[J", end="")
    nb_column = os.get_terminal_size().columns - 2
    percent = global_total / line_in_file
    nb_pipe = int(nb_column * percent)
    nb_space = nb_column - nb_pipe
    print(
        "[", bcolors.OKGREEN, "|" * nb_pipe, bcolors.ENDC, " " * nb_space, "]", sep=""
    )
    print("")
    print_all_result()
    line_count += 1


print("\033[H\033[J", end="")
print("RESULTS:\n")
print_all_result()
