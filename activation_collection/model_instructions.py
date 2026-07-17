INSTRUCTIONS = {
    "ask-tf": "Is the following statement TRUE or FALSE?\n{statement} Answer:",
    "ask-able": "Are you able to evaluate the following statement according to its truthfulness?\n{statement} Answer:",
    "ask-arith": "Are you able to evaluate the following arithmetic expression according to its correctness?\n{statement} Answer:",
    "ask-correct": "Is the following correct?\n{statement} Answer:",
    "no-prompt": "{statement}",
    "random-prompt": "Green table running bright.\n{statement} Answer:",
    "read-prompt": "Read the following sentence.\n{statement} Answer:",
}


def build_prompt(instruction, ex):
    # Period-align the read-out position across tasks: ensure the statement ends
    # in a period. Factual sentences already do; arithmetic (e.g. "46-75=-30")
    # gets one appended. This makes the no-prompt last token a period for all
    # datasets, and puts a period before " Answer:" under instructed prompts.
    statement = ex["__text__"].rstrip()
    if not statement.endswith("."):
        statement += "."
    template = INSTRUCTIONS[instruction]
    return template.format(statement=statement), {}


def label(ex):
    return bool(ex["is_correct"]), {}
