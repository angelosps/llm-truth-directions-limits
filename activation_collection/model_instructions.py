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
    # Make every statement (including arithmetic) to end with a period
    statement = ex["__text__"].rstrip()
    if not statement.endswith("."):
        statement += "."
    template = INSTRUCTIONS[instruction]
    return template.format(statement=statement), {}


def label(ex):
    return bool(ex["is_correct"]), {}
