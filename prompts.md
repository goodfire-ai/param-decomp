# Logical Deduction Prompts

Prompts where the model must deduce the next token through some form of reasoning.
Tested against the `pile_llama_simple_mlp-4L` model (run `s-55ea3f9b`).

## Tier 1: Very strong deduction (>50% prob)


| Prompt                                            | Top prediction | Prob  | Deduction type               |
| ------------------------------------------------- | -------------- | ----- | ---------------------------- |
| `["a", "b", "c", "`                               | `d`            | 0.973 | Alphabetic list continuation |
| `def add(a, b):\n return a +`                     | `b`            | 0.943 | Function name → body         |
| `if x:\n y = True\nelse:\n y =`                   | `False`        | 0.600 | If/else boolean opposite     |
| `a:1 b:2 c:`                                      | `3`            | 0.597 | Key-value pattern            |
| `The king sat on his`                             | `throne`       | 0.574 | Factual deduction            |
| `x = True\nif x:\n print("yes")\nelse:\n print("` | `no`           | 0.474 | Structural else → opposite   |
| `if x > 0:\n return True\nelse:\n return`         | `False`        | 0.515 | If/else boolean opposite     |


## Tier 2: Good deduction (20-50% prob)


| Prompt                                               | Top prediction | Prob  | Deduction type         |
| ---------------------------------------------------- | -------------- | ----- | ---------------------- |
| `if x == 0:\n return False\nelse:\n return`          | `True`         | 0.409 | Boolean opposite       |
| `yes no yes no yes`                                  | `no`           | 0.364 | Alternation pattern    |
| `x = 1 if True else`                                 | `0`            | 0.352 | Ternary "else" value   |
| `if True and False:\n print("yes")\nelse:\n print("` | `no`           | 0.339 | Boolean logic via code |
| `def is_even(x):\n return x %`                       | `2`            | 0.287 | Function name → modulo |
| `x=1 y=2 z=`                                         | `3`            | 0.236 | Pattern continuation   |


## Tier 3: Weaker but interesting


| Prompt                        | Top prediction | Prob  | Deduction type                              |
| ----------------------------- | -------------- | ----- | ------------------------------------------- |
| `return True if x else`       | `False`        | 0.186 | Ternary opposite                            |
| `def square(x):\n return x *` | `x`            | 0.177 | Function name → operation                   |
| `dog is to dogs as cat is to` | `cats`         | 0.136 | Analogy (2nd place; `dogs` is 1st at 0.227) |
| `def half(x):\n return x /`   | `2`            | 0.131 | Function name → divisor                     |
| `Q: Is the sun hot? A:`       | `Yes`          | 0.126 | Q&A factual                                 |
| `Q: Is water wet? A:`         | `Yes`          | 0.124 | Q&A factual                                 |


## Notes

- **Structural vs semantic deduction**: `x = True\nif x: print("yes") else: print("` always
predicts `no` regardless of x's value (True, False, or 0). The model does *structural* deduction
(else branch → opposite of "yes") rather than tracking the variable. Interesting to analyze in
the graph.
- **Function name deduction**: `def add(a, b): return a +` → `b` at 0.943 is the strongest
example. The model infers the function body from the name alone.
- **The model can't do arithmetic**: `1 + 1 =` → top prediction is  `0` (0.206), not  `2`.
- **Boolean concepts don't work outside code**: `True and False is` →  `the` (0.194). The model
only handles boolean logic when embedded in code structure (if/else).

