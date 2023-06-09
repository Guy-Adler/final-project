"""
When characters do not match the type for their position, use these mappings to get the most
likely character which does.

Using the confusion matrix, each predicted character can be mapped to a list with the probability of true characters.
For example, take the following confusion matrix, where the true values are the rows and the predicted are the columns:

+---+---+------+------+------+
| A | : | 0.80 | 0.02 | 0.04 |
+---+---+------+------+------+
| B | : | 0.05 | 0.93 | 0.20 |
+---+---+------+------+------+
| C | : | 0.15 | 0.05 | 0.76 |
+---+---+------+------+------+
|       |   A  |   B  |   C  |
+---+---+------+------+------+
|       |      Predicted     |
+---+---+------+------+------+

The likely ture characters for a prediction of 'A' are (in sorted order): 'A', 'C', 'B'.

The maps take the most likely character that matches the character type.
"""
MAPPINGS = {'0': {'a': 'O', 'A': 'O', 'd': '0', 'D': '0', 'S': 'F'},
            '1': {'a': 'T', 'A': 'T', 'd': '1', 'D': '1', 'S': 'M'},
            '2': {'a': 'E', 'A': 'E', 'd': '2', 'D': '2', 'S': 'M'},
            '3': {'a': 'I', 'A': 'I', 'd': '3', 'D': '3', 'S': 'F'},
            '4': {'a': 'A', 'A': '<', 'd': '4', 'D': '4', 'S': 'F'},
            '5': {'a': 'S', 'A': 'S', 'd': '5', 'D': '5', 'S': 'F'},
            '6': {'a': 'J', 'A': 'J', 'd': '6', 'D': '6', 'S': 'F'},
            '7': {'a': 'Y', 'A': 'Y', 'd': '7', 'D': '7', 'S': 'F'},
            '8': {'a': 'Z', 'A': 'Z', 'd': '8', 'D': '8', 'S': 'F'},
            '9': {'a': 'F', 'A': 'F', 'd': '9', 'D': '9', 'S': 'F'},
            'A': {'a': 'A', 'A': 'A', 'd': '4', 'D': '4', 'S': 'M'},
            'B': {'a': 'B', 'A': 'B', 'd': '8', 'D': '8', 'S': 'M'},
            'C': {'a': 'C', 'A': 'C', 'd': '7', 'D': '7', 'S': 'F'},
            'D': {'a': 'D', 'A': 'D', 'd': '6', 'D': '6', 'S': 'F'},
            'E': {'a': 'E', 'A': 'E', 'd': '4', 'D': '4', 'S': 'F'},
            'F': {'a': 'F', 'A': 'F', 'd': '6', 'D': '6', 'S': 'F'},
            'G': {'a': 'G', 'A': 'G', 'd': '9', 'D': '9', 'S': 'F'},
            'H': {'a': 'H', 'A': 'H', 'd': '5', 'D': '5', 'S': 'M'},
            'I': {'a': 'I', 'A': 'I', 'd': '0', 'D': '0', 'S': 'F'},
            'J': {'a': 'J', 'A': 'J', 'd': '9', 'D': '9', 'S': 'F'},
            'K': {'a': 'K', 'A': 'K', 'd': '7', 'D': '<', 'S': 'F'},
            'L': {'a': 'L', 'A': 'L', 'd': '6', 'D': '6', 'S': 'F'},
            'M': {'a': 'M', 'A': 'M', 'd': '7', 'D': '7', 'S': 'M'},
            'N': {'a': 'N', 'A': 'N', 'd': '7', 'D': '7', 'S': 'M'},
            'O': {'a': 'O', 'A': 'O', 'd': '0', 'D': '0', 'S': 'F'},
            'P': {'a': 'P', 'A': 'P', 'd': '0', 'D': '0', 'S': 'F'},
            'Q': {'a': 'Q', 'A': 'Q', 'd': '4', 'D': '4', 'S': 'F'},
            'R': {'a': 'R', 'A': 'R', 'd': '9', 'D': '9', 'S': 'M'},
            'S': {'a': 'S', 'A': 'S', 'd': '8', 'D': '8', 'S': 'F'},
            'T': {'a': 'T', 'A': 'T', 'd': '5', 'D': '5', 'S': 'F'},
            'U': {'a': 'U', 'A': 'U', 'd': '5', 'D': '5', 'S': 'F'},
            'V': {'a': 'V', 'A': 'V', 'd': '9', 'D': '9', 'S': 'M'},
            'W': {'a': 'W', 'A': 'W', 'd': '4', 'D': '4', 'S': 'F'},
            'X': {'a': 'X', 'A': 'X', 'd': '2', 'D': '2', 'S': 'F'},
            'Y': {'a': 'Y', 'A': 'Y', 'd': '7', 'D': '7', 'S': 'M'},
            'Z': {'a': 'Z', 'A': 'Z', 'd': '4', 'D': '4', 'S': 'F'},
            '<': {'a': 'M', 'A': '<', 'd': '7', 'D': '<', 'S': 'M'}
            }
