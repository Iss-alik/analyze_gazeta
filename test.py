import re

text = "Hello\n\nWorld\n\n\nPython\n\n\n\nProgramming"

# Split the string where two or more newlines appear
result = re.split(r'\n{2,}', text)

print(result)