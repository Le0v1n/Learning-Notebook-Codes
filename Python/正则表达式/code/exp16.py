import re

pattern = r"apple|banana"
text = "I like bananas and apple"

match = re.search(pattern, text)
if match:
    print("Match found:", match.group())
else:
    print("No match") 