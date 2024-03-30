import re

text = """<div class="el">
        <p class="t1">           
            <span>
                <a>Python开发工程师</a>
            </span>
        </p>
        <span class="t2">南京</span>
        <span class="t3">1.5-2万/月</span>
</div>
<div class="el">
        <p class="t1">
            <span>
                <a>java开发工程师</a>
            </span>
        </p>
        <span class="t2">苏州</span>
        <span class="t3">1.5-2/月</span>
</div>"""

p1 = r"class=\"t1\">.*?<a>(.*?)</a>"
p2 = r"(?s)class=\"t1\">.*?<a>(.*?)</a>"

matches1 = re.findall(pattern=p1, string=text, flags=re.DOTALL)
matches2 = re.findall(pattern=p2, string=text)
print(f"{matches1 = }")
print(f"{matches2 = }")