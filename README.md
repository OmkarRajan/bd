# 1.Map Reduce program to read a text file count the number of occurrences of the words.

# 1. mapper.py

- mapper.py

```py
#!/usr/bin/env python
import sys
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        print '%s\t%s' % (word, 1)

```

# 1. reducer.py

- reducer.py

```py
#!/usr/bin/env python
from operator import itemgetter
import sys
current_word = None
current_count = 0
word = None
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)
    try:
        count = int(count)
    except ValueError:
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print '%s\t%s' % (current_word, current_count)
        current_count = count
        current_word = word
if current_word == word:
    print '%s\t%s' % (current_word, current_count)

```
# 
# 2: Map Reduce program to display the number of transactions and sales count from shopping cart.

# 2. mapper.py

```py
#!/usr/bin/env python
import string
import fileinput
for line in fileinput.input():
    data = line.strip().split("\t")
    if len(data) == 6:
        date, time, location, item, cost, payment = data
        print "{0}\t{1}".format(location, cost)
```

# 2. reducer.py

```py
#!/usr/bin/env python
import fileinput
transactions_count = 0
sales_total = 0
for line in fileinput.input():
    data = line.strip().split("\t")    
    if len(data) != 2:
        continue
    current_key, current_value = data
    transactions_count += 1
    sales_total += float(current_value)
print transactions_count, "\t", sales_total

```
# 
# 3: Map Reduce program to calculate the average age for each gender.
# 3. mapper.py

```py
#!/usr/bin/python
import sys
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")
    if len(line) >=2:
        gender = line[1]
        age = line[2]
        print '%s\t%s' % (gender, age)
```

# 3. reducer.py

```py
#!/usr/bin/python
import sys
gender_age = {}
for line in sys.stdin:
    line = line.strip()
    gender, age = line.split('\t')
    if gender in gender_age:
        gender_age[gender].append(int(age))
    else:
        gender_age[gender] = []
        gender_age[gender].append(int(age))
for gender in gender_age.keys():
    ave_age = sum(gender_age[gender])*1.0 / len(gender_age[gender])
    print '%s\t%s'% (gender, ave_age)
```
# 3. age.csv

```
1,1,21,5.4,50
2,2,22,5.5,53
3,1,23,5.6,44
4,2,24,5.7,59
5,1,25,5.8,35
6,1,30,5.9,60
7,2,35,5.10,55
```

#
# 4. TERMWORK4
# 4. mapper.py

```py
#!/usr/bin/env python
import sys
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")
    if len(line) >=2:
        pid = line[0]
        opinion = line[4]
        print '%s\t%s' % (pid, opinion)
```

# 4. reducer.py

```py
#!/usr/bin/env python
import sys
opiniondic={}
count=0
for line in sys.stdin:
    line = line.strip()
    pid, opinion = line.split('\t')
    if opinion in opiniondic:
        opiniondic[opinion].append(count+1)
    else:
        opiniondic[opinion] = []
        opiniondic[opinion].append(count+1)
for op in opiniondic.keys():
    count=len(opiniondic[op])
    print '%s\t%s'% (op,count)
```
# 4. opinion.csv

```
1,M,25000,2,Agree
2,F,50000,1,DisAgree
3,M,75000,0,neutral
4,F,85000,2,Agree
5,M,35000,1,DisAgree
6,F,95000,0,neutal
7,M,45000,2,Agree
```

#
# 5. TERMWORK5
# 5. mapper.py

```py
#!/usr/bin/env python
import sys
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")
    if len(line) >=2:
        dept = line[2]
        sal = line[3]
        print '%s\t%s' % (dept, sal)
```

# 5. reducer.py

```py
#!/usr/bin/env python
import sys
deptdic={}


for line in sys.stdin:
    line = line.strip()
    dept,sal = line.split('\t')
    if dept in deptdic:
        deptdic[dept].append(int(sal))
    else:
        deptdic[dept] = []
        deptdic[dept].append(int(sal))
for dept in deptdic.keys():
    sum_sal = sum(deptdic[dept])
    print '%s\t%s'% (dept,sum_sal)
```
# 5. salary.csv

```
E001,sunita,account,15000
E002,anil,it,50000
E003,janavi,marketing,75000
E004,suny,account,85000
E005,sunita,it,95000
E006,anita,marketing,55000
E007,sunil,account,45000
```
# 6. commond

- cmd

```
jps

hdfs dfs -mkdir /jp

hdfs dfsadmin -safemode leave

hdfs dfs -mkdir /jp

hdfs dfs -1s /

hdfs dfs-copyFromLocal/home/hduser/Documents/janavi/word.txt /jp

hdfs dfs -1s /jp

1s-1

hadoop jar/home/hduser/Downloads/hadoop-streaming-2.7.3.jar \

-input/jp/word.txt \

-output/jp/jp123 \

-mapper /home/hduser/Documents/janavi/mapper.py \

-reducer /home/hduser/Documents/janavi/reducer.py

hdfs dfs -cat /jp/jp123/part-00000

```

