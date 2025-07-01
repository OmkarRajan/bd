
# 1. Pgm1:BFS ALGORITHM
```py
import matplotlib.pyplot as plt
import networkx as nx

graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

visited = []
queue = []

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)

    while queue:
        m = queue.pop(0)
        print(m, end=" ")

        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

print("Following is the Breadth-First Search")
bfs(visited, graph, '5')

G = nx.Graph(graph)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=12, font_weight="bold")
plt.title("Graph Visualization")
plt.show()


```

# 2. Pgm 2: DFS
```py
import matplotlib.pyplot as plt
import networkx as nx

graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

visited = set()

def dfs(visited, graph, node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

print("Following is the Depth-First Search")
dfs(visited, graph, '5')

G = nx.Graph(graph)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=12, font_weight="bold")
plt.title("Graph Visualization")
plt.show()


```

#  Pgm 3:TOWER OF HANOI 

```
def tower_of_hanoi(disks, source, auxiliary, target):
    if disks == 1:
        print('Move disk 1 from rod {} to rod {}.'.format(source, target))
        return
    tower_of_hanoi(disks - 1, source, target, auxiliary)
    print('Move disk {} from rod {} to rod {}.'.format(disks, source, target))
    tower_of_hanoi(disks - 1, auxiliary, source, target)

disks = int(input('Enter the number of disks: '))
tower_of_hanoi(disks, 'A', 'B', 'C')

```

# Pgm 4: AI Simple Chat Bot 

```py
print("How are you?") 
print("Are you working?") 
print("What is your name?") 
print("what did you do yesterday?") 
print("Quit") 
 
while True: 
    question = input("Enter one question from above list:") 
    question = question.lower() 
    if question in ['hi']: 
        print("Hello") 
    elif question in ['how are you?','how do you do?']: 
        print("I am fine") 
    elif question in ['are you working?','are you doing any job?']: 
        print("yes. I'am working in KLU") 
    elif question in ['what is your name?']: 
        print("My name is Emilia") 
        name=input("Enter your name?") 
        print("Nice name and Nice meeting you", name) 
    elif question in ['what did you do yesterday?']: 
        print("I saw Bahubali 5 times") 
    elif question in ['quit']: 
      break 
    else: 
                print("I don't understand what you said")

```


# Pgm 5: Linear regression  

```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 4, 5, 6])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear regression line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print('Intercept:', model.intercept_)
print('Slope:', model.coef_[0])

```

# Pgm 6: Implementation of Hangman Game in Python


```py
import time 
from time import sleep 
name = input("Enter Your Name:") 
print( "Hello" + name) 
print("Get ready!!") 
print ("") 
time.sleep(1) 
print ("Let us play Hangman!!") 
time.sleep(0.5) 
word = "Flower" 
wrd = '' 
chance = 10  
while chance > 0:          
    failed = 0             
    for char in word:       
        if char in wrd:     
            print (char)    
        else: 
            print( "_")     
            failed += 1    
    if failed == 0:         
        print( "You Won!!Congratulations!!" )  
        break              
    guess = input("Guess a Letter:")  
    wrd = wrd+guess                     
    if guess not in word:   
        chance -= 1        
        print ("Wrong Guess! Try Again") 
        print ("You have", + chance, 'more turn' ) 
        if chance == 0:            
            print ("You Lose! Better Luck Next Time" )
```
# Pgm 7: Write a Program to implement the Time series using python.


```py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
time_series = np.cumsum(np.random.randn(100)) + 50

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window_size = 5
ma = moving_average(time_series, window_size)

plt.figure(figsize=(10, 5))
plt.plot(time_series, label='Original Time Series')
plt.plot(range(window_size - 1, len(ma) + window_size - 1), ma, label='Moving Average', color='red')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Simple Moving Average")
plt.legend()
plt.show()

```
# Pgm 8: K Means Clustering.


```py
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans 
# Generate some example data 
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0) 
# Create a K-Means model 
kmeans = KMeans(n_clusters=4) 
kmeans.fit(X) 
# Predict the cluster for each data point 
y_kmeans = kmeans.predict(X) 
# Plot the results 
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis') 
centers = kmeans.cluster_centers_ 
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75) 
plt.xlabel("Feature 1") 
plt.ylabel("Feature 2") 
plt.title("K-Means Clustering") 
plt.show() 

```

# termwork 1
```
 jps
cd Documents/
sudo mkdir termwork1
cd termwork1/
sudo nano word.txt
Deer Bear River
car  car  River
Deer car  Bear
cat word.txt
 ls
sudo nano mapper.py
#!/usr/bin/env python

import sys

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    
    for word in words:
        print('%s\t%s' % (word, 1))
 cat mapper.py
ls
cat word.txt | python mapper.py
sudo nano reducer.py
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
            print('%s\t%s' % (current_word, current_count))
        current_count = count
        current_word = word

if current_word == word:
    print('%s\t%s' % (current_word, current_count))
 cat reducer.py
ls
cat word.txt | python mapper.py | sort –k1,1 | python reducer.py
jps
start-all.sh
jps
hdfs dfs -mkdir  /tw1
hdfs dfs -ls /
hdfs dfs –copyFromLocal/home/hduser/Documents/termwork1/word.txt /tw1
hdfs dfs  -ls  /tw1
sudo chmod 777 mapper.py reducer.py
ls -1
hadoop jar /home/hduser/Downloads/hadoop-streaming 2.7.3.jar \
-input /tw1/word.txt \
-output /tw1/8899 \
-mapper /home/hduser/Documents/termwork1/mapper.py \
-reducer /home/hduser/Documents/termwork1/reducer.py
hdfs dfs –cat /tw1/8899/part-00000
localhost:50070/







