import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

df = pd.read_csv("house-votes-84.data", sep=',', header=None)

# Analyze data
print("Shape = {}".format(df.shape))
print(df.describe())

for col in df.columns:
    df[col] = df[col].replace("?", df[col].mode().iloc[0])

print(df[df.values != "?"])

X = df.drop(0, axis=1)
y = df[0]

for col in df.columns:
    print(df[col].unique())

LE = LabelEncoder()
y = LE.fit_transform(y)

for col in X.columns:
    X[col] = LE.fit_transform(X[col])

f = open("Output_tree.txt", 'w')


def decision_tree_classifier(X_, y_, test_size, random_state_arr):
    accuracy_arr = []
    tree_nodes = []
    for random_s in random_state_arr:
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=test_size, random_state=random_s)
        DTC = DecisionTreeClassifier(random_state=random_s)
        DTC.fit(X_train, y_train)
        y_pred = DTC.predict(X_test)
        accurracy = accuracy_score(y_test, y_pred)
        accuracy_arr.append(accurracy)
        tree_nodes.append(DTC.tree_.node_count)   # internal nodes + leaves

    max_accuracy = max(accuracy_arr)
    min_accuracy = min(accuracy_arr)
    average_accuracy = sum(accuracy_arr) / len(accuracy_arr)
    max_node = max(tree_nodes)
    min_node = min(tree_nodes)
    average_node = sum(tree_nodes) / len(tree_nodes)
    acc_arr = [round(max_accuracy * 100, 2), round(min_accuracy * 100, 2), round(average_accuracy * 100, 2)]
    node_arr = [max_node, min_node, average_node]
    f.write("Train size = {} : \n".format(
        100 - (test_size * 100)))
    f.write("   max_accuracy = {} \n".format(round(max_accuracy * 100, 2)))
    f.write("   min_accuracy = {} \n".format(round(min_accuracy * 100, 2)))
    f.write("   average_accuracy = {} \n".format(round(average_accuracy * 100, 2)))
    f.write("   max nodes = {} \n".format(max_node))
    f.write("   min nodes = {} \n".format(min_node))
    f.write("   average nodes = {} \n \n".format(average_node))
    return ("accuracy", acc_arr), ("Tree Size", node_arr)


d = {}
random_states = []

for i in range(5):
    rand_num = random.randint(0, 100)
    if rand_num not in random_states:
        random_states.append(rand_num)

test_random = [(0.5, random_states), (0.4, random_states), (0.3, random_states), (0.2, random_states)]

for tr in test_random:
    d["{}".format(100 - (tr[0] * 100)) + ' %' + ' train'] = decision_tree_classifier(X, y, tr[0], tr[1])

x = []
y_acc = []
for i in d.keys():
    x.append(i)   # train size
    y_acc.append(d[i][0][1][2])   # average accuracy for each train size

for i in range(len(x)):
    x[i] = float(x[i].split()[0])

plt.plot(x, y_acc, color='orange', marker='o')
plt.xlabel("Train size")
plt.ylabel("Average accuracy")
plt.savefig("Average accuracy of each train size.png")
plt.show()

f.write("From the accuracy graph: \n")
f.write(
    "When the amount of training data is small, an ML model does not generalise well. This is usually because the model is overfitting, \n"
    "In general, model overfitting is reduced as the training dataset size increases, More data will almost always increase the accuracy of a model, \n"
    "However, that does not necessarily mean that spending resources to increase the training dataset size is the best way to affect the model’s predictive performance. \n \n")

y_nodes = []
for i in d.keys():
    y_nodes.append(d[i][1][1][2])

plt.plot(x, y_nodes, color='orange', marker='o')
plt.xlabel("Train size")
plt.ylabel("Average nodes of tree size")
plt.savefig("Average nodes of tree size of each train size.png")
plt.show()

f.write("From the graph of average nodes of tree size: \n")
f.write(
    "Decision trees are very interpretable – as long as they are short. The number of terminal nodes increases quickly \n")

f.close()
