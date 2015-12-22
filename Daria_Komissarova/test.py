import classifier
import random


def read_titanic():
    input_file = open('titanic.data', 'r')
    global data
    global labels
    data = {}
    labels = {}
    key_obj = 0
    for row in input_file:
        str_list = row.strip('\n').split(',')
        labels[key_obj] = int(str_list[-1])
        data[key_obj] = [int(i) for i in str_list[0:-1]]
        key_obj += 1
    input_file.close()

    return


def read_tic_tac():
    input_file = open('tic-tac.data', 'r')
    global data
    global labels
    data = {}
    labels = {}
    key_obj = 0
    for row in input_file:
        str_list = row.strip('\n').split(',')
        labels[key_obj] = int(str_list[-1])
        data[key_obj] = []
        for symb in str_list[0:-1]:
            if symb == 'x':
                data[key_obj].append(1)
                data[key_obj].append(0)
            elif symb == 'o':
                data[key_obj].append(0)
                data[key_obj].append(1)
            else:
                print('some_error'+str(key_obj))
        key_obj += 1
    input_file.close()

    return


data = {}
labels = {}
read_tic_tac()

print(len(data), len(labels))

train_portion = 0.8
train_ind = set(random.sample(range(len(data)), round(train_portion*len(data))))
test_ind = set(range(len(data))) - train_ind
train_data = {}
train_labels = {}
for ind in train_ind:
    train_data[ind] = data[ind]
    train_labels[ind] = labels[ind]

test_data = {}
for ind in test_ind:
    test_data[ind] = data[ind]

pos, neg = classifier.classify_test_data(train_data, test_data, train_labels)

label_positive = 1
label_negative = 0

accuracy = 0
for key in pos:
    if labels[key] == label_positive:
        accuracy += 1

for key in neg:
    if labels[key] == label_negative:
        accuracy += 1

print(accuracy, len(test_data))
