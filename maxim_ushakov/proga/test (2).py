import classifier
import random


file_name = 'data_sets/cars/car.data'
file = open(file_name, 'r')
data = {}
labels = {}
key_obj = 0
for row in file:
    dict_list_str = row.strip('\n').split(',')
    labels[key_obj] = dict_list_str[-1]
    data[key_obj] = dict_list_str[0:-1]
    key_obj += 1
file.close()

train_portion = 0.8
train_ind = set(random.sample(range(len(data)), round(train_portion*len(data))))
test_ind = set(range(len(data))) - train_ind
train_data = {}
for ind in train_ind:
    train_data[ind] = data[ind]

test_data = {}
for ind in test_ind:
    test_data[ind] = data[ind]

file = open('data_sets/out_data', 'w')
for obj_key, attr_list in data.items():
    row = str(obj_key)+':'+','.join(attr_list)+'\n'
    file.writelines(row)
file.close()

"""file = open('data_sets/out_test', 'w')
for obj_key, attr_list in test_data.items():
    row = str(obj_key)+':'+','.join(attr_list)+'\n'
    file.writelines(row)
file.close()"""

test_label = classifier.classify_test_data(train_data, test_data, labels)

print(test_label)
