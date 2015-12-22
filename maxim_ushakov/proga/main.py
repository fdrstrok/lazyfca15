from FCA import FCA
import random


def read_data1():
    data_func1 = {}
    label_func1 = {}
    obj_index = 0
    temp_max = 0
    with open('data_sets\connect-4.data', 'r') as input_file:
        for row in input_file:
            obj_index += 1
            attr_list_str = row.strip('\n').split(',')
            label_func1[obj_index] = attr_list_str[-1]

            attr_indices = []
            attr_list = []
            attr_index = -1
            l_b = 6
            r_b = 0
            b_b = 5
            u_b = 0
            for attr in attr_list_str[0:-1]:
                attr_index += 1
                if attr != 'b':
                    attr_indices.append(attr_index)
                    attr_list.append(attr)
                    l_b = min(l_b, attr_index // 6)
                    r_b = max(r_b, attr_index // 6)
                    u_b = max(u_b, attr_index % 6)
                    b_b = min(b_b, attr_index % 6)

            #print(attr_indices)
            #print(l_b, r_b, u_b, b_b)
            u_b -= b_b
            r_b -= l_b

            if u_b > 3 or r_b > 3:
                continue

            temp_max = max(temp_max, (u_b+1)*(r_b+1))
            attr_indices = [(attr_ind // 6 - l_b)*(u_b+1)+(attr_ind % 6 - b_b) for attr_ind in attr_indices]
            """attr_list_full = ['b' for i in range((u_b+1)*(r_b+1))]
            for i in range(len(attr_indices)):
                #print((u_b+b_b+1)*(r_b+l_b+1), attr_indices[i]+l_b*6+b_b)
                attr_list_full[attr_indices[i]] = attr_list[i]"""

            data_func1[obj_index] = set([str(attr_indices[i])+':'+attr_list[i] for i in range(len(attr_list))])

    input_file.close()
    print(temp_max)

    return data_func1, label_func1

percent_train = 0.8
data, label = read_data1()
train_ind = set(random.sample(data.keys(), round(percent_train*len(data))))
test_ind = set(data.keys()) - train_ind

train = {}
train_label = {}
for obj in train_ind:
    train[obj] = data[obj]
    train_label[obj] = label[obj]

test = {}
test_label = {}
for obj in test_ind:
    test[obj] = data[obj]
    test_label[obj] = label[obj]

fca = FCA(train, test, train_label)
test_label_predict = fca.classify_test_set1()

accuracy = 0
for obj in test_label.keys():
    if test_label[obj] == test_label_predict[obj]:
        accuracy += 1

print('accuracy = ', accuracy/len(test_label))