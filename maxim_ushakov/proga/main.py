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


def read_data2():
    data_func2 = {}
    label_func2 = {}
    obj_index = 0
    with open('data_sets\car.data', 'r') as input_file:
        for row in input_file:
            obj_index += 1

            attr_list_str = row.strip('\n').split(',')
            label_func2[obj_index] = attr_list_str[-1]

            attr_set = set()

            if attr_list_str[0] == 'vhigh':
                attr_set |= {'0:m+', '0:h+', '0:v+'}
            elif attr_list_str[0] == 'high':
                attr_set |= {'0:m+', '0:h+', '0:h-'}
            elif attr_list_str[0] == 'med':
                attr_set |= {'0:m+', '0:m-', '0:h-'}
            elif attr_list_str[0] == 'low':
                attr_set |= {'0:l-', '0:m-', '0:h-'}
            else:
                print('error', attr_list_str[0])

            if attr_list_str[1] == 'vhigh':
                attr_set |= {'1:m+', '1:h+', '1:v+'}
            elif attr_list_str[1] == 'high':
                attr_set |= {'1:m+', '1:h+', '1:v-'}
            elif attr_list_str[1] == 'med':
                attr_set |= {'1:m+', '1:h-', '1:v-'}
            elif attr_list_str[1] == 'low':
                attr_set |= {'1:m-', '1:h-', '1:v-'}
            else:
                print('error', attr_list_str[1])

            if attr_list_str[3] == '2':
                attr_set |= {'3:2-', '3:4-'}
            elif attr_list_str[3] == '4':
                attr_set |= {'3:4+', '3:4-'}
            elif attr_list_str[3] == 'more':
                attr_set |= {'3:4+', '3:more+'}
            else:
                print('error', attr_list_str[3])

            if attr_list_str[4] == 'big':
                attr_set |= {'4:m+', '4:b+'}
            elif attr_list_str[4] == 'med':
                attr_set |= {'4:m+', '4:m-'}
            elif attr_list_str[4] == 'small':
                attr_set |= {'4:m-', '4:s-'}
            else:
                print('error', attr_list_str[4])

            if attr_list_str[5] == 'high':
                attr_set |= {'5:m+', '5:h+'}
            elif attr_list_str[5] == 'med':
                attr_set |= {'5:m+', '5:m-'}
            elif attr_list_str[5] == 'low':
                attr_set |= {'5:m-', '5:l-'}
            else:
                print('error', attr_list_str[5])

            data_func2[obj_index] = attr_set.copy()

    input_file.close()

    return data_func2, label_func2


percent_train = 0.8
data, label = read_data2()
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
test_label_predict = fca.classify_test_set2()

accuracy = 0
num_classified = 0
for obj in test_label.keys():
    if test_label[obj] == test_label_predict[obj]:
        accuracy += 1
    if test_label_predict[obj] != 'contradictory':
        num_classified += 1

print('accuracy = ', accuracy, num_classified, len(test_label))