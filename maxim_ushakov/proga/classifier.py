def evaluate_implications(train_data, labels):
    implications = {}
    index_implication = 0
    extents = []
    attr_extents = {}

    for obj_key, attributes in train_data.items():
        for attr in attributes:
            try:
                attr_extents[attr].add(obj_key)
            except:
                attr_extents[attr] = set([obj_key])
    current_extents = [objects for attr_key, objects in attr_extents.items()]

    while len(current_extents) > 0:
        objects = current_extents.pop()

        temp_obj = objects.pop()
        objects.add(temp_obj)
        label = labels[temp_obj]
        flag_is_implication = True
        for obj in objects:
            if labels[obj] != label:
                flag_is_implication = False
                break

        if flag_is_implication:
            implications[index_implication] = (objects, label)
            index_implication += 1
        else:
            for objects2 in extents:
                new_objects = objects & objects2
                if len(new_objects) == 0:
                    continue

                flag_is_new_objects = True
                for objects3 in current_extents:
                    if new_objects == objects3:
                        flag_is_new_objects = False
                        break
                if not flag_is_new_objects:
                    continue
                for objects3 in extents:
                    if new_objects == objects3:
                        flag_is_new_objects = False
                        break
                if not flag_is_new_objects:
                    continue
                for temp_key, objects3 in implications.items():
                    if new_objects == objects3:
                        flag_is_new_objects = False
                        break
                if not flag_is_new_objects:
                    continue

                current_extents.append(new_objects)
            extents.append(objects)

    for index in range(index_implication):
        objects = implications[index][0]
        first_obj = objects.pop()
        attributes = train_data[first_obj]
        for obj in objects:
            attributes &= train_data[obj]
        implications[index] = (attributes, implications[index][1])

    return implications

#функция классификатора, обучается на train_data
#классифицирует test_data
#train_data - словарь, в котором ключи - объекты
#по данным ключам хранятся списки аттрибутов объекта
#test_data - словарь, устроен так же как и train_data
#labels - словарь, в которым ключи объекты,
#и по каждому ключу хранятся метки объектов (может включать и тестовую выборку)
#attributes - список возможных аттрибутов объектов с их значениями, например:
#['color:white','color:red','color:blue','shape:circle','country:Russia']
def classify_test_data(train_data, test_data, labels):

    for obj in train_data.keys():
        train_data[obj] = set([str(i)+':'+train_data[obj][i] for i in range(len(train_data[obj]))])

    for obj in test_data.keys():
        test_data[obj] = set([str(i)+':'+test_data[obj][i] for i in range(len(test_data[obj]))])

    implications = evaluate_implications(train_data, labels)
    print(len(implications))

    test_labels = {}
    for obj_key, attr_list in test_data.items():
        flag_is_classified = False
        test_labels[obj_key] = -1
        for ind_implication, implication in implications.items():
            if implication[0].issubset(attr_list):
                if flag_is_classified:
                    if test_labels[obj_key] != implication[1]:
                        test_labels[obj_key] = -1
                        break
                else:
                    flag_is_classified = True
                    test_labels[obj_key] = implication[1]

    return test_labels
