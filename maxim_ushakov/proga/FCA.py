class FCA:
    def __init__(self, train, test, label):

        self.test = test.copy()
        self.label = label
        self.category = {}

        self.obj_dict = train.copy()
        self.obj_names = list(self.obj_dict.keys())
        self.attr_dict = {}
        for obj, attr_set in self.obj_dict.items():
            for attr in attr_set:
                try:
                    self.attr_dict[attr].add(obj)
                except:
                    self.attr_dict[attr] = {obj}
        self.attr_names = list(self.attr_dict.keys())
        print(len(self.obj_names), len(self.attr_names))

        self.separate_data_into_categories()

    def separate_data_into_categories(self):
        self.category = {}

        for obj, l in self.label.items():
            try:
                self.category[l].add(obj)
            except:
                self.category[l] = {obj}

    def calculate_hypothesis(self, label_index, max_num_contradictions=0, max_part_contradictions=0, min_support=1):
        hypothesis_extent = []
        hypothesis_intent = []
        num_concepts = 1

        attr_set = set(self.attr_names)
        obj_set = self.category[label_index].copy()

        for obj in obj_set:
            attr_set &= self.obj_dict[obj]

        if self.check_hypothesis(attr_set, obj_set, label_index, max_num_contradictions, max_part_contradictions, min_support):
            hypothesis_extent.append(obj_set.copy())
            hypothesis_intent.append(attr_set.copy())

        while len(attr_set) < len(self.attr_names):

            for ind_attr in range(len(self.attr_names)):
                if self.attr_names[ind_attr] in attr_set:
                    attr_set.discard(self.attr_names[ind_attr])
                else:
                    attr_set.add(self.attr_names[ind_attr])
                    attr_set_cl = set(self.attr_names)
                    obj_set = self.category[label_index].copy()

                    for attr in attr_set:
                        obj_set &= self.attr_dict[attr]

                    for obj in obj_set:
                        attr_set_cl &= self.obj_dict[obj]

                    flag = True
                    for attr in attr_set_cl - attr_set:
                        if self.attr_names.index(attr) > ind_attr:
                            attr_set.discard(self.attr_names[ind_attr])
                            flag = False
                            break

                    if flag:
                        attr_set = attr_set_cl.copy()
                        num_concepts += 1
                        if self.check_hypothesis(attr_set, obj_set, label_index, max_num_contradictions, max_part_contradictions, min_support):
                            hypothesis_extent.append(obj_set.copy())
                            hypothesis_intent.append(attr_set.copy())
                        break

        print(num_concepts)
        return hypothesis_extent, hypothesis_intent

    def check_hypothesis(self, attr_set, obj_set, label_index, max_num_contradictions, max_part_contradictions, min_support):
        if len(obj_set) < min_support:
            return False

        max_num_contradictions = max(max_num_contradictions, max_part_contradictions*len(obj_set))
        num_contradictions = 0
        for label, obj_set in self.category.items():
            if label == label_index:
                continue

            for obj in obj_set:
                if attr_set <= self.obj_dict[obj]:
                    num_contradictions += 1
                    if num_contradictions > max_num_contradictions:
                        return False

        return True

    def classify_test_set1(self, max_num_contradictions=0, min_support=1, threshold=0.51):
        extents = {}
        intents = {}
        label_test = {}
        num_empty_objects = 0

        for label, obj_set in self.category.items():
            print(label)
            extents[label], intents[label] = self.calculate_hypothesis(label,max_num_contradictions, 0, min_support)
            print(len(intents))

        for obj, attr_set in self.test.items():
            num_hypothesis = {}
            total_num_hypothesis = 0
            for label in self.category.keys():
                num_hypothesis[label] = 0
                for intent in intents[label]:
                    if intent <= attr_set:
                        num_hypothesis[label] += 1
                total_num_hypothesis += num_hypothesis[label]

            if total_num_hypothesis == 0:
                num_empty_objects += 1

            label_test[obj] = 'contradictory'
            for label in self.category.keys():
                if num_hypothesis[label] >= threshold*total_num_hypothesis and num_hypothesis[label] > 0:
                    label_test[obj] = label
                    break

        print(num_empty_objects)
        return label_test

    def classify_test_set2(self, max_part_contradictions=0, min_part_support=0.001, threshold=0):
        extents = {}
        intents = {}
        label_test = {}
        num_empty_objects = 0
        num_empty_h = 0

        for label, obj_set in self.category.items():
            print(label)
            extents[label], intents[label] = self.calculate_hypothesis(label, 0, max_part_contradictions, int(min_part_support*len(obj_set)))
            print(len(intents))

        for obj, attr_set in self.test.items():
            num_objects = {}
            total_num_objects = 0

            total_num_hypothesis = 0
            for label in self.category.keys():
                num_objects[label] = set()
                for intent_id in range(len(intents[label])):
                    if intents[label][intent_id] <= attr_set:
                        num_objects[label] |= extents[label][intent_id]
                        #total_num_hypothesis += 1
                        if len(extents[label][intent_id]) == 0:
                            print('SOS, SOS, error')
                total_num_objects += len(num_objects[label])

            label_test[obj] = 'contradictory'
            if total_num_hypothesis == 0:
                num_empty_h += 1

            if total_num_objects == 0:
                num_empty_objects += 1
                continue

            max_value_of_index = 0
            max_label = ''
            for label in self.category.keys():
                if len(num_objects[label])*len(self.obj_names) / (total_num_objects*len(self.category[label])) - 1 >= max_value_of_index:
                    max_value_of_index = len(num_objects[label])*len(self.obj_names) / (total_num_objects*len(self.category[label])) - 1
                    max_label = label

            if max_value_of_index > threshold:
                label_test[obj] = max_label

        print(num_empty_objects)
        #print(num_empty_h)
        return label_test