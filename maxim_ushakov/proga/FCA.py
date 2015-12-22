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

    def calculate_hypothesis(self, label_index, max_num_contradictions=0, min_support=1):
        hypothesis_extent = []
        hypothesis_intent = []

        attr_set = set(self.attr_names)
        obj_set = self.category[label_index].copy()

        for obj in obj_set:
            attr_set &= self.obj_dict[obj]

        if self.check_hypothesis(attr_set, obj_set, label_index, max_num_contradictions, min_support):
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
                        if self.check_hypothesis(attr_set, obj_set, label_index, max_num_contradictions, min_support):
                            hypothesis_extent.append(obj_set.copy())
                            hypothesis_intent.append(attr_set.copy())
                        break

        return hypothesis_extent, hypothesis_intent

    def check_hypothesis(self, attr_set, obj_set, label_index, max_num_contradictions, min_support):
        if len(obj_set) < min_support:
            return False

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

    def classify_test_set1(self, max_num_contradictions=0, min_support=1, part_aggregation=0.51):
        extents = {}
        intents = {}
        label_test = {}

        for label, obj_set in self.category.items():
            print(label)
            extents[label], intents[label] = self.calculate_hypothesis(label,max_num_contradictions, min_support)
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

            label_test[obj] = 'contradictory'
            for label in self.category.keys():
                if num_hypothesis[label] >= part_aggregation*total_num_hypothesis and num_hypothesis[label] > 0:
                    label_test[obj] = label
                    break

        return label_test