from __future__ import print_function
import random


class FCAData(object):
    dataset = []  # list of objects read from dataset file
    bin_data = []  # binarized data
    attrs = []  # attributes filter
    attr_inds = []  # attributes indexes for `attrs`
    attr_vals = {'cat': {}, 'num': {}, 'int': {}}

    # target attributes
    tgt = ['G1', 'G2', 'G3']

    # categorical attributes
    cat = ['sex', 'address', 'famsize']

    # numerical attributes
    num = ['Medu', 'Fedu', 'failures']

    # interval numerical attributes
    int = {'absences': [[0, 1], [1, 2], [2, 3], [3, 6], [6, 999]]}

    def __init__(self, datasetfname, k=4, score_threshold=0.5, binarize=True,
                 randomise=False, split=True, _cat=None, _num=None, _int=None):
        """
        Initializing data, reading dataset and binarization.
        """
        if _cat:
            self.cat = _cat
        if _num:
            self.num = _num
        if _int:
            self.int = _int
        self.randomise = randomise
        self.score_threshold = score_threshold
        self.fname = datasetfname
        self.k = int(k) if k >= 2 else 2
        attrs = list(set(self.cat) | set(self.num) | set(self.tgt) |
                     set(self.int.keys()))
        self._read_data(attrs)
        if not self.dataset:
            raise ValueError('No data in file "{}".'.format(self.fname))
        self._get_attr_vals()
        if binarize:
            self._binarize()
        else:
            self._nonbinarized()
        if split:
            self._split_data()
        else:
            self._write_whole_data()
        print("data prepared")

    def _write_whole_data(self):
        """
        Write whole data to file.
        """
        lines = ['{}\n'.format(','.join(d)) for d in self.bin_data]
        with open('prepared_data/dataset.csv', 'w') as fout:
            fout.writelines(lines)

    def _k_fold(self):
        """
        Generates K (training, validation) pairs from the items in X.

        Each pair is a partition of X, where validation is an iterable of
        length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

        If randomise is true, a copy of X is shuffled before partitioning,
        otherwise its order is preserved in training and validation.
        """
        header = '{}\n'.format(','.join(self.bin_data[0]))
        l = self.bin_data[1:]
        if self.randomise:
            random.shuffle(l)
        for j in range(self.k):
            training = [header]
            validation = [header]
            training += ['{}\n'.format(','.join(x))
                        for i, x in enumerate(l) if i % self.k != j]
            validation += ['{}\n'.format(','.join(x))
                          for i, x in enumerate(l) if i % self.k == j]
            yield training, validation, j + 1

    def _split_data(self):
        """
        Splitting data in k pairs of train, test files for classifier.
        """
        for training, validation, k in self._k_fold():
            trainfname = "prepared_data/train{}.csv".format(k)
            with open(trainfname, 'w') as fout:
                fout.writelines(training)
            testfname = "prepared_data/test{}.csv".format(k)
            with open(testfname, 'w') as fout:
                fout.writelines(validation)

    def _get_bin_header_and_pos(self):
        """
        Get binarized data headers:
            cat = cat, cat.vals = ['a','b','c'] =>
                => result_cat = ['cat:a', 'cat:b', 'cat:c'],
        and headers positions.
        """
        bin_attr_headers = []
        bin_attr_pos = {a: {} for i, a in self.attrs}
        cnt = 0
        for i, a in self.attrs:
            if a in self.cat:
                for av in self.attr_vals['cat'][a]:
                    bin_attr_pos[a].update({av: cnt})
                    bin_attr_headers.append('{}-{}'.format(a, av))
                    cnt += 1
            elif a in self.num:
                for av in self.attr_vals['num'][a]:
                    bin_attr_pos[a].update({av: cnt})
                    bin_attr_headers.append('{}-{}'.format(a, av))
                    cnt += 1
            elif a in self.int:
                for intr in self.int[a]:
                    bin_attr_pos[a].update({intr[0]: cnt})
                    bin_attr_headers.append('{}_{}-{}'.format(
                        a, intr[0], intr[1]))
        bin_attr_headers.append('class')
        return bin_attr_headers, bin_attr_pos

    def _bin_cat(self, attr_name, binarized, bin_attr_pos, v, cross_label='x'):
        """
        Binarize category feature by rule:
            cat = cat, cat.vals = ['a','b','c'] =>
                => result_cat = ['cat:a', 'cat:b', 'cat:c'].
            o.cat = 'a' => o.cat:a = 'x', o.cat:b = 'o', o.cat:c = 'o'.
        """
        cat_vals = self.attr_vals['cat'][attr_name]
        for cv in cat_vals:
            if v == cv:
                binarized[bin_attr_pos[attr_name][cv]] = cross_label

    def _bin_num(self, attr_name, binarized, bin_attr_pos, v, cross_label='x'):
        """
        Binarize numerical feature by rule:
            for all a (a < b and b = 'x' => a = 'x').
        """
        num_vals = self.attr_vals['num'][attr_name]
        for nv in num_vals:
            if int(v) == int(nv):
                binarized[bin_attr_pos[attr_name][int(v)]] = cross_label
            for m in range(1, int(v) + 1):
                if m in bin_attr_pos[attr_name]:
                    binarized[bin_attr_pos[attr_name][m]] = cross_label

    def _bin_int(self, attr_name, binarized, bin_attr_pos, v, cross_label='x'):
        """
        Binarize numerical feature by intervals.
        """
        for intr in self.int[attr_name]:
            if intr[0] <= float(v) < intr[1]:
                binarized[bin_attr_pos[attr_name][intr[0]]] = cross_label

    def _nonbinarized(self):
        """
        Replace target by `class` attribute.
        """
        self.bin_data.append([a for i, a in self.attrs[:-3]])
        self.bin_data[0].append('class')
        for d in self.dataset:
            data = d[:-2]
            avg_score = sum([int(s) for s in d[-3:]]) / 60.
            # setting 'positive'|'negative' class
            # depending on student average math score
            data[-1] = 'positive' \
                if avg_score > self.score_threshold else 'negative'
            self.bin_data.append(data)

    def _binarize(self):
        """
        Binarization of data read from dataset. Resolving category features,
        numerical features and class for each object.
        """
        bin_attr_headers, bin_attr_pos = self._get_bin_header_and_pos()
        n = len(bin_attr_headers)
        self.bin_data.append(bin_attr_headers)

        for d in self.dataset:
            binarized = ['0' for i in range(n)]
            for i, v in enumerate(d):
                attr_name = self.attrs[i][1]
                if attr_name in self.cat:
                    self._bin_cat(attr_name, binarized, bin_attr_pos, v, '1')
                elif attr_name in self.num:
                    self._bin_num(attr_name, binarized, bin_attr_pos, v, '1')
                elif attr_name in self.int:
                    self._bin_int(attr_name, binarized, bin_attr_pos, v, '1')
            avg_score = sum([int(s) for s in d[-3:]]) / 60.

            # setting 'positive'|'negative' class
            # depending on student average math score
            binarized[-1] = 'positive' \
                if avg_score > self.score_threshold else 'negative'
            self.bin_data.append(binarized)

    def _get_attr_vals(self):
        """
        Get attribute values from dataset.
        """
        attr_vals = {
            'cat': {c: [] for c in self.cat},
            'num': {c: [] for c in self.num},
            'int': {c: [] for c in self.int},
        }
        for d in self.dataset:
            for i, v in enumerate(d):
                attr_name = self.attrs[i][1]
                if attr_name in self.cat:
                    attr_vals['cat'][attr_name].append(v)
                elif attr_name in self.num:
                    if 'U' == v:
                        raise ValueError(attr_name, self.num, attr_name in self.num)
                    attr_vals['num'][attr_name].append(int(v))
                elif attr_name in self.int:
                    attr_vals['int'][attr_name].append(int(v))
        for k in attr_vals:
            for c in attr_vals[k]:
                attr_vals[k][c] = list(set(attr_vals[k][c]))
        self.attr_vals = attr_vals

    def _read_data(self, attrs):
        """
        Reading data from .csv file, where first line is headers (attributes),
        and all next lines are objects.
        """
        with open(self.fname, 'r') as fin:
            line = fin.readline()
            if not line:
                raise ValueError('File "{}" is empty.'.format(self.fname))
            self.attrs = self._filter_requested_attrs(line, attrs)
            self.attr_inds = [a[0] for a in self.attrs]
            lines = fin.readlines()
            for line in lines:
                self.dataset.append(self._get_obj_from_file(line))

    def _get_obj_from_file(self, line):
        """
        Get object as list of attribute values.
        """
        data = line.strip().replace('"', '').split(';')
        return [v for i, v in enumerate(data) if i in self.attr_inds]

    def _filter_requested_attrs(self, line, attrs):
        """
        Get list of attributes requested by user if they are present in dataset.
        """
        fattrs = self._get_attrs_from_file(line)
        return [a for a in fattrs if a[1] in attrs]

    @staticmethod
    def _get_attrs_from_file(line):
        """
        Get attributes from file header as list of tuples (pos, title).
        """
        attrs = line.strip().split(';')
        return [(i, a) for i, a in enumerate(attrs)]


class LazyFCA(object):
    attrib_names = []
    data = []
    plus = []
    minus = []
    plus_intent = []
    minus_intent = []
    unknown = []
    RESTPL = {'TP': 0., 'FP': 0., 'TN': 0., 'FN': 0.}

    def __init__(self, k=1, path_to_data='prepared_data/', threshold=1.,
                 show_progress=False, binarized=True):
        """
        Initializing data, running k tests on k pairs of train, test files and
        printing total results of tests.
        """
        self.binarized = binarized
        self.threshold = threshold
        self.show_progress = show_progress
        self.min_agg = 99
        self.max_agg = 0

        self.k = int(k) if k >= 1 else 1
        self.path_to_data = path_to_data
        total = self.RESTPL.copy()
        if self.show_progress:
            print('starting data analysis')
        self.avg_prec, self.avg_rec, self.avg_acc, self.f1 = 0, 0, 0, 0
        for i in range(1, self.k + 1):
            if self.show_progress:
                print('running test {}'.format(i))
            self.read_data(i)
            res, prec, rec, acc, f1 = self.test_data()
            self.avg_prec += prec
            self.avg_rec += rec
            self.avg_acc += acc
            self.f1 += f1
            for k in total:
                total[k] += res[k]
        if self.show_progress:
            print('analysis total results: ')
            for k, v in total.items():
                print('{}: {:.3f}'.format(k, v / self.k))
        if self.show_progress:
            print('min_agg: {}\nmax_agg: {}'.format(self.min_agg, self.max_agg))
        self.print_tests_avg_scores()

    def print_tests_avg_scores(self):
        self.avg_prec /= float(self.k)
        self.avg_rec /= float(self.k)
        self.avg_acc /= float(self.k)
        self.f1 /= float(self.k)
        print('avg precision: {:.3f}\n'
              'avg recall:    {:.3f}\n'
              'avg accuracy:  {:.3f}\n'
              'avg F1 score:  {:.3f}'.format(
                self.avg_prec, self.avg_rec, self.avg_acc, self.f1))

    def make_intent(self, example):
        """
        Make intent from object:
            set of 'attr_name:attr_val' values.
        """
        return set(
            [i + ':' + str(k) for i, k in zip(self.attrib_names, example)])

    def intersect_ctx(self, eintent, ctx_intent):
        """
        Intersect intent of given example with given context (plus, minus) and
        count ratio of non empty intersections to cardinal of context,
        ratio of attributes in intersections to cardinal of attributes and
        cardinal of context.
        """
        cnt = 0
        attr_cnt = 0
        eintent.discard('class:positive')
        eintent.discard('class:negative')
        for ci in ctx_intent:
            candidate = eintent & ci
            if self.binarized:
                candidate = set([c for c in candidate if ':0' not in c])
            if not candidate:
                continue
            cnt += 1
            attr_cnt += len(candidate)
        cnt /= float(len(ctx_intent))
        attr_cnt /= float(len(self.attrib_names) - 1) * float(len(ctx_intent))
        return cnt, attr_cnt

    def intersect_pos_ctx(self, eintent):
        """
        Intersect given example with positive objects.
        """
        return self.intersect_ctx(eintent, self.plus_intent)

    def intersect_neg_ctx(self, eintent):
        """
        Intersect given example with negative objects.
        """
        return self.intersect_ctx(eintent, self.minus_intent)

    @staticmethod
    def aggregation(pos, neg, pos_attrs, neg_attrs):
        """
        Aggregation function:
                   positive_ratio + 1   positive_attributes_ratio + 1
            aggr = ------------------ * -----------------------------
                   negative_ratio + 1   negative_attributes_ratio + 1
        """
        return (pos + 1) / (neg + 1) * (pos_attrs + 1) / (neg_attrs + 1)

    def aggr2(self, pos, neg, pos_attrs, neg_attrs):
        return abs((pos_attrs + 1) / float(pos + 1) -
                   (neg_attrs + 1) / float(neg + 1))  # Good thresh=1.
        # return (pos / float(len(self.plus_intent))) / \
        #        (neg / float(len(self.minus_intent))) * \
        #        (pos_attrs - neg_attrs) / float(pos + neg)

    def check_hypothesis(self, example):
        """
        Check classifier hypothesis for the given example.
        """
        eintent = self.make_intent(example)
        pos, pos_attrs = self.intersect_pos_ctx(eintent)
        neg, neg_attrs = self.intersect_neg_ctx(eintent)
        agg = self.aggregation(pos, neg, pos_attrs, neg_attrs)
        if agg < self.min_agg:
            self.min_agg = agg
        if agg > self.max_agg:
            self.max_agg = agg
        return agg >= self.threshold

    @staticmethod
    def get_hypothesis_results(u, res, result):
        if res:
            if u[-1] == 'positive':
                result['TP'] += 1
            else:
                result['FP'] += 1
        else:
            if u[-1] == 'negative':
                result['TN'] += 1
            else:
                result['FN'] += 1

    @staticmethod
    def get_scores(result):
        accuracy = (result['TP'] + result['TN']) / \
                   max(1, result['TP'] + result['TN'] +
                       result['FP'] + result['FN'])
        precision = result['TP'] / max(1, result['TP'] + result['FP'])
        recall = result['TP'] / max(1, result['TP'] + result['FN'])
        f1 = 2 * precision * recall / max(1, precision + recall)
        return accuracy, precision, recall, f1

    @staticmethod
    def get_avg_result(result):
        all_pos = result['TP'] + result['FP']
        all_neg = result['TN'] + result['FN']

        result['TP'] /= all_pos if all_pos else 1
        result['FP'] /= all_pos if all_pos else 1
        result['TN'] /= all_neg if all_neg else 1
        result['FN'] /= all_neg if all_neg else 1

    def test_data(self):
        """
        Test hypothesis for data from train, test pair.
        """
        result = self.RESTPL.copy()
        self.plus_intent = [self.make_intent(e) for e in self.plus]
        self.minus_intent = [self.make_intent(e) for e in self.minus]
        for i, u in enumerate(self.unknown):
            if self.show_progress:
                print(
                    '\rchecking {}/{}'.format(i + 1, len(self.unknown)), end='')
            res = self.check_hypothesis(u)
            self.get_hypothesis_results(u, res, result)
        if self.show_progress:
            print('')

        accuracy, precision, recall, f1 = self.get_scores(result)
        self.get_avg_result(result)

        return result, precision, recall, accuracy, f1

    def read_data(self, index):
        """
        Read data from train, test pair and return positive, negative and
        testing objects.
        """
        trainfname = "{}train{}.csv".format(self.path_to_data, index)
        with open(trainfname, "r") as q:
            if not self.attrib_names:
                header = q.readline()
                self.attrib_names = header.strip().split(',')
            train = [a.strip().split(",") for a in q]
            plus = [a for a in train if a[-1] == "positive"]
            minus = [a for a in train if a[-1] == "negative"]

        testfname = "{}test{}.csv".format(self.path_to_data, index)
        with open(testfname, "r") as w:
            unknown = [a.strip().split(",") for a in w]
        unknown.pop(0)
        self.plus, self.minus, self.unknown = plus, minus, unknown


def experiment(cat, threshold, binarize=True, randomise=False,
               show_progress=False, target_threshold=0.53):
    print('\tthreshold: {}; {}'.format(
        threshold,
        'binarized' if binarize else 'non binarized'))
    FCAData('dataset/student-mat.csv', score_threshold=target_threshold,
            randomise=randomise, binarize=binarize, _cat=cat)
    LazyFCA(k=4, path_to_data='prepared_data/', threshold=threshold,
            show_progress=show_progress)


def global_test(num):
    if num <= 4:
        print('experiments with all attributes')
        cat = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Mjob',
               'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
               'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
               'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
               'Walc', 'health']
        if num == 1:
            experiment(cat, 1.015, binarize=False)
        elif num == 2:
            experiment(cat, 1.005, binarize=False)
        elif num == 3:
            experiment(cat, 1.015, binarize=True)
        elif num == 4:
            experiment(cat, 1.005, binarize=True)
    elif 4 < num:
        print('experiments with filtered attributes')
        cat = ['sex', 'address', 'famsize']
        if num == 5:
            experiment(cat, 1.015, binarize=False)
        elif num == 6:
            experiment(cat, 1.005, binarize=False)
        elif num == 7:
            experiment(cat, 1.015, binarize=True)
        elif num == 8:
            experiment(cat, 1.005, binarize=True)
        elif num == 9:
            # the same as 7 but with different students avg score thresh
            experiment(cat, 1.025, binarize=True, target_threshold=0.8)
        elif num == 10:
            experiment(cat, 1.005, binarize=True, target_threshold=0.4)


global_test(7)
