import pandas as pd
import numpy as np

from os import path

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/tic-tac/'


def prepare_data(data, t='tic-tac'):
    if t == 'tic-tac':
        attrib_names = [
            'top-left-square',
            'top-middle-square',
            'top-right-square',
            'middle-left-square',
            'middle-middle-square',
            'middle-right-square',
            'bottom-left-square',
            'bottom-middle-square',
            'bottom-right-square',
            'class'
        ]

        y = data.iloc[:,-1] == 'positive'
        for i, col in enumerate(data.columns):
            data[col] = attrib_names[i] + ':' + data[col].astype(str)

        data = data.drop(columns=data.columns[-1])

        return np.array([set(r) for r in data.values]), np.array(y)
    elif t == 'titanic':
        attrib_names = [
            'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Family', 'IsAlone'
        ]

        y = data['Survived'] == 1

        label = LabelEncoder()

        data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        title_names = (data['Title'].value_counts() < 10)
        data['Title'] = data['Title'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)

        data['Embarked'] = data['Embarked'].fillna('S')
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

        data['Family'] = data['SibSp'] + data['Parch']
        data['IsAlone'] = (data['Family'] == 0).astype(int)

        data['FareBin'] = pd.qcut(data['Fare'], 4)
        data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
        data['FamilyBin'] = pd.cut(data['Family'].astype(int), 5)

        data['Sex'] = label.fit_transform(data['Sex'])
        data['Embarked'] = label.fit_transform(data['Embarked'])
        data['Sex'] = label.fit_transform(data['Sex'])
        data['Age'] = label.fit_transform(data['AgeBin'])
        data['Fare'] = label.fit_transform(data['FareBin'])
        data['Family'] = label.fit_transform(data['FamilyBin'])
        data['Title'] = label.fit_transform(data['Title'])

        data = data.drop(columns=set(data.columns) - set(attrib_names))

        for i, col in enumerate(data.columns):
            data[col] = attrib_names[i] + ':' + data[col].astype(str)

        return np.array([set(r) for r in data.values]), np.array(y)

    else:
        raise Exception('Unsupported type ' + t)


def generators_sup(positive, negative, sample, min_sup=0.0, eps=0.0):
    """
    If there is an intersection in positve class element
    and sample then calculate support for this intersection.
    If support is bigger than min_sup then classify as positive.
    The same applies to negative class. If there is no intersection
    with needed support, choose the class with the biggest support
    """
    pos = True
    biggest_sup = -100

    for p in positive:
        inter = sample & p
        if len(inter) > 0:
            sup = len([pe for pe in positive if pe.issuperset(inter)]) / len(positive)

            if len([n for n in negative if n.issuperset(inter)]) / len(negative) <= eps:
                if sup > biggest_sup:
                    biggest_sup = sup
                    pos = True

                if sup > min_sup:
                    return True

    for n in negative:
        inter = sample & n

        sup = len([ne for ne in negative if ne.issuperset(inter)]) / len(negative)

        if len([p for p in positive if p.issuperset(inter)]) / len(positive) <= eps:
            if sup > biggest_sup:
                biggest_sup = sup
                pos = False

            if sup > min_sup:
                return False

    return pos


def generators_card(positive, negative, sample, min_card=0.0, eps=0.0):
    """
    If there is an intersection in positve class element
    and sample and this intersection is not included in
    an element of negative class then this positive
    element votes for this sample. After that calculate
    proportion of votes in positive and negative classes
    and make decision.
    """
    pos = 0
    neg = 0

    for p in positive:
        inter = sample & p
        if float(len(inter)) / len(sample) > min_card:
            if len([n for n in negative if n.issuperset(inter)]) / len(negative) <= eps:
                pos += 1

    for n in negative:
        inter = sample & n
        if float(len(inter)) / len(sample) > min_card:
            if len([p for p in positive if p.issuperset(inter)]) / len(positive) <= eps:
                neg += 1

    pos_score = float(pos) / len(positive)
    neg_score = float(neg) / len(negative)

    if pos_score >= neg_score:
        return True
    else:
        return False


def calculate_metrics(test_y, predicted_y):
    TP = np.sum(test_y & predicted_y)
    TN = np.sum(~(test_y | predicted_y))
    FP = np.sum(~test_y & predicted_y)
    FN = np.sum(test_y & ~predicted_y)
    TPR = float(TP) / np.sum(test_y)
    TNR = float(TN) / np.sum(~test_y)
    FPR = float(FP) / (TP + FN)
    NPV = float(TN) / (TN + FN)
    FDR = float(FP) / (TP + FP)
    acc = accuracy_score(test_y, predicted_y)
    prec = precision_score(test_y, predicted_y)
    rec = recall_score(test_y, predicted_y)

    return [TP, TN, FP, FN, TPR, TNR, FPR, NPV, FDR, acc, prec, rec]


def average_metrics(metrics_arr):
    res = []
    for metrics in zip(*metrics_arr):
        res.append(sum(metrics) / len(metrics))

    return res


def print_results(metrics):
    print("""True Positive: {}\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}
    \nTrue Positive Rate: {}\nTrue Negative Rate: {}\nNegative Predictive Value: {}
    \nFalse Positive Rate: {}\nFalse Discovery Rate: {}\nAccuracy: {}\nRecall: {}""".format(
        *[round(m, 4) for m in metrics]))


def one_sample():
    train = pd.read_csv('data/tic-tac/train1.csv')
    test = pd.read_csv('data/tic-tac/test1.csv')

    train, train_y = prepare_data(train)
    test, test_y = prepare_data(test)

    positive = train[train_y]
    negative = train[~train_y]

    predicted_y = np.array([generators_card(positive, negative, s) for s in test])
    metrics = calculate_metrics(test_y, predicted_y)
    print_results(metrics)


def cross_validation(algorithm, data_path, t='titanic', k=11):
    results = []
    for i in range(0, k):
        train = pd.read_csv(path.join(data_path, 'train{}.csv'.format(i)))
        test = pd.read_csv(path.join(data_path, 'test{}.csv'.format(i)))

        train, train_y = prepare_data(train, t)
        test, test_y = prepare_data(test, t)

        positive = train[train_y]
        negative = train[~train_y]

        predicted_y = np.array([algorithm(positive, negative, s) for s in test])
        metrics = calculate_metrics(test_y, predicted_y)
        results.append(metrics)

    averaged = average_metrics(results)
    # accuracy
    return averaged[-3]


def test_card(data_path, t='tic-tac', k=11, frac=10):
    print('card    eps     acc')

    best_card, best_eps, best_acc = -1, -1, -1
    for c in range(1, 10):
        card = c / frac
        for e in range(0, 100, 5):
            eps = e / 100

            def f(positive, negative, s):
                return generators_card(positive, negative, s, card, eps)

            acc = cross_validation(f, data_path, t, k)

            if acc > best_acc:
                best_card = card
                best_eps = eps
                best_acc = acc

            print('{0:.2f}     {1:.2f}    {2:.3f}'.format(card, eps, acc))

    print('best_sup: {}, best_eps: {}, best_acc: {}'.format(best_card, best_eps, best_acc))


def test_sup(data_path, t='tic-tac', k=11, frac=100):
    print('sup    eps     acc')
    best_sup, best_eps, best_acc = -1, -1, -1

    for i in range(0, 10, 1):
        sup = i / frac
        for e in range(0, 10, 1):
            eps = e / 100

            def f(positive, negative, s):
                return generators_sup(positive, negative, s, sup, eps)

            acc = cross_validation(f, data_path, t, k)

            if acc > best_acc:
                best_sup = sup
                best_eps = eps
                best_acc = acc

            print('{0:.2f}     {1:.2f}    {2:.3f}'.format(sup, eps, acc))

    print('best_sup: {}, best_eps: {}, best_acc: {}'.format(best_sup, best_eps, best_acc))


def main():
    # test_sup('data/tic-tac/')
    # test_sup('data/titanic', 'titanic', 3, 100)
    test_card('data/tic-tac/', 'tic-tac', 11, 10)


if __name__ == '__main__':
    main()
