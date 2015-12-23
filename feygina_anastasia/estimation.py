

def accuracy(res):
    return float(res["PP"] + res["NN"]) / max(1, res["PP"] + res["NN"] + res["PN"] + res["NP"] + res["contradictory"])


def precision(res):
    return float(res["NN"]) / max(1, res["NN"] + res["PN"])


def recall(res):
    return float(res["NN"]) / max(1, res["NN"] + res["NP"])


def F1_score(res):
    prec = precision(res)
    rec = recall(res)
    return 2 * prec * rec / max(1, prec + rec)


def summary(res):
    stats = {}
    stats["accuracy"] = accuracy(res)
    stats["precision"] = precision(res)
    stats["recall"] = recall(res)
    stats["f1"] = F1_score(res)
    return stats

