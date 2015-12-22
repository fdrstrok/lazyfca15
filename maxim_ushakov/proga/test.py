from FCA import FCA

train = {1: ['b', 'b', 'b'], 2: ['a', 'b', 'a'], 3: ['b', 'b', 'a'], 4: ['b', 'a', 'a']}
test = {5: ['b', 'a', 'b'], 6: ['a', 'a', 'a']}
label = {1: 0, 2: 1, 3: 1, 4: 1}

fca = FCA(train, test, label)
print(fca.data)
fca.CalculateConcepts()
print(fca.Extents)
print(fca.Intents)
fca.CalculateRelation()
print(fca.ConceptRelation)
train_label = fca.ClassifyTestSet()
print(train_label)