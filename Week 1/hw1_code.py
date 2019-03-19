import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

def load_data():
    # Import the data
    fakeArr = [[line.rstrip('\n'), 0] for line in open('./data/clean_fake.txt')]
    realArr = [[line.rstrip('\n'), 1] for line in open('./data/clean_real.txt')]
    total = fakeArr + realArr

    np.random.shuffle(total)

    Y = []
    X = []
    for headline in total:
        Y.append(headline[1])
        X.append(headline[0])

    # Vectorize the data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    arr = X.toarray()

    return vectorizer, arr[:2286], arr[2286: 2286 + 490], arr[2286 + 490:], Y[:2286], Y[2286: 2286 + 490], Y[2286 + 490:]

def errorF(X, XofTree):

    count = 0
    for i in range(len(X)):
        if ( X[i] == XofTree[i]):
            count += 1

    return 1 - count/len(X)

def select_model(vectorizer, trainingX, validationX, testX, trainingY, validationY, testY):

    depth = []
    error_gini = []
    error_ig = []
    for i in range(1, 35):
        depth.append(i)

        clf = DecisionTreeClassifier(random_state=0, max_depth=i, criterion='gini')
        clf.fit(trainingX, trainingY)
        YofTree = clf.predict(X=validationX)
        error = errorF(validationY, YofTree)
        error_gini.append(error)

        clf = DecisionTreeClassifier(random_state=0, max_depth=i, criterion='entropy')
        clf.fit(trainingX, trainingY)
        YofTree = clf.predict(X=validationX)
        error = errorF(validationY, YofTree)
        error_ig.append(error)

    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.title('Error According to Max Depth of Decision Tree')
    plt.xticks(np.arange(min(depth), max(depth)+1, 4.0))
    plt.plot(depth, error_gini, depth, error_ig)
    plt.legend(['Using Gini', 'Using Info Gain'])
    plt.show()

    maxDepth = 6
    criterion = 'gini'

    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, criterion=criterion)
    clf.fit(trainingX, trainingY)
    probsOfClasses = clf.predict_proba(trainingX)
    dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True,special_characters=True,feature_names=vectorizer.get_feature_names())
    graph = graphviz.Source(dot_data)
    graph.render('tree_output')

    for i in range(len(depth)):
        print('For depth {0}, gini error = {1}, ig error = {2}'.format(i, error_gini[i], error_ig[i]))

    return None

def compute_information_gain(vectorizer, X, Y, feature):
    
    try:
        featureIdx = vectorizer.get_feature_names().index(feature)
    except:
        return -1

    total = len(Y)
    reals = 0
    fakes = 0

    for i in range(total):
        if (Y[i] == 0):
            fakes += 1
        elif (Y[i] == 1):
            reals += 1

    H_Y = -1 * ( (reals/total) * np.log2((reals/total)) + (fakes/total) * np.log2((fakes/total)) )

    realAndFeature = 0
    fakeAndFeature = 0

    for i in range(total):
        if ( X[i][featureIdx] > 0 and Y[i] == 0 ):
            fakeAndFeature += 1
        elif ( X[i][featureIdx] > 0 and Y[i] == 1 ):
            realAndFeature += 1

    pRealAndFeature = realAndFeature/total
    pFakeAndFeature = fakeAndFeature/total
    pFeature = ( fakeAndFeature + realAndFeature )/total

    H_YgiveFeature = -1 * ( pRealAndFeature * np.log2( pRealAndFeature/pFeature ) + pFakeAndFeature * np.log2( pFakeAndFeature/pFeature ) )

    return H_Y - H_YgiveFeature

def main():
    vectorizer, trainingX, validationX, testX, trainingY, validationY, testY = load_data()
    select_model(vectorizer, trainingX, validationX, testX, trainingY, validationY, testY)

    IG = compute_information_gain(vectorizer, trainingX, trainingY, "donald")
    print("I(Y, donald) = ", IG)
    IG = compute_information_gain(vectorizer, trainingX, trainingY, "trumps")
    print("I(Y, trumps) = ", IG)
    IG = compute_information_gain(vectorizer, trainingX, trainingY, "the")
    print("I(Y, the) = ", IG)
    IG = compute_information_gain(vectorizer, trainingX, trainingY, "hillary")
    print("I(Y, hillary) = ", IG)
    IG = compute_information_gain(vectorizer, trainingX, trainingY, "voting")
    print("I(Y, voting) = ", IG)

main()
