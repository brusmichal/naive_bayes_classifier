from sklearn import datasets
import pandas as pd
from NaiveBayesClassifier import NaiveBayesClassifier


def main():
    test()


def test():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    classifier = NaiveBayesClassifier(df, iris.target_names, 5)
    classifier.test()
    classifier.find_prediction([4, 2, 5, 1])


if __name__ == '__main__':
    main()
