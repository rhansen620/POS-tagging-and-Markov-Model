#Rachel Hansen
from typing import Iterator, Sequence, Text, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


NDArray = Union[np.ndarray, spmatrix]
TokenSeq = Sequence[Text]
PosSeq = Sequence[Text]


def read_ptbtagged(ptbtagged_path: str) -> Iterator[Tuple[TokenSeq, PosSeq]]:
    """Reads sentences from a Penn TreeBank .tagged file.
    Each sentence is a sequence of tokens and part-of-speech tags.

    Penn TreeBank .tagged files contain one token per line, with an empty line
    marking the end of each sentence. Each line is composed of a token, a tab
    character, and a part-of-speech tag. Here is an example:

        What	WP
        's	VBZ
        next	JJ
        ?	.

        Slides	NNS
        to	TO
        illustrate	VB
        Shostakovich	NNP
        quartets	NNS
        ?	.

    :param ptbtagged_path: The path of a Penn TreeBank .tagged file, formatted
    as above.
    :return: An iterator over sentences, where each sentence is a tuple of
    a sequence of tokens and a corresponding sequence of part-of-speech tags.
    """
    f = open(ptbtagged_path)
    mytuples = []
    olist = []
    klist = []
    for line in f:
        p =line.strip("\n")
        r = p.split("\t")
    #    print (p)
 #   print (r)
        if p == "" or p == "\t ":
            if olist == "":
                continue 
            else:
   #         mytuples.append([" ".join(olist), " ".join(klist)])
                mytuples.append([olist, klist])
                olist = []
                klist = []
                continue 
    #  print (r)
        o = r[0]
        k = r[1]
        olist.append(o)
        klist.append(k)
    mytuples.append([olist, klist])
    #print (len(mytuples))  
    return mytuples 

class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        self.vectorizer = DictVectorizer()
        self.le = preprocessing.LabelEncoder()
        self.lrc = LogisticRegression(max_iter=100000, class_weight= {0:3,1:9})

    def train(self, tagged_sentences: Iterator[Tuple[TokenSeq, PosSeq]]) -> Tuple[NDArray, NDArray]:
        """Trains the classifier on the part-of-speech tagged sentences,
        and returns the feature matrix and label vector on which it was trained.

        The feature matrix should have one row per training token. The number
        of columns is up to the implementation, but there must at least be 1
        feature for each token, named "token=T", where "T" is the token string,
        and one feature for the part-of-speech tag of the preceding token,
        named "pos-1=P", where "P" is the part-of-speech tag string, or "<s>" if
        the token was the first in the sentence. For example, if the input is:

            What	WP
            's	VBZ
            next	JJ
            ?	.

        Then the first row in the feature matrix should have features for
        "token=What" and "pos-1=<s>", the second row in the feature matrix
        should have features for "token='s" and "pos-1=WP", etc. The alignment
        between these feature names and the integer columns of the feature
        matrix is given by the `feature_index` method below.

        The label vector should have one entry per training token, and each
        entry should be an integer. The alignment between part-of-speech tag
        strings and the integers in the label vector is given by the
        `label_index` method below.

        :param tagged_sentences: An iterator over sentences, where each sentence
        is a tuple of a sequence of tokens and a corresponding sequence of
        part-of-speech tags.
        :return: A tuple of (feature-matrix, label-vector).
        """
        mylist = []
        pppp = []
        for i in tagged_sentences:
            previouspos = "<s>" #placed here so that it auto-resets for every sentence 
        #    asentence = i[0] #takes the first value (the sentence) in tagged_sentences
         #   positems = i[1] #takes the second value (the pos) in tagged_sentences
            for index in range(len(i[0])):
                if len(i[0][index]) > 3:
                    mylist.append({"token": i[0][index], "pos-1": previouspos, "suffix": i[0][index][-3:], "prefix": i[0][index][:3]})
                else:
                    mylist.append({"token": i[0][index], "pos-1": previouspos, "suffix": i[0][index], "prefix": i[0][index]})
                previouspos = i[1][index] 
                pppp.append(previouspos)
    #    print(mylist)
        fm = self.vectorizer.fit_transform(mylist)
        lv = self.le.fit_transform(pppp)
        self.lrc.fit(fm,lv)
#        print (lv)
#        print(fm)
        return fm, lv

        

    def feature_index(self, feature: Text) -> int:
        """Returns the column index corresponding to the given named feature.

        The `train` method should always be called before this method is called.

        :param feature: The string name of a feature.
        :return: The column index of the feature in the feature matrix returned
        by the `train` method.
        """
        return self.vectorizer.get_feature_names().index(feature)
        

    def label_index(self, label: Text) -> int:
        """Returns the integer corresponding to the given part-of-speech tag

        The `train` method should always be called before this method is called.

        :param label: The part-of-speech tag string.
        :return: The integer for the part-of-speech tag, to be used in the label
        vector returned by the `train` method.
        """
        return self.le.transform([label])[0]        

    def predict(self, tokens: TokenSeq) -> PosSeq:
        """Predicts part-of-speech tags for the sequence of tokens.

        This method delegates to either `predict_greedy` or `predict_viterbi`.
        The implementer may decide which one to delegate to.

        :param tokens: A sequence of tokens representing a sentence.
        :return: A sequence of part-of-speech tags, one for each token.
        """
        _, pos_tags = self.predict_greedy(tokens)
        # _, _, pos_tags = self.predict_viterbi(tokens)
        return pos_tags

    def predict_greedy(self, tokens: TokenSeq) -> Tuple[NDArray, PosSeq]:
        """Predicts part-of-speech tags for the sequence of tokens using a
        greedy algorithm, and returns the feature matrix and predicted tags.

        Each part-of-speech tag is predicted one at a time, and each prediction
        is considered a hard decision, that is, when predicting the
        part-of-speech tag for token i, the model will assume that its
        prediction for token i-1 is correct and unchangeable.

        The feature matrix should have one row per input token, and be formatted
        in the same way as the feature matrix in `train`.

        :param tokens: A sequence of tokens representing a sentence.
        :return: The feature matrix and the sequence of predicted part-of-speech
        tags (one for each input token).
        """
        predictfm = []
        tags = ["<s>"]
        for token in tokens:
            if len(token):
                predictfm.append({"token": token, "pos-1": tags[-1], "suffix": token[-3:], "prefix": token[:3]})
            else:
                predictfm.append({"token": token, "pos-1": tags[-1], "suffix": token[-3:], "prefix": token[:3]})
            t = self.vectorizer.transform([predictfm[-1]])
            p = self.lrc.predict(t)
            newlabel = self.le.inverse_transform(p)
            tags.append(newlabel[0])
        newfm = self.vectorizer.transform(predictfm)
    #    print(tags)
        return newfm, tags[1:]



