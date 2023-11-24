def precision(tp, fp): 
    '''
    computes the classifier precision as TP/TP+FP

    :returns: precision
    '''
    return (tp)/(tp+fp)


def recall(tp, fn):
    '''
    computes the classifier recall as TP/TP+FN

    :returns: recall
    '''

    return (tp)/(tp+fn)


def accuracy(tp, tn, fp, fn):
    '''
    computes accuracy as TP+TN/TP+FP+TN+FN

    :returns: accuracy
    '''

    return (tp+tn)/(tp+tn+fp+fn)


def evaluate(tp, tn, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    acc = accuracy(tp, tn, fp, fn)
    f = 2*prec*rec/(prec+rec) #f-score

    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    print("Accuracy: " + str(acc))
    print("F-score(beta=1): " + str(f))
    