from __future__ import print_function
import collections

TEST_DIR = '/home/aida/Projects/IE-hw2/data/SemEval2010_task8_scorer-v1.2/'


def evaluate(y_true, y_pred):
    """Evaluate the model performance given ground-truth labels and prediction results
    These two lists have to be of same length. The i-th element in each list must refer
    to the same instance.

    Args:
        y_true (list): list of ground-truth labels
        y_pred (list): list of predicted labels
    """

    def precision_recall_fscore(y_true, y_pred, rel):
        """Calculate precision, recall, f1 for a specific relation"""
        tp = sum(r1 == r2 for r1, r2 in zip(y_true, y_pred) if r2.startswith(rel))
        p = tp / max(sum(r.startswith(rel) for r in y_pred), 1e-6)
        r = tp / max(sum(r.startswith(rel) for r in y_true), 1e-6)
        return p, r, 2 * p * r / max(p + r, 1e-6)

    label_list = list(set(y_true) - set(['Other']))
    # Relation ignore the direction part.
    relation_list = list(set([label.split('(')[0] for label in label_list]))

    prec, recall, fscore = {}, {}, {}
    for rel in relation_list:
        p, r, f1 = precision_recall_fscore(y_true, y_pred, rel)
        prec[rel] = p
        recall[rel] = r
        fscore[rel] = f1

    # Pretty print detailed report
    print('Results for the individual relations::')
    longest_relation, n = max(map(len, relation_list)), len(relation_list)
    for rel in relation_list:
        header = ('{:>' + str(longest_relation + 4) + '} :  ').format(rel)
        print('{}P = {:7.2f}%\tR = {:7.2f}%\tF1 = {:7.2f}%'.format(header, 100 * prec[rel],
                                                                   100 * recall[rel], 100 * fscore[rel]))

    p, r, f1 = sum(prec.values()) / n, sum(recall.values()) / n, sum(fscore.values()) / n
    print('MACRO-averaged result (excluding Other):')
    print('P = {:7.2f}%\tR = {:7.2f}%\tF1 = {:7.2f}%'.format(100 * p, 100 * r, 100 * f1))
    return p, r, f1


def read_test_data(true_fpath, pred_fpath):
    y_true_dict = collections.defaultdict(lambda: 'Other')
    y_pred_dict = collections.defaultdict(lambda: 'Other')
    for line in open(true_fpath):
        idx, label = line.strip().split()
        y_true_dict[idx] = label
    for line in open(pred_fpath):
        idx, label = line.strip().split()
        y_pred_dict[idx] = label
    idxs = sorted(y_true_dict.keys() | y_pred_dict.keys())
    return [y_true_dict[idx] for idx in idxs], [y_pred_dict[idx] for idx in idxs]


def main():
    y_true, y_pred = read_test_data(TEST_DIR + 'answer_key5.txt', TEST_DIR + 'proposed_answer5.txt')
    evaluate(y_true, y_pred)


if __name__ == '__main__':
    main()