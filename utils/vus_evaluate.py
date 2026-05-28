import contextlib
import os
import sys

import numpy as np
from vus.metrics import get_metrics
from vus.affiliation.generics import convert_vector_to_events
from vus.affiliation.metrics import pr_from_events
from utils.spot import SPOT


@contextlib.contextmanager
def quiet_non_tty_output():
    if sys.stderr.isatty():
        yield
        return

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def getThreshold(init_score, test_score, q=1e-2):
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold


def _to_label_vector(test_label):
    if test_label is None:
        return None

    label = np.asarray(test_label)
    if label.ndim > 1:
        label = (np.sum(label, axis=1) > 0).astype(int)
    return label.reshape(-1).astype(int)


def _affiliation_metrics(pred_vec, label_vec):
    events_pred = convert_vector_to_events(pred_vec.astype(np.float32))
    events_gt = convert_vector_to_events(label_vec.astype(np.float32))
    metrics = pr_from_events(events_pred, events_gt, (0, len(pred_vec)))
    precision = metrics['Affiliation_Precision']
    recall = metrics['Affiliation_Recall']
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def evaluate(init_score, test_score, test_label=None, q=1e-2, slidingWindow=5, metric_mode='aff'):
    init_score = np.asarray(init_score).reshape(-1)
    score_vec = np.asarray(test_score).reshape(-1)
    label_vec = _to_label_vector(test_label)

    if metric_mode not in {'aff', 'vus', 'all'}:
        raise ValueError(f'Unknown metric_mode: {metric_mode}')

    results = {}

    if metric_mode in {'vus', 'all'}:
        with quiet_non_tty_output():
            results.update(get_metrics(score_vec, label_vec, metric='vus', slidingWindow=slidingWindow))

    if metric_mode in {'aff', 'all'}:
        threshold = getThreshold(init_score, score_vec, q=q)
        pred_vec = (score_vec > threshold).astype(int)

        if label_vec is not None:
            aff_precision, aff_recall, aff_f1 = _affiliation_metrics(pred_vec, label_vec)
            results['Affiliation_Precision'] = aff_precision
            results['Affiliation_Recall'] = aff_recall
            results['Affiliation_F1'] = aff_f1
        results['q'] = q
        results['threshold'] = threshold


    return results
