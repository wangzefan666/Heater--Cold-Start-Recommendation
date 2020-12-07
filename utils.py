import time
import datetime
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn import preprocessing as prep
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class timer(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()

        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, timer._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch(iterable, _n=1, drop=True):
    """
    returns batched version of some iterable
    :param iterable: iterable object as input
    :param _n: batch size
    :param drop: if true, drop extra if batch size does not divide evenly,
        otherwise keep them (last batch might be shorter)
    :return: batched version of iterable
    """
    it_len = len(iterable)
    for ndx in range(0, it_len, _n):
        if ndx + _n < it_len:
            yield iterable[ndx:ndx + _n]
        elif drop is False:
            yield iterable[ndx:it_len]


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


def standardize(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_2(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 1 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_3(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 2 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_nzrow, :] /= 2.  # 将下面的 1 改为 2 不也一样吗
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


# prepare idcg
idcg_array = np.arange(100) + 1
idcg_array = 1 / np.log2(idcg_array + 1)
idcg_table = np.zeros(100)
for i in range(100):
    idcg_table[i] = np.sum(idcg_array[:(i + 1)])


def batch_eval_recall(_sess, tf_eval, eval_feed_dict, recall_k, eval_data):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation

    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """

    # 在测试集上得到预测结果 user-item
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval, feed_dict=eval_feed_dict(batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    # 为啥这里要加这一句？
    tf.local_variables_initializer().run()

    # filter non-zero targets
    # 这里的filter没啥用吧，其实过滤掉了1个user, 在lastFM里
    # 原因是: 测试集中有不属于训练集的item, 关于这些item的interactions就被扔掉了
    #        碰巧的是, 这些item恰好囊括了测试集中某user所有的邻接item
    y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]  # bool vector
    y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]  # index vector
    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        y = eval_data.R_test_inf[y_nz, :]

        preds_k = preds_all[:, :at_k]  # 排过序排过序，tf_eval是通过tf.nn.top_k得到的下标数组
        row = np.array([[i] * at_k for i in range(len(preds_k))], dtype=np.int).flatten()
        col = preds_k.flatten()
        x = sp.coo_matrix((np.ones(preds_k.size), (row, col)), shape=y.shape)  # @k all test predictions

        z = y.multiply(x)  # @k test predictions filtered by interactions
        recall.append(np.mean(np.divide(np.sum(z, 1), np.sum(y, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x.data = (np.ones_like(preds_k) * idcg_array[:at_k].reshape((1, -1))).flatten()  # give weights to @n items
        z = y.multiply(x)  # @k test predictions with rank weights filtered by interactions
        dcg = np.sum(z, axis=1)
        idcg = np.sum(y, axis=1) - 1  # -1是因为idcg_table的下标
        idcg[idcg >= at_k] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg


def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
    """
    Args:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        item_warm: train item set

    Returns:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    # np.tile(seq, n): repeat seq for n times
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    item_pos = pos_item_array.reshape((-1))
    # replace: whether element can be chosen more than once
    # ？？为什么 neg item 是直接在 warm item 里面随机抽取，是因为 adj 足够稀疏吗
    item_neg = np.random.choice(item_warm, size=neg * pos_user_array.shape[0], replace=True).reshape((-1))
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((item_pos, item_neg)), np.concatenate((target_pos, target_neg))


