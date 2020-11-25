import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
import data
import model
from utils import tfidf, set_seed
from sklearn.utils.extmath import randomized_svd
import argparse
from pprint import pprint

seed = 0
set_seed(seed)


def main():
    data_name = args.data
    model_select = args.model_select
    rank_out = args.rank  # output_dim
    data_batch_size = 1024  # train batch size
    dropout = args.dropout
    recall_at = [20, 50, 100]
    eval_batch_size = 5000  # the batch size when test
    eval_every = args.eval_every
    num_epoch = 100
    neg = args.neg  # negative sampling rate
    _lr = args.lr
    _decay_lr_every = 10
    _lr_decay = 0.8

    dat = load_data(data_name)
    u_pref = dat['u_pref']  # all user pre embedding
    v_pref = dat['v_pref']  # all item pre embedding
    item_content = dat['item_content']  # all item context matrix
    test_eval = dat['test_eval']  # EvalData
    vali_eval = dat['vali_eval']  # EvalData
    user_list = dat['user_list']  # users of train interactions
    item_list = dat['item_list']  # items of train interactions
    item_warm = np.unique(item_list)  # train item set
    timer = utils.timer(name='main')

    # prep eval
    timer.tic()
    test_eval.init_tf(u_pref, v_pref, None, item_content, eval_batch_size, cold_item=True)  # init data for evaluation
    vali_eval.init_tf(u_pref, v_pref, None, item_content, eval_batch_size, cold_item=True)  # init data for evaluation
    timer.toc('initialized eval data').tic()

    # build model
    heater = model.Heater(latent_rank_in=u_pref.shape[1],
                          user_content_rank=0, item_content_rank=item_content.shape[1],
                          model_select=model_select, rank_out=rank_out,
                          reg=args.reg, alpha=args.alpha, dim=args.dim)
    heater.build_model()
    heater.build_predictor(recall_at)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf').tic()

        best_epoch = 0
        best_recall = 0  # val
        best_test_recall = 0  # test
        for epoch in range(num_epoch):
            user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
            random_idx = np.random.permutation(user_array.shape[0])  # 生成一个打乱的 range 序列
            data_batch = [(n, min(n + data_batch_size, len(random_idx))) for n in
                          range(0, len(random_idx), data_batch_size)]
            loss_epoch = 0.
            reg_loss_epoch = 0.
            diff_loss_epoch = 0.
            rec_loss_epoch = 0.
            for (start, stop) in data_batch:

                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # dropout: used in randomized training
                # indicator's target is the CF pretrain rep
                # set the dropped rows' position in indicator to be 1
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                    zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                else:
                    zero_index = np.array([])
                dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_index) > 0:
                    dropout_indicator[zero_index] = 1

                _, _, loss_out, rec_loss_out, reg_loss_out, diff_loss_out = sess.run(
                    [heater.preds, heater.optimizer, heater.loss, heater.rec_loss, heater.reg_loss, heater.diff_loss],
                    feed_dict={
                        heater.Uin: u_pref[batch_users, :],
                        heater.Vin: v_pref[batch_items, :],
                        heater.Vcontent: item_content[batch_items, :],
                        heater.dropout_item_indicator: dropout_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }
                )
                loss_epoch += loss_out
                rec_loss_epoch += rec_loss_out
                reg_loss_epoch += reg_loss_out
                diff_loss_epoch += diff_loss_out
                if np.isnan(loss_epoch):
                    raise Exception('f is nan')

            if (epoch + 1) % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            # eval on val
            if epoch % eval_every == 0:
                recall, precision, ndcg = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                                  eval_feed_dict=heater.get_eval_dict,
                                                                  recall_k=recall_at, eval_data=vali_eval)
            # if get a better eval result on val, update test result
            # best_recall and best_test_recall are global variables while others are local ones
            if np.sum(recall) > np.sum(best_recall):
                best_recall = recall
                test_recall, test_precision, test_ndcg = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                                                 eval_feed_dict=heater.get_eval_dict,
                                                                                 recall_k=recall_at,
                                                                                 eval_data=test_eval)
                best_test_recall = test_recall
                best_epoch = epoch

            # print results at every epoch
            timer.toc('%d loss=%.4f reg_loss=%.4f diff_loss=%.4f rec_loss=%.4f' % (
                epoch, loss_epoch/len(data_batch), reg_loss_epoch/len(data_batch),
                diff_loss_epoch/len(data_batch), rec_loss_epoch/len(data_batch)
            )).tic()
            print('\t\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_at]))  # ljust: padding to fixed len
            print('Current recall\t\t%s' % (' '.join(['%.6f' % i for i in recall])))
            print('Current precision\t%s' % (' '.join(['%.6f' % i for i in precision])))
            print('Current ndcg\t\t%s' % (' '.join(['%.6f' % i for i in ndcg])))
            print('Current test recall\t%s' % (' '.join(['%.6f' % i for i in test_recall])))
            print('Current test precision\t%s' % (' '.join(['%.6f' % i for i in test_precision])))
            print('Current test ndcg\t%s' % (' '.join(['%.6f' % i for i in test_ndcg])))
            print('best[%d] vali recall:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_recall])))
            print('best[%d] test recall:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_test_recall])))


def load_data(data_name):
    timer = utils.timer(name='main')
    data_path = './data/' + data_name
    user_pref_file = data_path + '/U_BPR.npy'
    item_pref_file = data_path + '/V_BPR.npy'
    item_content_file = data_path + '/item_features.txt'
    train_file = data_path + '/train.csv'
    test_file = data_path + '/test.csv'
    vali_file = data_path + '/vali.csv'
    dat = {}

    # load preference data
    timer.tic()
    dat['u_pref'] = np.load(user_pref_file)
    dat['v_pref'] = np.load(item_pref_file)
    timer.toc('loaded U:%s,V:%s' % (str(dat['u_pref'].shape), str(dat['v_pref'].shape))).tic()

    # pre-process preference data
    _, dat['u_pref'] = utils.standardize(dat['u_pref'])
    _, dat['v_pref'] = utils.standardize_2(dat['v_pref'])
    timer.toc('standardized U,V').tic()

    # load item(article) content data
    # load_svmlight_file(file): 读取svmlight格式的数据文件，文件存放格式
    # <label> <feature-id>:<feature-value> <feature-id>:<feature-value> ...
    # 其中 zero_based 选项，如果为 False 的话会将所有的 indices 减 1
    # 返回 (X, y)，其中 X 是 scipy.sparse matrix，y 是 numpy.ndarray
    item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
    # tfidf 文本特征化
    item_content = tfidf(item_content)
    # svd 特征降维
    u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
    item_content = u * s
    # 特征标准化
    _, item_content = utils.standardize(item_content)
    dat['item_content'] = item_content
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()

    # load split
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    timer.toc('read train triplets %s' % str(train.shape))

    dat['test_eval'] = data.load_eval_data(test_file)
    dat['vali_eval'] = data.load_eval_data(vali_file)
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main_CiteULike")

    parser.add_argument('--data', type=str, default='CiteULike', help='path to eval in the downloaded folder')
    parser.add_argument('--model-select', nargs='+', type=int, default=[200],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units')
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--neg', type=int, default=5, help='negative sampling rate')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='diff loss parameter')
    parser.add_argument('--reg', type=float, default=0.0001, help='regularization')
    parser.add_argument('--dim', type=int, default=5, help='number of experts')

    args = parser.parse_args()
    pprint(vars(args))

    main()
