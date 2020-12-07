import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import data
import model
import scipy.sparse as sp
import argparse
import scipy.sparse
from pprint import pprint
from utils import *


seed = 0
set_seed(seed)


def main():
    data_name = args.data
    model_select = args.model_select
    rank_out = args.rank
    data_batch_size = 1024
    dropout = args.dropout
    recall_at = [20, 50, 100]
    eval_batch_size = 5000  # the batch size when test
    eval_every = args.eval_every
    num_epoch = 100
    neg = args.neg

    _lr = args.lr
    _decay_lr_every = 2
    _lr_decay = 0.9

    dat = load_data(data_name)
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    test_eval = dat['test_eval']
    vali_eval = dat['vali_eval']
    user_content = dat['user_content']
    user_list = dat['user_list']
    item_list = dat['item_list']
    item_warm = np.unique(item_list)
    timer = utils.timer(name='main').tic()

    # prep eval
    eval_batch_size = eval_batch_size
    timer.tic()
    test_eval.init_tf(u_pref, v_pref, user_content, None, eval_batch_size, cold_user=True)  # init data for evaluation
    vali_eval.init_tf(u_pref, v_pref, user_content, None, eval_batch_size, cold_user=True)  # init data for evaluation
    timer.toc('initialized eval data').tic()

    heater = model.Heater(latent_rank_in=u_pref.shape[1],
                          user_content_rank=user_content.shape[1],
                          item_content_rank=0,
                          model_select=model_select,
                          rank_out=rank_out, reg=args.reg, alpha=args.alpha, dim=args.dim)
    heater.build_model()
    heater.build_predictor(recall_at)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        best_epoch = 0
        best_recall = 0  # val
        best_test_recall = 0  # test
        for epoch in range(num_epoch):
            user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
            random_idx = np.random.permutation(user_array.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
            loss_epoch = 0.
            reg_loss_epoch = 0.
            diff_loss_epoch = 0.
            rec_loss_epoch = 0.
            for (start, stop) in data_batch:

                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # content
                user_content_batch = user_content[batch_users, :].todense()
                # dropout
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                    zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                else:
                    zero_index = np.array([])
                dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_index) > 0:
                    dropout_indicator[zero_index] = 1

                _, _, loss_out, rec_loss_out, reg_loss_out, diff_loss_out = sess.run(
                    [heater.preds, heater.optimizer, heater.loss,
                     heater.rec_loss, heater.reg_loss, heater.diff_loss],
                    feed_dict={
                        heater.Uin: u_pref[batch_users, :],
                        heater.Vin: v_pref[batch_items, :],
                        heater.Ucontent: user_content_batch,
                        heater.dropout_user_indicator: dropout_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }
                )
                loss_epoch += loss_out
                rec_loss_out += rec_loss_out
                reg_loss_epoch += reg_loss_out
                diff_loss_epoch += diff_loss_out
                if np.isnan(loss_epoch):
                    raise Exception('f is nan')

            if (epoch + 1) % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            if epoch % eval_every == 0:
                recall, precision, ndcg = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                                  eval_feed_dict=heater.get_eval_dict,
                                                                  recall_k=recall_at, eval_data=vali_eval)

            # checkpoint
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
                epoch, loss_epoch / len(data_batch), reg_loss_epoch / len(data_batch),
                diff_loss_epoch / len(data_batch), rec_loss_epoch / len(data_batch)
            )).tic()
            print(
                '\t\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_at]))  # ljust: padding to fixed len
            print('Current recall\t\t%s' % (' '.join(['%.6f' % i for i in recall])))
            print('Current precision\t%s' % (' '.join(['%.6f' % i for i in precision])))
            print('Current ndcg\t\t%s' % (' '.join(['%.6f' % i for i in ndcg])))
            print('Current test recall\t%s' % (' '.join(['%.6f' % i for i in test_recall])))
            print('Current test precision\t%s' % (' '.join(['%.6f' % i for i in test_precision])))
            print('Current test ndcg\t%s' % (' '.join(['%.6f' % i for i in test_ndcg])))
            print('best[%d] vali recall:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_recall])))
            print('best[%d] test recall:\t%s' % (best_epoch, ' '.join(['%.6f' % i for i in best_test_recall])))


def load_data(data_name):
    timer = utils.timer(name='main').tic()
    data_path = './data/' + data_name
    u_file = data_path + '/U_BPR.npy'
    v_file = data_path + '/V_BPR.npy'
    user_content_file = data_path + '/user_content.npz'
    train_file = data_path + '/train.csv'
    test_file = data_path + '/test.csv'
    vali_file = data_path + '/vali.csv'

    dat = {}
    # load preference data
    timer.tic()
    u_pref = np.load(u_file)
    v_pref = np.load(v_file)
    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref
    timer.toc('loaded U:%s,V:%s' % (str(u_pref.shape), str(v_pref.shape))).tic()

    # pre-process
    _, dat['u_pref'] = utils.standardize_2(u_pref)
    _, dat['v_pref'] = utils.standardize(v_pref)
    timer.toc('standardized U,V').tic()

    # load content data
    timer.tic()
    user_content = scipy.sparse.load_npz(user_content_file)
    dat['user_content'] = user_content.tolil(copy=False)
    timer.toc('loaded user feature sparse matrix: %s' % (str(user_content.shape))).tic()

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['warm_item'] = np.unique(train['iid'].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['vali_eval'] = data.load_eval_data(vali_file, cold_user=True, test_item_ids=dat['warm_item'])
    dat['test_eval'] = data.load_eval_data(test_file, cold_user=True, test_item_ids=dat['warm_item'])
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main_LastFM")
    parser.add_argument('--data', type=str, default='LastFM', help='path to eval in the downloaded folder')
    parser.add_argument('--model-select', nargs='+', type=int, default=[200],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units')
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='diff loss parameter')
    parser.add_argument('--reg', type=float, default=0.001, help='regularization')
    parser.add_argument('--dim', type=int, default=5, help='number of experts')

    args = parser.parse_args()
    pprint(vars(args))

    main()
