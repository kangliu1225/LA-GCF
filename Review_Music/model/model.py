# -*- coding: utf-8 -*-
import argparse
import dgl.function as fn
from util import *
import torch
import dgl
from load_data_graph_augmutation import *
from util import *
import random
import heapq
import pickle
import torch.nn.functional as F
from abc import ABC
import torch.optim as optim
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['DGLBACKEND'] = 'pytorch'


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)


seed_everything(2024)

global_emb_size = 64
for path in os.listdir("../data/"):
    if ".txt" not in path:
        dataset_name = path
# dataset_name = os.listdir("../data/")[0]
eps = 1e-12


class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info, strange_users, strange_users_max, user_items_train = load_sentiment_data(
            dataset_path)
        # sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(
        #     dataset_path)

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        self.review_feat_updated = {}

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            # 让物品的id从max user id开始，相当于将用户和物品节点视为一类节点；
            item_id = [int(i) + self._num_user for i in info['item_id'].to_list()]
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self.user_item_rating = {}
        self.user_rating_count = {}
        self.user_ratings_test = {}
        self.user_item_ratings = {}

        self.user_items = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                np.array(data[2], dtype=np.int64)

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)

            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)  # @ss 双向
            rating_values = np.concatenate([rating, rating], axis=0)  # @ss 双向
            # rating_values = np.concatenate([rating], axis=0)                 # @ss 双向

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                    self.user_item_ratings[uid] = {}
                    self.user_items[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))
                self.user_item_ratings[uid][iid] = rating[i]
                self.user_items[uid].append(iid)

                if uid not in self.user_rating_count:
                    self.user_rating_count[uid] = [0, 0, 0, 0, 0]

                self.user_rating_count[uid][rating[i] - 1] += 1

            return rating_pairs, rating_values

        def _generate_valid_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_ratings_test:
                    self.user_ratings_test[uid] = []

                self.user_ratings_test[uid].append(rating[i])

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)
        self.valid_rating_pairs, self.valid_rating_values = _generate_valid_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        count_mis = 0
        count_same = 0
        count_all = 0
        for uid, items in self.user_ratings_test.items():
            count_all += len(items)
            max_rate_train = np.where(self.user_rating_count[uid] == np.max(self.user_rating_count[uid]))[0]
            for i in items:
                if i - 1 not in max_rate_train:
                    count_mis += 1
                else:
                    count_same += 1

        print(count_mis, count_same, count_all, len(self.test_rating_values))

        ## find and collect extremely distributed samples
        self.extra_dist_pairs = {}
        self.extra_uid, self.extra_iid, self.extra_r_idx = [], [], []
        for uid, l in self.user_rating_count.items():

            max_count = np.max(l)
            max_idx = np.where(l == max_count)[0]

            for i, c in enumerate(l):
                # if c == 0 or abs(max_idx.max() - i) <= 1 or abs(max_idx.min() - i) <= 1:
                if i in max_idx or c == 0:
                    continue

                if c / max_count <= 0.2:
                    if uid not in self.extra_dist_pairs:
                        self.extra_dist_pairs[uid] = []
                    self.extra_dist_pairs[uid].append((i + 1, c))
                    for item in self.user_item_rating[uid]:
                        self.extra_uid.append(uid)
                        self.extra_iid.append(item[0])
                        self.extra_r_idx.append(i)

        self.item_rate_review = {}

        for u, d in self.user_item_ratings.items():
            for i, r in d.items():
                review = self.train_review_feat[(u, i - self._num_user)]
                if i not in self.item_rate_review:
                    self.item_rate_review[i] = {}
                if r not in self.item_rate_review[i]:
                    self.item_rate_review[i][r] = []
                self.item_rate_review[i][r].append(review)

        self.mean_review_feat_list_1 = []
        self.mean_review_feat_list_2 = []
        self.mean_review_feat_list_3 = []
        self.mean_review_feat_list_4 = []
        self.mean_review_feat_list_5 = []
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value
            if key[1] + self._num_user not in self.user_item_ratings[key[0]]:
                continue

            if self.user_item_ratings[key[0]][key[1] + self._num_user] == 1:
                self.mean_review_feat_list_1.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 2:
                self.mean_review_feat_list_2.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 3:
                self.mean_review_feat_list_3.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 4:
                self.mean_review_feat_list_4.append(value)

            else:
                self.mean_review_feat_list_5.append(value)

        # self.mean_review_feat_1 = torch.mean(torch.stack(self.mean_review_feat_list_1, dim=0), dim=0)
        # self.mean_review_feat_2 = torch.mean(torch.stack(self.mean_review_feat_list_2, dim=0), dim=0)
        # self.mean_review_feat_3 = torch.mean(torch.stack(self.mean_review_feat_list_3, dim=0), dim=0)
        # self.mean_review_feat_4 = torch.mean(torch.stack(self.mean_review_feat_list_4, dim=0), dim=0)
        # self.mean_review_feat_5 = torch.mean(torch.stack(self.mean_review_feat_list_5, dim=0), dim=0)

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph()

    def update_graph(self, uid_list, iid_list, r_list):
        uid_list, iid_list, r_list = np.array(uid_list), np.array(iid_list), np.array(r_list)
        rating_pairs = (uid_list, iid_list)
        rating_pairs_rev = (iid_list, uid_list)
        self.train_rating_pairs = np.concatenate([self.train_rating_pairs, rating_pairs, rating_pairs_rev], axis=1)

        self.train_rating_values = np.concatenate([self.train_rating_values, r_list, r_list], axis=0)
        # c0, c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0, 0
        #
        # for i, u in enumerate(uid_list):
        #
        #     r = r_list[i]
        #     iid = iid_list[i]
        #     if r in self.item_rate_review[iid]:
        #         review = torch.mean(torch.stack(self.item_rate_review[iid][r], dim=0), dim=0)
        #         c0 += 1
        #     elif r == 1:
        #         review = self.mean_review_feat_1
        #         c1 += 1
        #     elif r == 2:
        #         review = self.mean_review_feat_2
        #         c2 += 1
        #     elif r == 3:
        #         review = self.mean_review_feat_3
        #         c3 += 1
        #     elif r == 4:
        #         review = self.mean_review_feat_4
        #         c4 += 1
        #     else:
        #         review = self.mean_review_feat_5
        #         c5 += 1
        #
        #     self.review_feat_updated[(u, iid_list[i])] = review
        #     self.review_feat_updated[(iid_list[i], u)] = review
        # print(c0, c1, c2, c3, c4, c5)

        self.train_enc_graph_updated = self._generate_enc_graph()

    def _generate_enc_graph(self):
        # user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user), dtype=np.float32)
        # for i in range(len(self.train_rating_values)):
        #     user_item_r[[self.train_rating_pairs[0][i], self.train_rating_pairs[1][i]]] = self.train_rating_values[i]
        record_size = self.train_rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(self.train_rating_pairs[0][x], self.train_rating_pairs[1][x])] for
                            x in
                            range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = self.train_rating_pairs

        graph_dict = {}
        left_dict={}
        right_dict={}
        row_col_dict={}
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_rating_values == rating)
            ridx=ridx[0][:len(ridx[0])//2]
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            row_col_dict[rating]=(rrow,rcol- self._num_user,review_feat_list[ridx])

            graph_dict[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        # ############# 用于统计每次交互的评分在用户中出现的次数和项目中出现的次数之和
        # ui_ratings = {}
        # for i in range(len(rating_row)):
        #     if rating_row[i] not in ui_ratings.keys():
        #         ui_ratings[rating_row[i]] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        #     ui_ratings[rating_row[i]][self.train_rating_values[i]] += 1
        #
        # # ui_ratings_list = []
        # # for i in range(len(ui_ratings)):
        # #     ui_ratings_list += [ui_ratings[i]]
        # def  ui_ratings_calc_norm(x, d):
        #     return 1/math.pow(x,d)
        # score_freq = []
        # for i in range(len(rating_row)):
        #     # ui_ratings_list[rating_row[i]]   ui_ratings_list[rating_col[i]]
        #     score_freq += [ui_ratings_calc_norm(ui_ratings[rating_row[i]][self.train_rating_values[i]],
        #                                         0.25) * ui_ratings_calc_norm(
        #         ui_ratings[rating_col[i]][self.train_rating_values[i]], 0.25)]
        # ##################################################################
        # Aspect图构建
        # 读取aspect嵌入
        file_path = r"D:\workspace\dataset\aspect\dm5\13.aspect_embeddings_from_total.pkl"
        aspect_emb_dict = torch.load(file_path, map_location='cpu')
        user_path = r"D:\workspace\dataset\aspect\dm5\12.user_aspect_sentiment.json"
        with open(user_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
        aspect_set = set()
        for u in user_data.values():
            aspect_set.update(u.keys())
        aspect_ids = sorted(list(aspect_set))
        aspect2id = {a: i for i, a in enumerate(aspect_ids)}
        id2aspect = {v: k for k, v in aspect2id.items()}
        aspect_emb_list = [aspect_emb_dict[id2aspect[x]] for x in range(len(aspect2id.values()))]

        user_src, user_dst, user_sentiment_list, = [], [], []
        for user, aspects in user_data.items():
            for aspect, sentiments in aspects.items():
                for s, count in sentiments.items():
                    for _ in range(count):
                        user_src.append(int(user))
                        user_dst.append(aspect2id[aspect])
                        user_sentiment_list.append(sentiment_map[s])
        # item
        item_path = r"D:\workspace\dataset\aspect\dm5\12.item_aspect_sentiment.json"
        with open(item_path, "r", encoding="utf-8") as f:
            item_data = json.load(f)
        item_src, item_dst, item_sentiment_list = [], [], []
        for item, aspects in item_data.items():
            for aspect, sentiments in aspects.items():
                for s, count in sentiments.items():
                    for _ in range(count):
                        item_src.append(int(item))
                        item_dst.append(aspect2id[aspect])
                        item_sentiment_list.append(sentiment_map[s])
        ## 512维度向量
        input_path = r"D:\workspace\dataset\aspect\dm5\12.user_aspect_total.json"
        with open(input_path, "r", encoding="utf-8") as f:
            user_aspects = json.load(f)
        vec_dim = len(aspect2id)
        user_vectors = {}
        for user, aspects in user_aspects.items():
            h_u = np.zeros(vec_dim, dtype=np.float32)
            for a, count in aspects.items():
                if a in aspect2id:  # 只考虑在 aspect2id 中的 aspect
                    idx = aspect2id[a]
                    h_u[idx] = count
            user_vectors[user] = h_u
        user_vectors_list = [
            user_vectors[str(x)] if str(x) in user_vectors else np.zeros(vec_dim)
            for x in range(self._num_user)
        ]
        ## 512维度向量
        input_path = r"D:\workspace\dataset\aspect\dm5\12.item_aspect_score.json"
        with open(input_path, "r", encoding="utf-8") as f:
            item_aspects = json.load(f)
        vec_dim = len(aspect2id)
        item_vectors = {}
        for item, aspects in item_aspects.items():
            h_i = np.zeros(vec_dim, dtype=np.float32)
            for a, count in aspects.items():
                if a in aspect2id:  # 只考虑在 aspect2id 中的 aspect
                    idx = aspect2id[a]
                    h_i[idx] = count
            item_vectors[item] = h_i
        item_vectors_list = [
            item_vectors[str(x)] if str(x) in item_vectors else np.zeros(vec_dim)
            for x in range(self._num_item)
        ]
        # ##################################################################
        user_item_src = self.train_datas[0]
        user_item_dst=[x - self._num_user for x in self.train_datas[1]]
        hetero_g = dgl.heterograph({
                ("user", "user-item", "item"): (user_item_src, user_item_dst),
                ("item", "item-user", "user"): (user_item_dst, user_item_src),
            ("user", "user-item-1", "item"): (row_col_dict[1][0], row_col_dict[1][1]),
            ("user", "user-item-2", "item"): (row_col_dict[2][0], row_col_dict[2][1]),
            ("user", "user-item-3", "item"): (row_col_dict[3][0], row_col_dict[3][1]),
            ("user", "user-item-4", "item"): (row_col_dict[4][0], row_col_dict[4][1]),
            ("user", "user-item-5", "item"): (row_col_dict[5][0], row_col_dict[5][1]),
            ("item", "item-user-1", "user"): (row_col_dict[1][1], row_col_dict[1][0]),
            ("item", "item-user-2", "user"): (row_col_dict[2][1], row_col_dict[2][0]),
            ("item", "item-user-3", "user"): (row_col_dict[3][1], row_col_dict[3][0]),
            ("item", "item-user-4", "user"): (row_col_dict[4][1], row_col_dict[4][0]),
            ("item", "item-user-5", "user"): (row_col_dict[5][1], row_col_dict[5][0]),
                ("aspect", "aspect-user", "user"): (user_dst, user_src),
                ("aspect", "aspect-item", "item"): (item_dst, item_src)
            },
            num_nodes_dict={
                "user": self._num_user,
                "item": self._num_item,
                "aspect": len(aspect_ids)
        })
        #### >>>检查哪些项目没有 aspect-item 边
        user_deg_from_aspect = hetero_g.in_degrees(etype=("aspect", "aspect-user", "user"))
        no_aspect_user = torch.nonzero(user_deg_from_aspect == 0, as_tuple=True)[0]
        item_deg_from_aspect = hetero_g.in_degrees(etype=("aspect", "aspect-item", "item"))
        no_aspect_item = torch.nonzero(item_deg_from_aspect == 0, as_tuple=True)[0]
        print("没有 aspect-user 边的用户数量:", len(no_aspect_user))
        print("示例用户ID:", no_aspect_user[:10])
        print("没有 aspect-item 边的项目数量:", len(no_aspect_item))
        print("示例项目ID:", no_aspect_item[:10])
        #### <<<
        hetero_g.nodes["user"].data["mask_wo_aspect"] = (hetero_g.in_degrees(etype= "aspect-user") > 0).float()
        hetero_g.nodes["item"].data['mask_wo_aspect'] = (hetero_g.in_degrees(etype= "aspect-item") > 0).float()


        hetero_g.nodes["user"].data['aspect_count'] = torch.tensor(user_vectors_list, dtype=torch.float32)
        hetero_g.nodes["item"].data['aspect_count'] = torch.tensor(item_vectors_list, dtype=torch.float32)

        ratings = self.train_rating_values[:len(self.train_rating_values)//2]
        ratings = [x-1 for x in ratings]
        hetero_g.edges['user-item'].data['review_feat'] = review_feat_list[:int(len(review_feat_list) / 2)]
        hetero_g.edges['user-item'].data['rating'] = torch.tensor(ratings).int()
        hetero_g.edges['item-user'].data['review_feat'] = review_feat_list[:int(len(review_feat_list) / 2)]
        hetero_g.edges['item-user'].data['rating'] = torch.tensor(ratings).int()

        hetero_g.edges['user-item-1'].data['review_feat'] = row_col_dict[1][2]
        hetero_g.edges['user-item-2'].data['review_feat'] = row_col_dict[2][2]
        hetero_g.edges['user-item-3'].data['review_feat'] = row_col_dict[3][2]
        hetero_g.edges['user-item-4'].data['review_feat'] = row_col_dict[4][2]
        hetero_g.edges['user-item-5'].data['review_feat'] = row_col_dict[5][2]
        hetero_g.edges['item-user-1'].data['review_feat'] = row_col_dict[1][2]
        hetero_g.edges['item-user-2'].data['review_feat'] = row_col_dict[2][2]
        hetero_g.edges['item-user-3'].data['review_feat'] = row_col_dict[3][2]
        hetero_g.edges['item-user-4'].data['review_feat'] = row_col_dict[4][2]
        hetero_g.edges['item-user-5'].data['review_feat'] = row_col_dict[5][2]
        hetero_g.nodes["user"].data['cui1'] = _calc_norm(hetero_g.out_degrees(etype='user-item-1'), 0.5)
        hetero_g.nodes["item"].data['ciu1'] = _calc_norm(hetero_g.out_degrees(etype='item-user-1'), 0.5)
        hetero_g.nodes["user"].data['cui2'] = _calc_norm(hetero_g.out_degrees(etype='user-item-2'), 0.5)
        hetero_g.nodes["item"].data['ciu2'] = _calc_norm(hetero_g.out_degrees(etype='item-user-2'), 0.5)
        hetero_g.nodes["user"].data['cui3'] = _calc_norm(hetero_g.out_degrees(etype='user-item-3'), 0.5)
        hetero_g.nodes["item"].data['ciu3'] = _calc_norm(hetero_g.out_degrees(etype='item-user-3'), 0.5)
        hetero_g.nodes["user"].data['cui4'] = _calc_norm(hetero_g.out_degrees(etype='user-item-4'), 0.5)
        hetero_g.nodes["item"].data['ciu4'] = _calc_norm(hetero_g.out_degrees(etype='item-user-5'), 0.5)
        hetero_g.nodes["user"].data['cui5'] = _calc_norm(hetero_g.out_degrees(etype='user-item-5'), 0.5)
        hetero_g.nodes["item"].data['ciu5'] = _calc_norm(hetero_g.out_degrees(etype='item-user-5'), 0.5)



        hetero_g.edges['aspect-user'].data['sentiment'] = torch.tensor(user_sentiment_list, dtype=torch.int64)
        hetero_g.edges['aspect-item'].data['sentiment'] = torch.tensor(item_sentiment_list, dtype=torch.int64)
        hetero_g.nodes["user"].data['cui'] = _calc_norm(hetero_g.out_degrees(etype='user-item'), 0.5)
        hetero_g.nodes["user"].data['cau'] = _calc_norm(hetero_g.in_degrees(etype='aspect-user'), 0.5)
        hetero_g.nodes["item"].data['ciu'] = _calc_norm(hetero_g.out_degrees(etype='item-user'), 0.5)
        hetero_g.nodes["item"].data['cai'] = _calc_norm(hetero_g.in_degrees(etype='aspect-item'), 0.5)
        hetero_g.nodes["aspect"].data['cau'] = _calc_norm(hetero_g.out_degrees(etype='aspect-user'), 0.5)
        hetero_g.nodes["aspect"].data['fe'] = torch.stack(aspect_emb_list).to(torch.float32)
        hetero_g.nodes["aspect"].data['cai'] = _calc_norm(hetero_g.out_degrees(etype='aspect-item'), 0.5)
        return hetero_g

        # graph_dict["single"] = dgl.graph((rating_row, rating_col), num_nodes=self._num_user + self._num_item)
        # graph_dict["single"].edata['review_feat'] = review_feat_list
        # graph_dict["single"].edata['score'] = torch.tensor(self.train_rating_values).int()

        c = []
        for r_1 in self.possible_rating_values.tolist():
            c.append(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)
        # c_sum_mean = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 1)

        for r_1 in self.possible_rating_values.tolist() + ["single"]:
            graph_dict[str(r_1)].ndata['ci'] = c_sum
            if r_1 != "single":
                graph_dict[str(r_1)].ndata['c_mask'] = torch.where(graph_dict[str(r_1)].ndata['ci_r'].eq(0), 0, 1)  # 等于0的是0
            # graph_dict[str(r_1)].ndata['ci_mean'] = c_sum_mean

        return graph_dict

    def _train_data(self, batch_size=1024):

        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        # np.random.shuffle(idx) ################################   @ss
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size if i != n_batch - 1 else len(self.train_rating_values)
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

        return u_list, i_list, r_list

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list


def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.model_short_name = 'RGC'
    args.dataset_name = dataset_name
    args.dataset_path = f'../data/{dataset_name}/{dataset_name}.json'
    args.emb_size = 64
    args.emb_dim = 64
    args.origin_emb_dim = 60

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    # args.batch_size = 271466
    args.batch_size = 1111271466

    return args


gloabl_dropout = 0.5

global_review_size = 64


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        nn.init.xavier_uniform_(self.w.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss

    # def measure_sim(self, x, y):
    #     if len(y.shape) > len(x.shape):
    #         _l = y.shape[1]
    #         _x = x @ self.w
    #         _x = _x.unsqueeze(1)
    #         return (_x * y).sum(-1)
    #
    #     else:
    #         return (x @ self.w * y_neg).sum(-1)


class GCN_interaction(nn.Module):
    def __init__(self, params):
        super(GCN_interaction, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')
        self.review_w_2 = nn.Linear(64, global_review_size, bias=False, device='cuda:0')
        self.feature2 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.feature3 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))

    def forward(self, g, feature, is_training=False, freeze=" "):
        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])
        g.update_all(lambda edges: {
            'm': (torch.cat([edges.data['r']], -1)) * self.dropout(edges.src['ci_r'])},
                     fn.sum(msg='m', out='h_1'))
        rst_re = g.dstdata['h_1'] * g.dstdata['ci_r']

        # if 1:#not is_training:
        #     g.srcdata['h_r'] = torch.ones_like(g.srcdata['h_r']) #* torch.mean(g.srcdata['h_r'],0).unsqueeze(0)
        if not is_training:
            g.edata['r_fe'] = self.review_w_2(g.edata['review_feat'])
            u, i = torch.split(self.feature2, [self.num_user, self.num_item], 0)
            if 'u' in freeze:
                u = torch.zeros_like(u)
            if "i" in freeze:
                i = torch.zeros_like(i)
            if "r" in freeze:
                g.edata['r_fe'] = torch.zeros_like(g.edata['r_fe'])
            g.srcdata['h_r_2'] = torch.cat([u, i], 0)
        else:
            g.edata['r_fe'] = self.review_w_2(g.edata['review_feat'])
            g.srcdata['h_r_2'] = self.feature2
        # g.srcdata['h_r_2'] = self.feature2
        g.update_all(lambda edges: {
            'm': (torch.cat([edges.src['h_r_2'] + edges.data['r_fe']], -1)) * self.dropout(edges.src['ci_r'])},
                     fn.sum(msg='m', out='h_2'))
        rst = g.dstdata['h_2'] * g.dstdata['ci_r']

        g.srcdata['h_r_3'] = self.feature3
        g.update_all(lambda edges: {
            'm': (torch.cat([edges.src['h_r_3']], -1)) * self.dropout(edges.src['ci_r'])},
                     fn.sum(msg='m', out='h_3'))
        rst_id = g.dstdata['h_3'] * g.dstdata['ci_r']

        return rst, rst_re, rst_id


class GCN_interaction_all(nn.Module):
    def __init__(self, params):
        super(GCN_interaction_all, self).__init__()
        self.num_users = params.num_users
        self.num_items = params.num_items
        self.num_user = params.num_users
        self.num_item = params.num_items
        self.dropout7 = nn.Dropout(0.7)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout0 = nn.Dropout(0)
        self.review_1 = nn.Linear(64, global_review_size, bias=False)
        self.review_2 = nn.Linear(64, global_review_size, bias=False)
        self.rating_1 = nn.Embedding(5, 64)
        self.rating_2 = nn.Embedding(5, 64)
        self.rating_3 = nn.Embedding(5, 64)
        self.rating_4 = nn.Embedding(5, 64)
        self.rating_5 = nn.Embedding(5, 64)
        self.rating_6 = nn.Embedding(5, 64)
        self.f_aspect = nn.Linear(64, global_review_size, bias=False)
        self.f_aspect_2 = nn.Linear(64, global_review_size, bias=False)
        self.f_aspect_count_user = nn.Linear(2048, global_review_size, bias=True)
        self.f_aspect_count_item = nn.Linear(2048, global_review_size, bias=True)
        self.user_sentiment = nn.Embedding(3, params.emb_dim)
        self.item_sentiment = nn.Embedding(3, params.emb_dim)
        self.feature1 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.feature2 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.feature3 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.weight = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
            for r in [1,2,3,4,5]
        })
        self.f = 0

    def forward(self, g, feature, is_training=False, freeze=" "):
        # if self.f == 0:
        #     g.update_all(lambda edges: {
        #         'm': (torch.cat([edges.data['review_feat']], -1) ) },
        #                  fn.mean(msg='m', out='h'))
        #     g.srcdata['h_re'] = g.dstdata['h'] #* g.dstdata['ci']

        g.nodes["user"].data["fe"], g.nodes["item"].data["fe"] = torch.split(self.feature1, [self.num_users, self.num_items], dim=0)
        g.nodes["user"].data["fe2"], g.nodes["item"].data["fe2"] = torch.split(self.feature2, [self.num_users, self.num_items], dim=0)
        g.nodes["user"].data["fe3"], g.nodes["item"].data["fe3"] = torch.split(self.feature3,[self.num_users, self.num_items], dim=0)

        for i in [1,2,3,4,5]:
            g.nodes["user"].data["fe-"+str(i)], g.nodes["item"].data["fe-"+str(i)] = torch.split(self.weight[str(i)],
                                                                             [self.num_users, self.num_items], dim=0)
            g.edges[f'user-item-{i}'].data['re'] = self.review_1(g.edges[f'user-item-{i}'].data['review_feat'])
            g.edges[f'item-user-{i}'].data['re'] = self.review_2(g.edges[f'item-user-{i}'].data['review_feat'])

        g.edges['aspect-user'].data['sentiment_fe'] = self.user_sentiment(g.edges['aspect-user'].data['sentiment'])
        g.edges['aspect-item'].data['sentiment_fe'] = self.item_sentiment(g.edges['aspect-item'].data['sentiment'])
        g.edges['user-item'].data['rating_fe1'] = self.rating_1(g.edges['user-item'].data['rating'])
        g.edges['item-user'].data['rating_fe2'] = self.rating_2(g.edges['item-user'].data['rating'])
        g.edges['user-item'].data['rating_fe3'] = self.rating_3(g.edges['user-item'].data['rating'])
        g.edges['item-user'].data['rating_fe4'] = self.rating_4(g.edges['item-user'].data['rating'])
        g.edges['user-item'].data['rating_fe5'] = self.rating_5(g.edges['user-item'].data['rating'])
        g.edges['item-user'].data['rating_fe6'] = self.rating_6(g.edges['item-user'].data['rating'])
        g.edges['user-item'].data['re'] = self.review_1(g.edges['user-item'].data['review_feat'])
        g.edges['item-user'].data['re'] = self.review_2(g.edges['item-user'].data['review_feat'])
        g.nodes["aspect"].data["f_fe"] = self.f_aspect(g.nodes["aspect"].data["fe"])
        g.nodes["aspect"].data["f_fe_2"] = self.f_aspect_2(g.nodes["aspect"].data["fe"])

        #分支1 三元图
        funcs = {
            'aspect-user': (
                lambda edges: {
                    'm': (edges.src['f_fe']) * self.dropout3(
                        edges.src['cau'])},
                fn.sum(msg='m', out='h')),
            'aspect-item': (
                lambda edges: {
                    'm': (edges.src['f_fe_2']) * self.dropout3(
                        edges.src['cai'])},
                fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs, "stack")
        mask = g.nodes['user'].data['mask_wo_aspect'].unsqueeze(-1)
        h = g.nodes['user'].data['h'][:, 0, :]
        cau = g.nodes['user'].data['cau']
        fe2 = g.nodes['user'].data['fe2']
        g.nodes['user'].data['from_a'] = mask * (h * cau) + (1 - mask) * fe2

        mask = g.nodes['item'].data['mask_wo_aspect'].unsqueeze(-1)
        h = g.nodes['item'].data['h'][:, 0, :]
        cai = g.nodes['item'].data['cai']
        fe2 = g.nodes['item'].data['fe2']
        g.nodes['item'].data['from_a'] = mask * (h * cai) + (1 - mask) * fe2

        funcs = {
            'user-item': (
                lambda edges: {
                    'm': (edges.src['from_a'] ) * torch.sigmoid(
                        edges.data['rating_fe1']) * self.dropout5(edges.src['cui'])},
                fn.sum(msg='m', out='h')),
            'item-user': (
                lambda edges: {
                    'm': (edges.src['from_a'] ) * torch.sigmoid(
                        edges.data['rating_fe2']) * self.dropout5(edges.src['ciu'])},
                fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs, "stack")
        user_fe = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cui']
        item_fe = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['ciu']
        user_item_senti_fe = torch.cat([user_fe, item_fe], 0)


        ############### h_u h_i
        g.nodes["user"].data["aspect_count_fe"] = self.f_aspect_count_user(g.nodes["user"].data["aspect_count"])
        g.nodes["item"].data["aspect_count_fe"] = self.f_aspect_count_item(g.nodes["item"].data["aspect_count"])

        mask = g.nodes['user'].data['mask_wo_aspect'].unsqueeze(-1)
        h = g.nodes["user"].data["aspect_count_fe"]
        fe3 = g.nodes['user'].data['fe3']
        g.nodes['user'].data['aspect_count_fe'] = mask * (h * cau) + (1 - mask) * fe3

        mask = g.nodes['item'].data['mask_wo_aspect'].unsqueeze(-1)
        h = g.nodes["item"].data["aspect_count_fe"]
        fe3 = g.nodes['item'].data['fe3']
        g.nodes['item'].data['aspect_count_fe'] = mask * (h * cai) + (1 - mask) * fe3

        funcs = {
            'user-item': (
                lambda edges: {
                    'm': (edges.src['aspect_count_fe']) * torch.sigmoid(
                        edges.data['rating_fe3']) * self.dropout5(edges.src['cui'])},
                fn.sum(msg='m', out='h')),
            'item-user': (
                lambda edges: {
                    'm': (edges.src['aspect_count_fe']) * torch.sigmoid(
                        edges.data['rating_fe4']) * self.dropout5(edges.src['ciu'])},
                fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs, "stack")
        user_fe = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cui']
        item_fe = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['ciu']
        user_item_count_fe = torch.cat([user_fe, item_fe], 0)
        ############### 交互图
        user_item_fe_list=[]
        for i in [1,2,3,4,5]:
            funcs = {
                f'user-item-{i}': (
                    lambda edges: {
                        'm': (edges.src[f'fe-{i}']  + edges.data[f're'])  * self.dropout7(edges.src[f'cui{i}'])},
                    fn.sum(msg='m', out='h')),
                f'item-user-{i}': (
                    lambda edges: {
                        'm': (edges.src[f'fe-{i}']  + edges.data[f're']) * self.dropout7(edges.src[f'ciu{i}'])},
                    fn.sum(msg='m', out='h')),
            }
            g.multi_update_all(funcs, "stack")
            user_fe = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data[f'cui{i}']
            item_fe = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data[f'ciu{i}']
            user_item_fe_list.append(torch.cat([user_fe, item_fe], 0))
        user_item_fe5=torch.cat(user_item_fe_list, dim=-1)

        ############# 交互 评论图
        user_item_re_list = []
        for i in [1, 2, 3, 4, 5]:
            funcs = {
                f'user-item-{i}': (
                    lambda edges: {
                        'm': (edges.data[f're']) * self.dropout7(edges.src[f'cui{i}'])},
                    fn.sum(msg='m', out='h_r')),
                f'item-user-{i}': (
                    lambda edges: {
                        'm': ( edges.data[f're']) * self.dropout7(edges.src[f'ciu{i}'])},
                    fn.sum(msg='m', out='h_r')),
            }
            g.multi_update_all(funcs, "stack")
            user_fe = g.nodes['user'].data['h_r'][:, 0, :] * g.nodes['user'].data[f'cui{i}']
            item_fe = g.nodes['item'].data['h_r'][:, 0, :] * g.nodes['item'].data[f'ciu{i}']
            user_item_re_list.append(torch.cat([user_fe, item_fe], 0))
        user_item_re = torch.cat(user_item_re_list, dim=-1)


        ###
        funcs = {
            f'user-item': (
                lambda edges: {
                    'm': (edges.src[f'fe'] * torch.sigmoid(edges.data[f'rating_fe5'])) * self.dropout7(edges.src[f'cui'])},
                fn.sum(msg='m', out='h')),
            f'item-user': (
                lambda edges: {
                    'm': (edges.src[f'fe'] * torch.sigmoid(edges.data[f'rating_fe6'])) * self.dropout7(edges.src[f'ciu'])},
                fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs, "stack")
        user_fe = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cui']
        item_fe = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['ciu']
        user_item_fe = torch.cat([user_fe, item_fe],0)


        return user_item_senti_fe,user_item_count_fe,user_item_fe,user_item_fe5, user_item_re


class MLP(nn.Module):
    def __init__(self, params, input_num, output_num, dropout, bias=True):
        super(MLP, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items
        if bias:
            self.dropout = nn.Dropout(dropout)
            self.fc_user = nn.Linear(global_review_size * input_num, global_review_size * output_num)
            self.fc_item = nn.Linear(global_review_size * input_num, global_review_size * output_num)
        else:
            self.dropout = nn.Dropout(dropout)
            self.fc_user = nn.Linear(global_review_size * input_num, global_review_size * output_num, bias=False)
            self.fc_item = nn.Linear(global_review_size * input_num, global_review_size * output_num, bias=False)

    def forward(self, feature):
        user_feat, item_feat = torch.split(feature, [self.num_user, self.num_item], dim=0)
        user_feat = self.dropout(user_feat)
        u_feat = self.fc_user(user_feat)
        item_feat = self.dropout(item_feat)
        i_feat = self.fc_item(item_feat)
        feat = torch.cat([u_feat, i_feat], dim=0)
        return feat


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items
        # self.weight = nn.ParameterDict({
        #     str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
        #     for r in ["single"]
        # })

        # self.encoder_interaction = nn.ModuleDict({
        #     str(i): GCN_interaction(params) for i in [1, 2, 3, 4, 5]
        # })
        self.encoder_interaction_single = GCN_interaction_all(params)
        # self.encoder_interaction_5_single = GCN_interaction_all()

        # self.mlp_5 = MLP(params, 5 * 1,1 * 1,0.7)
        # self.mlp_5_re = MLP(params, 5 * 1, 1 * 1,0.7 )
        # self.mlp_5_id = MLP(params, 5, 1, 0.7)
        self.mlp_single = MLP(params, 1 * 1, 1 * 1, 0.3, bias=True)
        self.mlp_single_1 = MLP(params, 1 * 1, 1 * 1, 0.3, bias=True)
        self.mlp_single_2 = MLP(params, 1, 1, 0.3, bias=True)
        self.mlp_single_3 = MLP(params, 1, 1, 0.3, bias=True)
        self.mlp_single_4 = MLP(params, 5, 5, 0.3, bias=True)
        self.mlp_single_5 = MLP(params, 5, 5, 0.3, bias=True)

        self.predictor_interaction_single = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 5, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5, 5, bias=False),
        )
        self.predictor_interaction_single_re = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 5, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5, 5, bias=False),
        )
        self.predictor_interaction_single_id = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 5, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5, 5, bias=False),
        )
        self.predictor_interaction_single_4 = nn.Sequential(
            nn.Linear(global_review_size * 5, global_review_size * 5, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5, 5, bias=False),
        )

        self.predictor_interaction_single_5 = nn.Sequential(
            nn.Linear(global_review_size * 5, global_review_size * 5, bias=False),
            nn.ReLU(),
            nn.Linear(global_review_size * 5, 5, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sce_criterion(self, x, y, alpha=1, tip_rate=0):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        if tip_rate != 0:
            loss = self.loss_function(loss, tip_rate)
            return loss
        loss = loss.mean()
        return loss

    def l2_norm_loss(self, x, y):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
        l2_norm_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
        return l2_norm_loss

    def get_loss_sim_inf(self, x, y):
        return torch.mean(torch.cosine_similarity(x, y, dim=-1))

    def cos_square_dis(self, x, y):
        # x: (batch_size, embedding_dim), y: (batch_size, embedding_dim)
        # 对于每一个样本 sqrt((xy)^2 / (x^2 * y^2))
        x_square = torch.sum(x ** 2, dim=-1)
        y_square = torch.sum(y ** 2, dim=-1)
        xy_square = torch.sum(x * y, dim=-1) ** 2
        cov = torch.mean(torch.sqrt(xy_square / (x_square * y_square + 1)))
        return cov

    def forward(self, enc_graph_dict, users, items, is_training=False, freeze=" "):

        # ---------- single --------------------------------------------------------------------------------------
        user_item_senti_fe, user_item_count_fe, user_item_fe, user_item_fe5, user_item_fe5_re = self.encoder_interaction_single(
            enc_graph_dict, 0, is_training, freeze)
        feat_single = self.mlp_single_1(user_item_senti_fe)
        user_embeddings_single, item_embeddings_single = feat_single[users], feat_single[items]
        pred_ratings_senti = self.predictor_interaction_single(
            torch.cat([user_embeddings_single * item_embeddings_single], -1))
        # ------- single re ------------------------------------------
        feat_single_re = self.mlp_single_2(user_item_count_fe)  # self.mlp_single_re(rst_re_single)  #rst_re_single#
        user_embeddings_single_re, item_embeddings_single_re = feat_single_re[users], feat_single_re[items]
        pred_ratings_count = self.predictor_interaction_single_re(user_embeddings_single_re * item_embeddings_single_re)
        # ------- single id ------------------------------------------
        feat_single_id = self.mlp_single_3(user_item_fe)  # self.mlp_single_id(rst_id_single)
        user_embeddings_single_id, item_embeddings_single_id = feat_single_id[users], feat_single_id[items]
        pred_ratings_single = self.predictor_interaction_single_id(
            user_embeddings_single_id * item_embeddings_single_id)
        #
        feat_single_5 = self.mlp_single_4(user_item_fe5)  # self.mlp_single_id(rst_id_single)
        user_embeddings_single_id, item_embeddings_single_id = feat_single_5[users], feat_single_5[items]
        pred_ratings_single5 = self.predictor_interaction_single_4(
            user_embeddings_single_id * item_embeddings_single_id)

        feat_single_5_re = self.mlp_single_4(user_item_fe5_re)  # self.mlp_single_id(rst_id_single)
        user_embeddings_single_re, item_embeddings_single_re = feat_single_5_re[users], feat_single_5_re[items]
        pred_ratings_single5_re = self.predictor_interaction_single_4(
            user_embeddings_single_re * item_embeddings_single_re)
        # ------- single rating ----用不到，忽略--------------------------------------
        # feat_single_ra = self.mlp_single(user_item_fe)  # self.mlp_single_ra(rst_ra_single)
        # user_embeddings_single_ra, item_embeddings_single_ra = feat_single_ra[users], feat_single_ra[items]
        # pred_ratings_single_ra = self.predictor_interaction_single_ra(user_embeddings_single_ra * item_embeddings_single_ra)
        # ------------------------------------------------------------------------------------------------------------
        # loss_kd_feat = self.sce_criterion(rst_ra_single, rst_re_single) + \
        #                self.sce_criterion(rst_id_single, rst_re_single) + \
        #                self.sce_criterion(rst_ra_single, rst_id_single)
        # get_loss_sim_inf
        # loss_kd_feat = self.get_loss_sim_inf(user_item_count_fe, user_item_senti_fe)+ \
        #                self.get_loss_sim_inf(user_item_fe, user_item_count_fe) + \
        #                self.get_loss_sim_inf(user_item_senti_fe, user_item_fe)
        loss_kd_feat = self.sce_criterion(user_item_fe, user_item_senti_fe) + \
                       self.sce_criterion(user_item_senti_fe, user_item_count_fe) + \
                       self.sce_criterion(user_item_count_fe, user_item_fe)

        loss_kd_feat_2 = self.sce_criterion(user_item_fe5, user_item_fe5_re)

        # loss_kd_feat=0
        return pred_ratings_single, pred_ratings_senti, pred_ratings_count, (loss_kd_feat), (loss_kd_feat_2), pred_ratings_single5, pred_ratings_single5_re

def evaluate(args, net,  dataset, flag='valid', add=False, epoch=256, beta=1):
    # if epoch < 200:
    #     return 10, 10, 10, 10
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    net.eval()
    with torch.no_grad():
        r_list = r_list.cpu().numpy()
        if epoch <= g_epoch:
            pred_ratings_single, pred_ratings_senti, pred_ratings_count, loss_kd_feat, loss_kd_feat_2, pred_ratings_single5, pred_ratings_single5_re \
                = net(dataset.train_enc_graph, u_list, i_list, freeze=" ")  # 冻结用户
        else:
            pred_ratings, pred_ratings_review, _ = net(dataset.train_enc_graph_updated, u_list, i_list)

        pred_soft = torch.softmax((1. * pred_ratings_single + 0.2 * pred_ratings_senti + pred_ratings_count + pred_ratings_single5 * 1. + pred_ratings_single5_re)/4.2, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse = ((real_pred_ratings - r_list) ** 2.).mean()
        mae = (np.abs(real_pred_ratings - r_list)).mean()

        pred_soft = torch.softmax(pred_ratings_single, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_single = ((real_pred_ratings - r_list) ** 2.).mean()

        pred_soft = torch.softmax(pred_ratings_senti, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_senti = ((real_pred_ratings - r_list) ** 2.).mean()

        pred_soft = torch.softmax(pred_ratings_count, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_count = ((real_pred_ratings - r_list) ** 2.).mean()

        pred_soft = torch.softmax(pred_ratings_single5, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_single5 = ((real_pred_ratings - r_list) ** 2.).mean()

        pred_soft = torch.softmax(pred_ratings_single5_re, -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_single5_re = ((real_pred_ratings - r_list) ** 2.).mean()

        pred_soft_test = (torch.softmax(pred_ratings_single, -1) + 0.2 * torch.softmax(pred_ratings_senti, -1) +
                          torch.softmax(pred_ratings_count, -1) + torch.softmax(pred_ratings_single5, -1) +
                          torch.softmax(pred_ratings_single5_re, -1))/4.2
        real_pred_ratings = (pred_soft_test * nd_possible_rating_values.view(1, -1)).sum(dim=1).cpu().numpy()
        mse_test = ((real_pred_ratings - r_list) ** 2.).mean()

    return mse, mse_single, mse_count, mse_senti, mse_single5, mse_single5_re, mse_test, mae



g_epoch = 1000

class CosineDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def ss(self, index):
        mm = 300

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        scaling_factor = mm / 10
        y = sigmoid((index - mm / 2) / scaling_factor)
        return -y
        if index < 100:
            return -max(y, 0)
        else:
            return -min(y, 1)

    def get_value(self, i):
        return self.ss(i)
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value

def sce_criterion(x, y, alpha=1, tip_rate=0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    # if tip_rate != 0:
    #     loss = self.loss_function(loss, tip_rate)
    #     return loss
    loss = loss.mean()
    return loss


def l2_norm_loss(x, y):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
    l2_norm_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
    return l2_norm_loss


def train(params):
    dataset = Data(params.dataset_name,
                   params.dataset_path,
                   params.device,
                   params.emb_size,
                   )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values

    print(
        f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=100)

    rating_loss_net = nn.CrossEntropyLoss()
    rating_loss_no_softmax_net = nn.CrossEntropyLoss()
    rating_loss_NLLLoss_net = nn.NLLLoss()
    # weights = [5, 4, 3,2, 1]
    # class_weights = torch.FloatTensor(weights).to(params.device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    kd_mse_loss = nn.MSELoss()
    kd_l1_loss = nn.L1Loss()
    kd_kl_loss = nn.KLDivLoss(reduction="mean", log_target=True)
    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_test_mse = np.inf
    best_test_mae = np.inf
    no_better_valid = 0
    best_iter = -1
    result = []

    # for r in [1, 2, 3, 4, 5, 'single']:
    #     dataset.train_enc_graph[str(r)] = dataset.train_enc_graph[str(r)].int().to(params.device)
    # print(dataset.train_enc_graph)
    dataset.train_enc_graph = dataset.train_enc_graph.int().to(
        params.device)
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(params.device)

    print("Training and evaluation.")
    u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
    for b in u_list:
        print(b.shape)
    max_batch_size = b.shape[0]
    review_label = torch.cat([torch.ones(max_batch_size), torch.zeros(max_batch_size), torch.zeros(max_batch_size)], 0).cuda()
    for iter_idx in range(1, 1000):
        decay_value = gradient_decay.get_value(iter_idx)
        # print(decay_value)
        net.train()

        train_mse_1 = 0.
        train_mse_2 = 0.
        train_mse_3 = 0.

        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]

            if iter_idx <= g_epoch:
                pred_ratings_single, pred_ratings_senti, pred_ratings_count,  loss_kd_feat , loss_kd_feat_2, pred_ratings_single5, pred_ratings_single5_re \
                    = net(dataset.train_enc_graph, batch_user, batch_item, is_training=True)
            else:
                pred_ratings, pred_ratings_review, loss_kd_feat = net(dataset.train_enc_graph_updated, batch_user, batch_item)

            real_pred_ratings_single = ((torch.softmax(pred_ratings_single, dim=1)) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
            real_pred_ratings_senti = ((torch.softmax(pred_ratings_senti, dim=1)) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
            real_pred_ratings_count = ((torch.softmax(pred_ratings_count, dim=1)) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

            loss_single = rating_loss_net(pred_ratings_single, batch_rating).mean()
            loss_single5 = rating_loss_net(pred_ratings_single5, batch_rating).mean()
            loss_single5_re = rating_loss_net(pred_ratings_single5_re, batch_rating).mean()
            loss_count = rating_loss_net(pred_ratings_count, batch_rating).mean()
            loss_senti = rating_loss_net(pred_ratings_senti, batch_rating).mean()
            # loss_sum = rating_loss_net((pred_ratings_single + pred_ratings_senti * 0.5 + pred_ratings_count * 0.5 + pred_ratings_single5+pred_ratings_single5_re)/4, batch_rating).mean()


            re_id, re, id, sig5, sig5_re = pred_ratings_single, pred_ratings_senti, pred_ratings_count,pred_ratings_single5, pred_ratings_single5_re
            ave=(re_id + re * 0.2 + id + sig5 * 1. + sig5_re) / 4.2
            ave= ave.detach()
            loss_kd_s_id_re = (kd_mse_loss(torch.softmax(ave, dim=-1), torch.softmax(id, dim=-1)) + \
                              kd_mse_loss(torch.softmax(ave, dim=-1), torch.softmax(re, dim=-1)) + \
                              kd_mse_loss(torch.softmax(ave, dim=-1), torch.softmax(re_id, dim=-1)) + \
                              kd_mse_loss(torch.softmax(ave, dim=-1), torch.softmax(sig5, dim=-1)) +
                               kd_mse_loss(torch.softmax(ave, dim=-1), torch.softmax(sig5_re, dim=-1)))

            # loss_kd  = kd_mse_loss(torch.softmax(pred_ratings_single5, dim=-1), torch.softmax(pred_ratings_single5_re, dim=-1))

            loss_final = (loss_single + loss_count + loss_senti + loss_single5 + loss_single5_re)*1 +  (0.6*loss_kd_s_id_re +0.4*loss_kd_feat+loss_kd_feat_2*0)*1

            # loss_final = (1 - decay_value)*loss_final

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            train_mse_3 += ((real_pred_ratings_single - batch_rating - 1) ** 2).sum()
            train_mse_1 += ((real_pred_ratings_senti - batch_rating - 1) ** 2).sum()
            train_mse_2 += ((real_pred_ratings_count - batch_rating - 1) ** 2).sum()

        mse, mse_single, mse_count, mse_senti, mse_single5, mse_single5_re, mse_test, mae= evaluate(args=params, net=net, dataset=dataset, flag='test', add=False, epoch=iter_idx, beta=1)

        if mse < best_test_mse:
            best_test_mse = mse
            # final_test_mae = mae
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if iter_idx > 1000:
                break
            if no_better_valid > params.train_early_stopping_patience :
                print("Early stopping threshold reached. Stop training.")
                break

        if mae < best_test_mae:
            best_test_mae = mae

        print(
            f'Epoch {iter_idx}, {loss_senti:.4f}, {loss_count:.4f}, {loss_single:.4f}, {loss_single5:.4f}, {loss_single5_re:.4f}, '
            f'Test_MSE_single={mse_single:.4f}, Test_MSE_senti={mse_senti:.4f}, Test_MSE_count={mse_count:.4f}, Test_MSE_single5={mse_single5:.4f}, Test_MSE_single5_re={mse_single5_re:.4f}, Test_MSE_={mse_test:.4f},  -->> Final_Test_MSE={mse:.4f} -->> Final_Test_MAE={mae:.4f}')
        result.append(mse)
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_mse:.4f}, Best Test MAE={best_test_mae:.4f}')

    # with open('ablation_distangle_id_origin.pickle', 'wb') as file:
    #     pickle.dump(result, file)


if __name__ == '__main__':
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    config_args = config()

    train(config_args)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
