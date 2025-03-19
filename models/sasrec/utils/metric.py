import numpy as np
from scipy.sparse import csr_matrix


def eval_hit(gt_item, pred_item, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_item, pred_item):
            if len(gt) == 0:
                continue
            tmp = 0
            for p in pred:
                if p in gt:
                    tmp = 1
                    break
            result.append(tmp)
    else:
        tmp = 0
        for pred in pred_item:
            if pred in gt_item:
                tmp = 1
                break
        result.append(tmp)
    return result


def eval_ndcg(gt_item, pred_item, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_item, pred_item):
            if len(gt) == 0:
                continue
            tmp = [0]
            for g in set(gt):
                if g in pred:
                    index = pred.index(g)
                    tmp.append(np.reciprocal(np.log2(index + 2)))
            result.append(max(tmp))
    else:
        tmp = [0]
        for gt in set(gt_item):
            if gt in pred_item:
                index = pred_item.index(gt)
                tmp.append(np.reciprocal(np.log2(index + 2)))
        result.append(max(tmp))
    return result


def eval_map(gt_items, pred_items, k, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_items, pred_items):
            if len(gt) == 0:
                continue
            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(pred):
                if p in gt and p not in pred[:i]:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            result.append(score / min(len(gt), k))
        return result
    else:
        if len(gt_items) == 0:
            return result
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(pred_items):
            if p in gt_items and p not in pred_items[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        result.append(score / min(len(gt_items), k))
    return result


def eval_diversity(user_rec_dict, num_candidates=100, topk=12):
    user2idx = {u: idx for idx, u in enumerate(sorted(list(user_rec_dict.keys())))}
    item2idx = {
        u: idx
        for idx, u in enumerate(
            sorted(set(np.concatenate(list(user_rec_dict.values()))))
        )
    }

    user_indices = []
    item_indices = []
    for user_id, item_id_lst in user_rec_dict.items():
        user_idx = user2idx[user_id]
        for item_id in item_id_lst:
            item_idx = item2idx[item_id]
            user_indices.append(user_idx)
            item_indices.append(item_idx)
    data = np.ones(len(item_indices)).tolist()
    shape = [len(user2idx), num_candidates]
    user_m = csr_matrix(
        (data, (user_indices, item_indices)), shape=tuple(shape), dtype=np.float32
    )

    user_m = user_m.toarray()
    user_sim_m = np.matmul(user_m, user_m.T)
    score = user_sim_m.sum(axis=0).sum()
    score = score - np.diagonal(user_sim_m).sum()

    # user_sim_matrix = user_m.dot(user_m.T)
    # user_sim_matrix.setdiag(np.zeros(len(user_indices)))

    diversity = 1 - (score / len(user2idx) ** 2) / topk
    return diversity


def eval_metric(user_gt_dict, user_rec_dict, user_type_dict, topk=12, get_result=False):
    recent_HR, recent_MAP, inter_HR, inter_MAP, new_HR, new_MAP = [], [], [], [], [], []
    recent_gt_items, recent_recommends = [], []
    inter_gt_items, inter_recommends = [], []
    new_gt_items, new_recommends = [], []
    for u in list(user_gt_dict.keys()):
        if user_type_dict.get(u, -1) == "recent":
            recent_gt_items.append(user_gt_dict[u])
            recent_recommends.append(user_rec_dict[u])
        elif user_type_dict.get(u, -1) == "inter":
            inter_gt_items.append(user_gt_dict[u])
            inter_recommends.append(user_rec_dict[u])
        else:
            new_gt_items.append(user_gt_dict[u])
            new_recommends.append(user_rec_dict[u])
    recent_HR += eval_hit(recent_gt_items, recent_recommends, batch=True)
    recent_MAP += eval_map(recent_gt_items, recent_recommends, topk, batch=True)
    inter_HR += eval_hit(inter_gt_items, inter_recommends, batch=True)
    inter_MAP += eval_map(inter_gt_items, inter_recommends, topk, batch=True)
    new_HR += eval_hit(new_gt_items, new_recommends, batch=True)
    new_MAP += eval_map(new_gt_items, new_recommends, topk, batch=True)

    res = {}
    for k, v in zip(
        ["recent", "inter", "new"],
        [[recent_HR, recent_MAP], [inter_HR, inter_MAP], [new_HR, new_MAP]],
    ):
        num_user = len(v[0])
        res["{}_{}".format(k, "NUM")] = num_user
        if num_user != 0:
            res["{}_{}".format(k, "HIT")] = int(np.sum(v[0]))
            res["{}_{}".format(k, "HR")] = np.mean(v[0])
            res["{}_{}".format(k, "MAP")] = np.mean(v[1])
        else:
            res["{}_{}".format(k, "HIT")] = 0
            res["{}_{}".format(k, "HR")] = 0
            res["{}_{}".format(k, "MAP")] = 0
        global_HR = recent_HR + inter_HR + new_HR
        global_MAP = recent_MAP + inter_MAP + new_MAP
        res["global_NUM"] = len(global_HR)
        res["global_HIT"] = int(np.sum(global_HR))
        res["global_HR"] = np.mean(global_HR)
        res["global_MAP"] = np.mean(global_MAP)

    if get_result:
        return res
    else:
        print(
            "[Rec result] RECENT USER: {}, RECENT_HIT: {}, RECENT_HR: {:.4f}, RECENT_MAP: {:.4f}".format(
                res["recent_NUM"],
                res["recent_HIT"],
                res["recent_HR"],
                res["recent_MAP"],
            )
        )
        print(
            "[Rec result] INTER USER: {}, INTER_HIT: {}, INTER_HR: {:.4f}, INTER_MAP: {:.4f}".format(
                res["inter_NUM"], res["inter_HIT"], res["inter_HR"], res["inter_MAP"]
            )
        )
        print(
            "[Rec result] NEW USER: {}, NEW_HIT: {}, NEW_HR: {:.4f}, NEW_MAP: {:.4f}".format(
                res["new_NUM"], res["new_HIT"], res["new_HR"], res["new_MAP"]
            )
        )
        print(
            "[Rec result] GLOBAL USER: {}, GLOBAL_HIT: {}, GLOBAL_HR: {:.4f}, GLOBAL_MAP: {:.4f}".format(
                res["global_NUM"],
                res["global_HIT"],
                res["global_HR"],
                res["global_MAP"],
            )
        )
