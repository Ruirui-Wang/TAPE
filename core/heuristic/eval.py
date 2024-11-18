import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# def evaluate_hits(test_pred, labels)
def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        hits = round(hits, 4)

        results[f'Hits@{K}'] = hits

    return results
        


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
                 
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)

    valid_mrr =mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()


    valid_mrr = round(valid_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    
    return results




def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap


    return results


def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}




def eval_hard_negs(pos_pred, neg_pred, k_list):
    """
    Eval on hard negatives
    """
    neg_pred = neg_pred.squeeze(-1)

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_pred >= pos_pred).sum(dim=-1)

    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_pred > pos_pred).sum(dim=-1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    results = {}
    for k in k_list:
        mean_score = (ranking_list <= k).to(torch.float).mean().item()
        results[f'Hits@{k}'] = round(mean_score, 4)

    mean_mrr = 1./ranking_list.to(torch.float)
    results['MRR'] = round(mean_mrr.mean().item(), 4)

    return results



def get_prediction(full_A, use_heuristic, pos_test_edge, neg_test_edge):

    pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge)
    neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge)

    return pos_test_pred, neg_test_pred

# def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
#     k_list = [1, 3, 10, 100]

#     result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
#     result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred )
#     result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)

#     result = {
#         'MRR': (
#             result_mrr_train['MRR'],
#             result_mrr_val['MRR'],
#             result_mrr_test['MRR'],
#         )
#     }
#     for K in [1,3,10, 100]:
#         result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])


#     return result

def get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred):

    k_list  = [1, 3, 10, 20, 50, 100]
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    result = {
        f'Hits@{K}': result_hit_test[f'Hits@{K}']
        for K in [1, 3, 10, 20, 50, 100]
    }
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))

    result['MRR'] = (result_mrr_test['MRR'])
    result['mrr_hit1']  = (result_mrr_test['mrr_hit1'])
    result['mrr_hit3']  = (result_mrr_test['mrr_hit3'])
    result['mrr_hit10']  = (result_mrr_test['mrr_hit10'])
    result['mrr_hit20']  = (result_mrr_test['mrr_hit20'])
    result['mrr_hit50']  = (result_mrr_test['mrr_hit50'])
    result['mrr_hit100']  = (result_mrr_test['mrr_hit100'])
    # print(result_mrr_test['mrr_hit100'])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_test = evaluate_auc(test_pred, test_true)

    result['AUC'] = (result_auc_test['AUC'])
    result['AP'] = (result_auc_test['AP'])
    result['ACC'] = 0
    return result