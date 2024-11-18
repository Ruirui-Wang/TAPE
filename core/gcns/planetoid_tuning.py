import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import time
import os.path as osp
from hl_gnn_planetoid.logger import Logger
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from data_utils.load import load_data_lp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from graphgps.utility.utils import (
    set_cfg, 
    get_git_repo_root_path, 
    custom_set_run_dir, 
    set_printing, 
    run_loop_settings, 
    create_logger,
    config_device,
    save_run_results_to_csv
)

from hl_gnn_planetoid.utils import *
from data_utils.lcc import *
from hl_gnn_planetoid.model import *
from hl_gnn_planetoid.metrics import do_csv
from hl_gnn_planetoid.visualization import visualization_geom_fig, visualization_beta, visualization, \
                                        visualization_epochs
from torch.utils.tensorboard import SummaryWriter

def train(model, predictor, data, split_edge, optimizer, batch_size, writer, epoch):
    predictor.train()
    model.train()
    
    pos_train_edge = split_edge['train']['edge'].to(data.edge_index.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(data.x, data.adj_t, data.edge_weight)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=data.x.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    avg_loss = total_loss / total_examples
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Log the training loss

    return avg_loss

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

def evaluate_mrr(pos_val_pred, neg_val_pred):
                 
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

def acc(pos_pred, neg_pred):
        hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

        # Initialize predictions with zeros and set ones where condition is met
        y_pred = torch.zeros_like(pos_pred)
        y_pred[pos_pred >= hard_thres] = 1

        # Do the same for negative predictions
        neg_y_pred = torch.ones_like(neg_pred)
        neg_y_pred[neg_pred <= hard_thres] = 0

        # Concatenate the positive and negative predictions
        y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

        # Initialize ground truth labels
        pos_y = torch.ones_like(pos_pred)
        neg_y = torch.zeros_like(neg_pred)
        y = torch.cat([pos_y, neg_y], dim=0)
        y_logits = torch.cat([pos_pred, neg_pred], dim=0)
        # Calculate accuracy    
        return (y == y_pred).float().mean().item()
    
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

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, writer, epoch):
    predictor.eval()
    model.eval()
    h = model(data.x, data.adj_t, data.edge_weight)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [1, 3, 10, 20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

        # Log the hits@K values
        writer.add_scalar(f'Accuracy/Train_Hits@{K}', train_hits, epoch)
        writer.add_scalar(f'Accuracy/Valid_Hits@{K}', valid_hits, epoch)
        writer.add_scalar(f'Accuracy/Test_Hits@{K}', test_hits, epoch)
    
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))  
    
    for name in ['MRR', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100']:
        results[name] = (result_mrr_test[name])
        writer.add_scalar(f'Accuracy/Test_{name}', result_mrr_test[name], epoch)
    
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    
    result_auc_test = evaluate_auc(test_pred, test_true)
    for name in ['AUC', 'AP']:
        results[name] = (result_auc_test[name])
        writer.add_scalar(f'Accuracy/Test_{name}',result_auc_test[name], epoch)

    result_acc_test = acc(pos_test_pred, neg_test_pred)
    results['ACC'] = (result_acc_test)
    writer.add_scalar(f'Accuracy/Test_ACC',result_acc_test, epoch)
    
    return results

def main():
    # FILE_PATH = f'{get_git_repo_root_path()}/'
    
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--init', type=str, choices=['SGC', 'RWR', 'KI', 'Random', 'WS', 'Null'], default='KI')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--norm_func', type=str, choices=['gcn_norm', 'col_stochastic_matrix', 'row_stochastic_matrix'], required=True)
    
    args = parser.parse_args()
    print(args)

    # CORA
    # param_grid = {
    #     'lr': [0.0001],#[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    #     'dropout': [0.4], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    #     'hidden_channels': [128, 256, 512, 1024, 2048, 4096, 8192],
    #     'mlp_num_layers': [5],#[2, 3, 4, 5, 6],
    #     'alpha': [0.4]#[0.1, 0.2, 0.3, 0.4, 0.5]
    # }
    
    # PUBMED
    param_grid = {
        'lr': [0.0001],#, 0.0005, 0.001, 0.005, 0.01, 0.05],
        'dropout': [0.6],#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'hidden_channels': [512],#[128, 256, 512, 1024, 2048],#, 4096, 8192],
        'mlp_num_layers': [5],#[2, 3, 4, 5, 6, 7],
        'alpha': [0.2]#[0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    grid = ParameterGrid(param_grid)
    best_result = None
    best_params = None
    for params in grid:
        print('PARAMETERS: ', params)
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        _, _, data = load_data_lp[args.dataset](args.dataset, if_lcc=True, alg_name='HL-GNN')
        
        split_edge = do_edge_split(data, True)
        
        llama_node_features = torch.load(f'hl_gnn_planetoid/node_features/llama_{args.dataset}_saved_node_features.pt', map_location=torch.device('cpu'))
        data.x = llama_node_features
        name = 'llama'
        
        # bert_node_features = torch.load(f'hl_gnn_planetoid/node_features/bert{args.dataset}saved_node_features.pt')#, map_location=torch.device('cpu'))
        # data.x = bert_node_features
        # name = 'bert'
        
        # e5_node_features = torch.load(f'hl_gnn_planetoid/node_features/e5-large{args.dataset}saved_node_features.pt')#, map_location=torch.device('cpu'))
        # data.x = torch.Tensor(e5_node_features)
        # name = 'e5'
        
        # minilm_node_features = torch.load(f'hl_gnn_planetoid/node_features/minilm{args.dataset}saved_node_features.pt')#, map_location=torch.device('cpu'))
        # data.x = torch.Tensor(minilm_node_features)
        # name = 'minilm'
        
        data = T.ToSparseTensor(remove_edge_index=False)(data)
        data = data.to(device)
        print(data)
        args.lr = params['lr']
        args.alpha = params['alpha']
        model = HLGNN(data, args).to(device)
        
        predictor = LinkPredictor(data.num_features, params['hidden_channels'], 1, params['mlp_num_layers'], params['dropout']).to(device)
        para_list = list(model.parameters()) + list(predictor.parameters())
        total_params = sum(p.numel() for param in para_list for p in param)
        total_params_print = f'Total number of model parameters is {total_params}'
        print(total_params_print)
        
        evaluator = Evaluator(name='ogbl-collab')
        
        loggers = {
            'Hits@1': Logger(args.runs, args),
            'Hits@3': Logger(args.runs, args),
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'MRR': Logger(args.runs, args),
            'mrr_hit1': Logger(args.runs, args),
            'mrr_hit3': Logger(args.runs, args), 
            'mrr_hit10': Logger(args.runs, args),
            'mrr_hit20': Logger(args.runs, args),
            'mrr_hit50': Logger(args.runs, args),
            'mrr_hit100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
            'AP': Logger(args.runs, args),
            'ACC': Logger(args.runs, args)
        }
        writer = SummaryWriter()
        beta_values = []
        for run in range(args.runs):
            model.reset_parameters()
            predictor.reset_parameters()
            optimizer = torch.optim.Adam(list(predictor.parameters()) + list(model.parameters()), lr=args.lr)
            
            start_time = time.time()
            for epoch in range(1, 1 + args.epochs):
                loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, writer, epoch)
                
                # Save beta values along with their layer indices on the last epoch
                if epoch % 100 == 0:
                    beta_values_epoch = [(epoch, layer, value.item()) for layer, value in enumerate(model.temp.detach().cpu())]
                    # Save beta values to a file
                    # with open(f'hl_gnn_planetoid/metrics_and_weights/beta_values{name}.txt', 'a') as f:
                    #     f.write(f"hl_gnn_planetoid/Type Heuristic:{args.init}, Dataset: {args.dataset}\n")
                    #     for epoch, layer, value in beta_values_epoch:
                    #         f.write(f'{epoch}\t{layer}\t{value}\n')

                if epoch % args.eval_steps == 0:
                    results = test(model, predictor, data, split_edge, evaluator, args.batch_size, writer, epoch)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        spent_time = time.time() - start_time
                        for key, result in results.items():
                            if key in ['Hits@1', 'Hits@3',
                                    'Hits@10', 'Hits@20', 'Hits@50',
                                    'Hits@100']:
                                train_hits, valid_hits, test_hits = result
                                print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                                writer.add_scalar(f'{key}/Train', train_hits, epoch)
                                writer.add_scalar(f'{key}/Valid', valid_hits, epoch)
                                writer.add_scalar(f'{key}/Test', test_hits, epoch)
                            else:
                                test_hits = result
                                print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Test: {100 * test_hits:.2f}%')
                                writer.add_scalar(f'{key}/Test', test_hits, epoch)
                        print('---')
                        print(f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                        print('---')
                        start_time = time.time()


            for key in loggers.keys():
                if key in ['Hits@1', 'Hits@3','Hits@10', 'Hits@20', 'Hits@50','Hits@100']:
                    loggers[key].print_statistics(run)
                else:
                    loggers[key].print_statistics_others(run)
        
        with open(f'hl_gnn_planetoid/metrics_and_weights/results_{name}.txt', 'a') as f:
            f.write(f"Type Heuristic:{args.init}, Dataset: {args.dataset}, Norm function: {args.norm_func}\n")

        for key in loggers.keys():
            if key in ['Hits@1', 'Hits@3','Hits@10', 'Hits@20', 'Hits@50','Hits@100']:
                
                loggers[key].print_statistics(key=key, emb_name=name)
            else:
                loggers[key].print_statistics_others(key=key, emb_name=name)
        
        
        # visualization_epochs(beta_values, args.dataset)
        # do_csv(f'hl_gnn_planetoid/metrics_and_weights/results_{name}.txt', name)
        writer.close()
        # Update best result if current result is better
        current_result = loggers['MRR'].results[0]  # Assuming 'MRR' is the primary metric
        if best_result is None or current_result > best_result:
            best_result = current_result
            best_params = params

    print(f'Best result: {best_result} with params: {best_params}')
    
if __name__ == "__main__":
    main()
