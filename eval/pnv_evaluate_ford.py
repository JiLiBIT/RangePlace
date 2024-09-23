# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import random
import tqdm
import time
from torchvision import transforms as transforms
from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader, PNVRangeImageLoader

class RangeShift:
    def __init__(self, r=0.08333):
        self.r = r

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxCxHxW array, original batch
            Return:
              BxCxHxW array, jittered batch
        """

        channels, height, width = e.shape
        W = width
        kshift = int(self.r * W)
        shifted_range_images_tensor = e.clone()
        shifted_range_images_tensor = torch.roll(shifted_range_images_tensor, shifts=kshift, dims=-1)
        
        return shifted_range_images_tensor
    

def evaluate(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False, degree = 0, rotate=False):
    # Run evaluation on all eval datasets

    # eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
    #                        'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']

    # eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
    #                     'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']
    
    eval_database_files = ['Hioformer_ford_evaluation_database.pickle']

    eval_query_files = ['Hioformer_ford_evaluation_query.pickle']
    
    assert len(eval_database_files) == len(eval_query_files)
    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder +'/' + database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder +'/' + query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets, log=log, show_progress=show_progress,degree = degree, rotate=rotate)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets, log: bool = False,
                     show_progress: bool = False, degree = 0, rotate=False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, set, device, params, degree = degree, rotate=rotate))

    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, set, device, params, degree = rotate, rotate=rotate))

    for m in range(len(database_sets)):
        for n in range(len(query_sets)):
            if m != n: continue
            pair_recall, pair_opr = get_recall(m, n, database_embeddings, query_embeddings, query_sets,
                                               database_sets, log=log)
            print(str(m),pair_recall)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


def get_latent_vectors(model, set, device, params: TrainingParams, degree=0, rotate=False):
    # Adapted from original PointNetVLAD code

    if params.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings
    ri_loader = PNVRangeImageLoader()
    pc_loader = PNVPointCloudLoader()

    model.eval()
    embeddings = None
    elapsed_times = []
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(params.dataset_folder, set[elem_ndx]["query"])
        # pc = ri_loader(pc_file_path)
        print("111",pc_file_path)
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)
        if rotate == True:
            t = [RangeShift(r=degree/360.0)
                 ]
            transform = transforms.Compose(t)

            pc = transform(pc)
        start_time = time.time()
        embedding = compute_embedding(model, pc, device, params)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding
    average_time = sum(1000*elapsed_times) / len(elapsed_times)
    print("Average computation time: {} ms".format(average_time))

    return embeddings


def compute_embedding(model, pc, device, params: TrainingParams):
    if params.model_params.quantizer is not None:
        coords, _ = params.model_params.quantizer(pc)
    else:
        coords = pc
    with torch.no_grad():
        # bcoords = ME.utils.batched_coordinates([coords])
        # feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        # batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
        batch = coords.unsqueeze(0).to(device)

        # temp = coords
        # minibatch = torch.stack(temp)
        # batch.append(minibatch)

        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding

def check(idx, idx2, query_sets, database_sets,n, m):
    rx = query_sets[n][idx]['easting']
    ry = query_sets[n][idx]['northing']

    tx = database_sets[m][idx2]['easting']
    ty = database_sets[m][idx2]['northing']

    if (rx-tx)*(rx-tx) + (ry-ty)*(ry-ty) <= 5*5:
        return True
    else:
        return False

def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)
    num_evaluated = 0
    top1_similarity_score = []
    for_plot = []       # for plot
    elapsed_times = []
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = query_sets[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        qname = query_sets[n][i]['query']
        num_evaluated += 1
        start_time = time.time()
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

        for_plot.append(query_sets[n][i]['easting'])
        for_plot.append(query_sets[n][i]['northing'])
        flag = False
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if i == indices[0][j]:
                    continue
                if check(i, indices[0][j],query_sets, database_sets, n, m) is False:
                    continue
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                for_plot.append(j)
                flag = True
                break
        if flag is False:
            for_plot.append(25)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    average_time = sum(1000*elapsed_times) / len(elapsed_times)
    print("Average searching time: {} ms".format(average_time))
    print("one_percent_retrieved: {}".format(one_percent_retrieved))
    print("num_evaluated: {}".format(num_evaluated))
    print("recall: {}".format(recall))
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--rotate', dest='rotate', action='store_true')
    parser.set_defaults(rotate=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
        
        # checkpoint = torch.load(args.weights)
        # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print("Rotation invariant test: {}".format(args.rotate))
    if args.rotate == True:
        for degree in range(90,91,30):
            stats = evaluate(model, device, params, args.log, show_progress=True, degree = 0, rotate=args.rotate)
            print_eval_stats(stats)
            model_params_name = os.path.split(params.model_params.model_params_path)[1]
            config_name = os.path.split(params.params_path)[1]
            model_name = os.path.split(args.weights)[1]
            model_name = os.path.splitext(model_name)[0]
            prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
            pnv_write_eval_stats("pnv_experiment_results" + str(degree) + ".txt", prefix, stats)

    else:
        stats = evaluate(model, device, params, args.log, show_progress=True, degree = 0, rotate=args.rotate)
        print_eval_stats(stats)

        # Save results to the text file
        model_params_name = os.path.split(params.model_params.model_params_path)[1]
        config_name = os.path.split(params.params_path)[1]
        model_name = os.path.split(args.weights)[1]
        model_name = os.path.splitext(model_name)[0]
        prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
        pnv_write_eval_stats("pnv_experiment_results.txt", prefix, stats)

