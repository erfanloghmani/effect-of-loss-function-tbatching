'''
This code explains the model trained in jodie.py.
The task is: interaction prediction, i.e., predicting which item will a user interact with?

To explain the model for one epoch:
$ python explain.py --network reddit --model jodie --epoch 49

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.
'''

from library_data import *
from library_models import *
import numpy as np
from itertools import compress
import json

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', default="myket", help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=0, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.')
parser.add_argument('--device', default='cpu', type=str, help='Torch device')
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print "No interaction prediction for %s" % args.network
    sys.exit(0)

# SET GPU
if args.device == 'gpu':
    if args.gpu == -1:
        args.gpu = select_free_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:%s' % args.gpu)
else:
    device = torch.device('cpu')


# LOAD NETWORK
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence,
 feature_sequence,
 y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1
true_labels_ratio = len(y_true) / (sum(y_true) + 1)
print "*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true))

# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated.
All interactions in one timespan are processed simultaneously.
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates.
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500


def explain():
    # INITIALIZE MODEL PARAMETERS
    model = JODIE(args, num_features, num_users, num_items).to(device)
    weight = torch.Tensor([1, true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()

    # INITIALIZE MODEL
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # LOAD THE MODEL
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch, device)
    if train_end_idx != train_end_idx_training:
        sys.exit('Training proportion during training and testing are different. Aborting.')

    # SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

    # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
    item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()

    user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()


def sample_sequence_list(previous_interaction_idxs, k):
    idxs = np.random.randint(0, 2, size=(k, len(previous_interaction_idxs)))
    sample_sequences = []
    for i in range(k):
        sample_sequences.append(list(compress(previous_interaction_idxs, idxs[i, :])))
    return sample_sequences, idxs


def get_explanation_predictions_for_interaction(interaction_idx, k=10):
    model = JODIE(args, num_features, num_users, num_items).to(device)
    weight = torch.Tensor([1, true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()

    # INITIALIZE MODEL
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # LOAD THE MODEL
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch, device)
    if train_end_idx != train_end_idx_training:
        sys.exit('Training proportion during training and testing are different. Aborting.')
    # SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

    # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
    item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()

    user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()

    c_userid = user_sequence_id[interaction_idx]
    itemid = item_sequence_id[interaction_idx]
    feature = feature_sequence[interaction_idx]
    user_timediff = user_timediffs_sequence[interaction_idx]
    item_timediff = item_timediffs_sequence[interaction_idx]
    timestamp = timestamp_sequence[interaction_idx]
    itemid_previous = user_previous_itemid_sequence[interaction_idx]

    previous_interaction_idxs = []
    for i in range(interaction_idx):
        if user_sequence_id[i] == c_userid:
            previous_interaction_idxs.append(i)
    sampled_sequences, idxs = sample_sequence_list(previous_interaction_idxs, k)
    all_predictions = []
    for c, seq in enumerate(sampled_sequences):
        print(c, seq)
        # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
        item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
        item_embeddings_static = item_embeddings_static.clone()

        user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
        user_embeddings_static = user_embeddings_static.clone()
        for i, pidx in enumerate(seq):
            userid = num_users - 1
            itemid = item_sequence_id[pidx]
            feature = feature_sequence[pidx]
            user_timediff = timestamp_sequence[pidx] - timestamp_sequence[seq[i - 1]] if i else timestamp_sequence[pidx]
            item_timediff = item_timediffs_sequence[pidx]
            itemid_previous = item_sequence_id[seq[i - 1]] if i else num_items

            user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
            user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
            item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
            item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
            feature_tensor = Variable(torch.Tensor(feature).to(device)).unsqueeze(0)
            user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).to(device)).unsqueeze(0)
            item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).to(device)).unsqueeze(0)
            item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

            # PROJECT USER EMBEDDING
            user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
            user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

            predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

            # UPDATE USER AND ITEM EMBEDDING
            user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
            item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

            # SAVE EMBEDDINGS
            item_embeddings[itemid, :] = item_embedding_output.squeeze(0)
            user_embeddings[userid, :] = user_embedding_output.squeeze(0)
            user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
            item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)
        all_predictions.append(predicted_item_embedding.cpu().numpy())
    return all_predictions, idxs, previous_interaction_idxs
