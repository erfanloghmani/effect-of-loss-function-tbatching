'''
This code trains the JODIE model for the given dataset.
The task is: interaction prediction.

How to run:
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.
'''

import json
from library_data import *
import library_models as lib
from library_models import *
from sklearn.cluster import KMeans

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--init_epoch', default=-1, type=int, help='Init epoch to start train the model from')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument("--state_change", default=False, action="store_true", help="True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.")
parser.add_argument('--device', default='gpu', type=str, help='which device to use')
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

if args.device == 'gpu':
    # SET GPU
    if args.gpu == -1:
        args.gpu = select_free_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, y_true, item_word_embs] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1  # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true) / (1.0 + sum(y_true))  # +1 in denominator in case there are no state change labels, which will throw an error.
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained.
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm).
The batches are then used to train JODIE.
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates.
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 600

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items, item_word_embs.shape[1]).to(device)
weight = torch.Tensor([1, true_labels_ratio]).to(device)
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()
MSELoss_no_reduce = nn.MSELoss(reduction='none')

N_CLUSTERS = 12
km = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(item_word_embs)
item_clusters = km.predict(item_word_embs)

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))  # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1)  # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1)  # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).to(device))  # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).to(device))  # one-hot vectors for static embeddings

item_word_embs_torch = torch.tensor(item_word_embs, dtype=torch.float).to(device)
user_last_word_in_cluster = torch.zeros((num_users, N_CLUSTERS, item_word_embs.shape[1])).to(device)
user_saw_cluster = torch.zeros((num_users, N_CLUSTERS)).to(device)

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


all_total_losses = []

if (args.init_epoch >= 0):
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.init_epoch)
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

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)
with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)

                # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                itemid_prev = user_previous_itemid_sequence[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan:
                    tbatch_start_time = timestamp  # RESET START TIME FOR THE NEXT TBATCHES

                    # ITERATE OVER ALL T-BATCHES
                    with trange(len(lib.current_tbatches_user)) as progress_bar3:
                        for i in progress_bar3:
                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))

                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).to(device)  # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).to(device)  # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).to(device)
                            feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).to(device))  # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device)).unsqueeze(1)
                            item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).to(device)).unsqueeze(1)
                            tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).to(device)
                            item_embedding_previous = item_embeddings[tbatch_itemids_previous, :]

                            for j, userid in enumerate(lib.current_tbatches_user[i]):
                                itemid_prev = lib.current_tbatches_previous_item[i][j]
                                user_last_word_in_cluster[j, item_clusters[itemid_prev], :] = item_word_embs_torch[itemid_prev]
                                user_saw_cluster[j, item_clusters[itemid_prev]] = 1

                            item_word_embs_input = item_word_embs_torch[tbatch_itemids, :]
                            item_word_embs_previous = item_word_embs_torch[tbatch_itemids_previous, :]
                            feature_tensor_full = torch.cat([feature_tensor, item_word_embs_input], dim=1)

                            # PROJECT USER EMBEDDING TO CURRENT TIME
                            user_embedding_input = user_embeddings[tbatch_userids, :]
                            user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                            user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_word_embs_previous, item_embedding_static[tbatch_itemids_previous, :], user_embedding_static[tbatch_userids, :]], dim=1)
                            user_only_embeddings = torch.cat([user_projected_embedding, user_embedding_static[tbatch_userids, :]], dim=1)


                            tbatch_user_last_word_in_cluster = user_last_word_in_cluster[tbatch_userids]
                            tbatch_user_saw_cluster = user_saw_cluster[tbatch_userids]
                            tbatch_full_user_repeat = torch.cat([user_embedding_input, user_embedding_static[tbatch_userids, :]], dim=1).unsqueeze(1).repeat((1, N_CLUSTERS, 1))

                            # PREDICT NEXT ITEM EMBEDDING
                            predicted_item_embedding = model.predict_item_embedding(
                                user_item_embedding, 
                                user_only_embeddings=user_only_embeddings,
                                item_word_embeddings=tbatch_user_last_word_in_cluster,
                                user_saw_clusters=tbatch_user_saw_cluster
                            )

                            weight_dynamic = predicted_item_embedding[:, -(item_word_embs.shape[1] + 2)]
                            weight_word = predicted_item_embedding[:, -(item_word_embs.shape[1] + 1)]

                            print(predicted_item_embedding.shape)
                            print(args.embedding_dim, item_word_embs.shape)
                            print(predicted_item_embedding[:, args.embedding_dim:-(item_word_embs.shape[1] + 2)].shape)
                            # CALCULATE PREDICTION LOSS
                            item_embedding_input = item_embeddings[tbatch_itemids, :]
                            # print(weight_dynamic.shape, predicted_item_embedding[:, :args.embedding_dim].shape, item_embedding_input.detach().shape) 
                            loss += torch.sum(torch.exp(weight_dynamic) * MSELoss_no_reduce(predicted_item_embedding[:, :args.embedding_dim], item_embedding_input.detach()).sum(1))
                            loss += torch.sum(MSELoss_no_reduce(predicted_item_embedding[:, args.embedding_dim:-(item_word_embs.shape[1] + 2)], item_embedding_static[tbatch_itemids, :]).sum(1))
                            loss += torch.sum(MSELoss_no_reduce(predicted_item_embedding[:, -item_word_embs.shape[1]:], item_word_embs_input.unsqueeze(1)).sum(2) * torch.exp(weight_word))

                            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                            user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor_full, select='user_update')
                            item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor_full, select='item_update')

                            item_embeddings[tbatch_itemids, :] = item_embedding_output
                            user_embeddings[tbatch_userids, :] = user_embedding_output

                            user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output
                            item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output

                            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                            loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                            loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                            # CALCULATE STATE CHANGE LOSS
                            if args.state_change:
                                loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss)

                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    item_embeddings_timeseries.detach_()
                    user_embeddings_timeseries.detach_()

                    # REINITIALIZE
                    reinitialize_tbatches()
                    tbatch_to_insert = -1

        # END OF ONE EPOCH
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
        del(item_embeddings_dystat)
        del(user_embeddings_dystat)
        all_total_losses.append(total_loss)
        json.dump(all_total_losses, open('results/jodie_%s_training_total_losses.json' % args.network, 'w'))

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
json.dump(all_total_losses, open('results/jodie_%s_training_total_losses.json' % args.network, 'w'))
