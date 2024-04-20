import os
from recommender_dataset import RecommenderDataset

# Source data directory
source_dir = "/Users/ihanus/Projects/recsys-inference-attacks/data/"
# dataset name
ratings_name = 'ratings.dat'
items_name = 'movies.dat'
users_name = 'users.dat'
dataset_name = "movielens"


# # Define attack metrics we care about
# metric_functions = ["precision", "recall", "f1", "accuracy"]
# print_roc_curve = False  # print the values of the ROC curve

# # Prepare result file for storing target model and attack metrics
# fout = open(result_file, "w")
# fout.write("dataset\tnum_rows\ttarget_model\ttrain_f1\ttest_f1\tattack_type")
# for metric in metric_functions:
#     fout.write("\tattack_" + metric)
# fout.write("\n")

# Load data
print("Running Dataset: " + dataset_name)
dataset = RecommenderDataset(dataset_name)
dataset.load_data(os.path.join(source_dir,users_name),
                  os.path.join(source_dir, items_name),
                  os.path.join(source_dir,ratings_name))
dataset.perform_matrix_factorization(20)
# string for reporting in the result file
# result_dataset = dataset_name + "\t" + str(dataset.get_num_rows())


# dataset_split_ratios = {
#     DataSplit.ATTACK_TRAIN_IN : 0.1,  # fraction of datapoints for training the
#     # attack model, included in target model training set
#     DataSplit.ATTACK_TRAIN_OUT : 0.1,  # fraction of datapoints for training the
#     # attack model, not included in target model training set
#     DataSplit.ATTACK_TEST_IN : 0.2,  # fraction of datapoints for evaluating the
#     # attack model, included in target model training set
#     DataSplit.ATTACK_TEST_OUT : 0.2,  # fraction of datapoints for evaluating the
#     # attack model, not included in target model training set
#     DataSplit.TARGET_ADDITIONAL_TRAIN : 0.1,  # fraction of datapoints included in
#     # target model training set, not used in the attack training or testing
#     DataSplit.TARGET_VALID : 0.1,  # fraction of datapoints for tuning the target model
#     DataSplit.TARGET_TEST : 0.2  # fraction of datapoints for evaluating the
#     # target model
# }

# dataset.prepare_target_and_attack_data(data_split_seed, dataset_split_ratios)

# # Register target model
# target_models = []
# target_models.append(RandomForestTargetModel())
# target_models.append(RandomForestTargetModel(n_estimators=1000))
# target_models.append(GradientBoostingTargetModel())
# target_models.append(GradientBoostingTargetModel(n_estimators=1000))
# target_models.append(LogisticRegressionTargetModel())
# target_models.append(SGDTargetModel())
# target_models.append(MLPTargetModel())
# target_models.append(MLPTargetModel(hidden_layer_sizes=(800,)))

# # Specify which attacks you would like to run.
# attacks = []
# attacks.append(AttackType.LossBasedBlackBoxAttack)
# attacks.append(AttackType.ExpectedLossBasedBlackBoxAttack)
# attacks.append(AttackType.ConfidenceBasedBlackBoxAttack)
# attacks.append(AttackType.ExpectedConfidenceBasedBlackBoxAttack)
# attacks.append(AttackType.MerlinAttack)
# attacks.append(AttackType.CombinedBlackBoxAttack)
# attacks.append(AttackType.CombinedWithMerlinBlackBoxAttack)
# attacks.append(AttackType.MorganAttack)

# # Setup threshold grids for the threshold based attacks we plan to run.
# threshold_grids = {
#     AttackType.LossBasedBlackBoxAttack.name: [-0.0001, -0.001, -0.01, -0.05, -0.1, -0.3,
#                                             -0.5, -0.7,-0.9, -1.0, -1.5, -10, -50, -100],
#     AttackType.ConfidenceBasedBlackBoxAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
#                                             0.99, 0.999, 1.0],
#     AttackType.MerlinAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0]
# }

# # Initiate AttackRunner
# attack_runner = AttackRunner(dataset,
#                         target_models,
#                         attacks,
#                         threshold_grids
#                         )

# attack_runner.train_target_models()

# # Set Cache
# cache_input = AttackType.MorganAttack in attacks \
#             or AttackType.CombinedBlackBoxAttack \
#             or AttackType.CombinedWithMerlinBlackBoxAttack in attacks

# # Run attacks
# for target_model in target_models:
#     result_target = attack_runner.target_model_result_strings.get(target_model.get_model_name())

#     for attack_type in attacks:
#         result_attack = attack_runner.run_attack(target_model,
#                                                 attack_type,
#                                                 metric_functions,
#                                                 print_roc_curve=print_roc_curve,
#                                                 cache_input=cache_input)
#         fout.write(result_dataset + "\t" + result_target + "\t" + result_attack)
#     fout.flush()
# fout.close()

# # Generates a plot
# ResultPlot.print_best_attack(
#     dataset_name=dataset.name,
#     result_filename=result_file,
#     graphs_dir=graph_dir,
#     metric_to_sort_on="attack_accuracy",
#     )