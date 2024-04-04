.. _quick-start-pe:

Privacy Estimation
==================

.. code-block:: python

    import os
    from guardian_ai.privacy_estimation.dataset import DataSplit, ClassificationDataset
    from guardian_ai.privacy_estimation.model import (
        RandomForestTargetModel,
        GradientBoostingTargetModel,
        LogisticRegressionTargetModel,
        SGDTargetModel,
        MLPTargetModel
    )
    from guardian_ai.privacy_estimation.attack import AttackType
    from guardian_ai.privacy_estimation.attack_runner import AttackRunner
    from guardian_ai.privacy_estimation.plot_results import ResultPlot

    # Source data directory
    source_dir = "<local_path_to_data>"
    # dataset name
    dataset_name = "titanic"
    # source file
    source_file = "titanic.csv"
    # does the dataset contain header
    contains_header = True
    # index of the target variable
    target_ix = 0
    # Seed for data splits
    data_split_seed = 42
    # File to save results in
    result_file = "titanic_out.txt"
    # directory to store graphs
    graph_dir = "."


    if target_ix == -1:
        target_ix = None  # this will automatically pick the last index

    ignore_ix = None  # specify if you need to ignore any features

    # Define attack metrics we care about
    metric_functions = ["precision", "recall", "f1", "accuracy"]
    print_roc_curve = False  # print the values of the ROC curve

    # Prepare result file for storing target model and attack metrics
    fout = open(result_file, "w")
    fout.write("dataset\tnum_rows\ttarget_model\ttrain_f1\ttest_f1\tattack_type")
    for metric in metric_functions:
        fout.write("\tattack_" + metric)
    fout.write("\n")

    # Load data
    print("Running Dataset: " + dataset_name)
    dataset = ClassificationDataset(dataset_name)
    dataset.load_data(os.path.join(source_dir,source_file),
                    contains_header=contains_header,
                    target_ix=target_ix,
                    ignore_ix=ignore_ix)

    # string for reporting in the result file
    result_dataset = dataset_name + "\t" + str(dataset.get_num_rows())


    dataset_split_ratios = {
        DataSplit.ATTACK_TRAIN_IN : 0.1,  # fraction of datapoints for training the
        # attack model, included in target model training set
        DataSplit.ATTACK_TRAIN_OUT : 0.1,  # fraction of datapoints for training the
        # attack model, not included in target model training set
        DataSplit.ATTACK_TEST_IN : 0.2,  # fraction of datapoints for evaluating the
        # attack model, included in target model training set
        DataSplit.ATTACK_TEST_OUT : 0.2,  # fraction of datapoints for evaluating the
        # attack model, not included in target model training set
        DataSplit.TARGET_ADDITIONAL_TRAIN : 0.1,  # fraction of datapoints included in
        # target model training set, not used in the attack training or testing
        DataSplit.TARGET_VALID : 0.1,  # fraction of datapoints for tuning the target model
        DataSplit.TARGET_TEST : 0.2  # fraction of datapoints for evaluating the
        # target model
    }

    dataset.prepare_target_and_attack_data(data_split_seed, dataset_split_ratios)

    # Register target model
    target_models = []
    target_models.append(RandomForestTargetModel())
    target_models.append(RandomForestTargetModel(n_estimators=1000))
    target_models.append(GradientBoostingTargetModel())
    target_models.append(GradientBoostingTargetModel(n_estimators=1000))
    target_models.append(LogisticRegressionTargetModel())
    target_models.append(SGDTargetModel())
    target_models.append(MLPTargetModel())
    target_models.append(MLPTargetModel(hidden_layer_sizes=(800,)))

    # Specify which attacks you would like to run.
    attacks = []
    attacks.append(AttackType.LossBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedLossBasedBlackBoxAttack)
    attacks.append(AttackType.ConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.MerlinAttack)
    attacks.append(AttackType.CombinedBlackBoxAttack)
    attacks.append(AttackType.CombinedWithMerlinBlackBoxAttack)
    attacks.append(AttackType.MorganAttack)

    # Setup threshold grids for the threshold based attacks we plan to run.
    threshold_grids = {
        AttackType.LossBasedBlackBoxAttack.name: [-0.0001, -0.001, -0.01, -0.05, -0.1, -0.3,
                                                -0.5, -0.7,-0.9, -1.0, -1.5, -10, -50, -100],
        AttackType.ConfidenceBasedBlackBoxAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                                                0.99, 0.999, 1.0],
        AttackType.MerlinAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0]
    }

    # Initiate AttackRunner
    attack_runner = AttackRunner(dataset,
                            target_models,
                            attacks,
                            threshold_grids
                            )

    attack_runner.train_target_models()

    # Set Cache
    cache_input = AttackType.MorganAttack in attacks \
                or AttackType.CombinedBlackBoxAttack \
                or AttackType.CombinedWithMerlinBlackBoxAttack in attacks

    # Run attacks
    for target_model in target_models:
        result_target = attack_runner.target_model_result_strings.get(target_model.get_model_name())

        for attack_type in attacks:
            result_attack = attack_runner.run_attack(target_model,
                                                    attack_type,
                                                    metric_functions,
                                                    print_roc_curve=print_roc_curve,
                                                    cache_input=cache_input)
            fout.write(result_dataset + "\t" + result_target + "\t" + result_attack)
        fout.flush()
    fout.close()

    # Generates a plot
    ResultPlot.print_best_attack(
        dataset_name=dataset.name,
        result_filename=result_file,
        graphs_dir=graph_dir,
        metric_to_sort_on="attack_accuracy",
        )
