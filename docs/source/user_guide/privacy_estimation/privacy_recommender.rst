******************
Membership Inference Attacks for Recommender Systems
******************


Overview
--------

This addition to the privacy estimation tool extends its capabilities to include membership
inference attacks specifically tailored for recommender systems. The primary goal remains the 
estimation of potential leakage of sensitive information in training data through attacks on 
Machine Learning (ML) models. However, the focus now shifts to understanding the privacy risks 
associated with user preferences and behavior data used in recommender system training. 
Recommender systems often handle sensitive information about users' preferences, interests, 
and behaviors. Membership inference attacks in this context aim to determine if a specific 
user's data was part of the training dataset used to build the recommendation model. This 
information leakage can lead to privacy breaches, especially in scenarios where user data 
confidentiality is critical.

Similar to traditional membership inference attacks, the recommender system attacks analyze the 
prediction patterns of the model to infer membership status. The attacks are designed as binary 
classifiers, where the features capture properties of the model's output related to user-item 
interactions, and the labels indicate membership ground truth (i.e., whether a user's data was 
in the training set). At attack time, the adversary presents data representing a user-item 
interaction to the target recommender model. The model outputs predictions based
on the user's historical behavior. The attack then extracts relevant features from these 
predictions and feeds them into the attack model to predict membership status. The success rate 
of these attacks provides an estimate of the risk of privacy leakage in the recommender system.

This enhancement extends the tool's applicability to privacy analysis in recommendation systems, 
ensuring a more comprehensive assessment of potential information leakage.


Current Scope and Assumptions
-----------------------------

This addition to Guardian AI maintains utilizes custom-made, but basic, PyTorch models that output 
lists of recommendations. Black-box access to the model recommendations is assumed.
The attacks implemented for recommender systems assume that the adversary
has access to a small amount of data reflecting user-item interactions from the same distribution 
as the training data of the target recommender model, as well as access to the recommender outputs.
While this assumption may seem strong, it mirrors realistic scenarios where adversaries gain limited 
access to such data through creation of false users and webscraping.

This implementation makes the assumption that each type of recommender system uses the most-popular algorithm
to solve the cold start problem. This means that the candidates that are most popular are recommended to users
who have no previous history of interaction with the system.

This tool currently supports membership inference attacks in the context of collaborative filtering models.
Attack for sequential recommender systems are not yet implemented.

Configuration
-------------
Note: inputs for the User IDs and Item IDs must be continuous, ranging from 0 - # Users and 0 - # Items respectively.

.. code-block:: python

    # Source data directory
    source_dir = "<local_path_to_data>"
    # dataset name
    dataset_name = "ratings"
    # source file
    source_file = "ratings.csv"
    # does the dataset contain header
    contains_header = True
    # index of the target variable
    target_ix = 0
    # Seed for data splits
    data_split_seed = 42
    # File to save results in
    result_file = "ratings_out.txt"
    # directory to store graphs
    graph_dir = "<local_directory_path_to_store_graphs>"
    # print the values of the ROC curve
    print_roc_curve = False
    # Define attack metrics we care about
    metric_functions = ["precision", "recall", "f1", "accuracy"]

    if target_ix == -1:
        target_ix = None  # this will automatically pick the last index

    ignore_ix = None  # specify if you need to ignore any features

    # Prepare result file for storing target model and attack metrics
    fout = open(result_file, "w")
    fout.write("dataset\tnum_rows\ttarget_model\ttrain_f1\ttest_f1\tattack_type")
    for metric in metric_functions:
        fout.write("\tattack_" + metric)
    fout.write("\n")

Load Data
---------

.. code-block:: python

    import os
    from guardian_ai.privacy_estimation.dataset import CFDataset

    print("Running Dataset: " + dataset_name)
    dataset = CFDataset(dataset_name)
    dataset.load_data(
        os.path.join(source_dir,source_file),
        contains_header=contains_header,
        target_ix=target_ix,
        ignore_ix=ignore_ix
    )

    # string for reporting in the result file
    result_dataset = dataset_name + "\t" + str(dataset.get_num_rows())




Prepare Data Splits
-------------------

The main idea here involves transforming the user-item matrix into per-user data.
For each user, we enumerate the items they have interacted with. We'll then select a portion of this
transformed dataset to train a shadow model. The shadow model's outputs will serve to train the
attack model for both members and non-members (``ATTACK_TRAIN_IN`` and ``ATTACK_TRAIN_OUT``). This approach
eliminates the need for additional variables for training the shadow model. A portion of the transformed
dataset will also be used to train the target model (``TARGET_TRAIN_MEMBERS`` and ``TARGET_NON_MEMBERS``). For
training both the shadow and target models, we employ the leave-one-out cross-validation method, eliminating
the necessity for separate testing datasets for the shadow and target models. The datasets for training
the target model will be utilized to assess the attack model as well. Moreover, a subset of the transformed
data is required to generate item vector representations (``ITEM_DATASET``), with the stipulation that this
subset encompasses all items included in both the target and shadow datasets. It's important to first execute
these detailed splits before merging them to form the appropriate training sets for the target and attack
models.

.. code-block:: python

    from guardian_ai.privacy_estimation.dataset import DataSplit

    dataset_split_ratios = {
        DataSplit.ATTACK_TRAIN_IN : 0.2,
        DataSplit.ATTACK_TRAIN_OUT : 0.2,
        DataSplit.TARGET_TRAIN_MEMBERS : 0.2,
        DataSplit.TARGET_NON_MEMBERS : 0.2,
        DataSplit.ITEM_DATASET: 0.2
    }

    dataset.prepare_target_and_attack_data(data_split_seed, dataset_split_ratios)


Register Target Model
---------------------

List of all the target recommender models to try on this dataset. See
``guardian_ai.privacy_estimation.recommender_models.py`` for the target models currently supported.
Typically, we train each of the target models once, and then run multiple
attacks against it to see which one performs the best, thus giving us the worst case
risk for that target model.

.. code-block:: python

    from guardian_ai.privacy_estimation.model import (
        MLPTargetModel, GMFTargetModel, NCFTargetModel,
    )

    target_models = []
    target_models.append(NCFTargetModel(10, [64, 32, 16, 8], 50, 20, 64, 0.001))
    target_models.append(MLPTargetModel(10, [64, 32, 16, 8], 20, 64, 0.001))
    target_models.append(GMFTargetModel(10, 50, 20, 64, 0.001))

Register Shadow Model
---------------------

List of all the shadow recommender models to try on this dataset. See
``guardian_ai.privacy_estimation.recommender_models.py`` for the models currently supported.
Typically, we train each of the shadow models once, and then run multiple
attacks to see which one performs the best.

.. code-block:: python

    from guardian_ai.privacy_estimation.model import (
        MLPTargetModel, GMFTargetModel, NCFTargetModel,
    )

    shadow_models = []
    shadow_models.append(NCFTargetModel(10, [64, 32, 16, 8], 50, 20, 64, 0.001))
    shadow_models.append(MLPTargetModel(10, [64, 32, 16, 8], 20, 64, 0.001))
    shadow_models.append(GMFTargetModel(10, 50, 20, 64, 0.001))




Initiate ``AttackRunner``
-------------------------

``AttackRunner`` is responsible for training all the target models.

.. code-block:: python

    from guardian_ai.privacy_estimation.attack_runner import AttackRunner

    attack_runner = AttackRunner(dataset,
                                target_models,
                                attacks,
                                threshold_grids,
                                shadow_models
                                )

    attack_runner.train_collaborative_filtering_models()




Run Attacks
-----------

Run specified attacks on the given target models and record success metrics.

.. code-block:: python

    for target_model in target_models:
        for shadow_model in shadow_models:
             for attack_type in attacks:
                    result_attack = attack_runner.run_attack(target_model,
                                                             attack_type,
                                                             metric_functions,
                                                             print_roc_curve=print_roc_curve,
                                                             cache_input=None,
                                                             item_vectors=item_vectors,
                                                             shadow_model)

Generates Plots
---------------

Generates a plot - in this case a table. Given a result file, sort attack performance
by the given metric and print out the best attacks for each dataset for each model


.. code-block:: python

    from guardian_ai.privacy_estimation.plot_results import ResultPlot

    ResultPlot.print_best_attack(
        dataset_name=dataset.name,
        result_filename=result_file,
        graphs_dir=graph_dir,
        metric_to_sort_on="attack_accuracy",
    )
