******************
Privacy Estimation
******************


Overview
--------

This tool helps estimate potential leakage of sensitive information in the training data
through attacks on Machine Learning (ML) models. The main idea is to carry out known
inference attacks on the target model trained on sensitive data, and measure their success
to estimate the risk of leakage.

This tool supports membership inference attacks, which try to infer whether a
given data point was included in the training dataset of the target model. The determination
of the existence of a data point in a sensitive training set is also considered a breach of
privacy (e.g., knowing that a patient's record was included in some clinical trial dataset).

The main idea behind such attacks is that ML models are known to memorize some of the data
that they are trained on. Therefore, the prediction patterns of the model over data points
seen at training time are different from other data points, even if they are from the same
distribution. The adversary exploits this fact and forms a hypothesis around some
properties of the model outputs to decide whether a data point was a member of the training
set of the target model.

Each of the attacks implemented in this tool is essentially a binary classifiers, which
captures a hypotheses by the adversary. The features of this classifier are some
properties of the model output, and the labels indicate membership ground truth.
The classifier, or the attack model, can be a simple threshold or a complex trained ML model.
At attack time, the adversary presents the attack data point to the target model, gets back
the prediction probabilities, and derives appropriate attack features, which are then
presented to the tuned attack model to get membership prediction. The prediction
performance of the attack model over many attack points is therefore a measure of the risk
of success of such attacks, and hence an estimate of the leakage of private information.


Current Scope and Assumptions
-----------------------------

This tool supports sklearn-style classification models, which can output prediction
probabilities. Black-box access to the model predictions and prediction probabilities by the
adversary is assumed. The attacks implemented currently assume that the adversary
has access to a small amount of data drawn from the same distribution as the training data
of the target model, and for which membership information is known. Though this may seem
like a strong assumption, it is realistic in scenarios where the adversary obtains small
amount of such data through side channel attacks. White-box attacks, which also assume
access to the model parameters are not yet implemented.

This tool currently supports membership inference attacks. Advanced inference attacks,
such as attribute inference attacks - determining specific attribute values
from a partial record, and property inference attacks - determining data distribution of the
training set are not yet implemented.


Configuration
-------------

.. code-block:: python

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
    from guardian_ai.privacy_estimation.dataset import ClassificationDataset

    print("Running Dataset: " + dataset_name)
    dataset = ClassificationDataset(dataset_name)
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

The main idea here is that we need to carve out a subset of the
target model's training data for training and testing the attack (``ATTACK_TRAIN_IN`` and
``ATTACK_TEST_IN``). The part of the target model's training data that is not used for the
attacks is ``TARGET_ADDITIONAL_TRAIN``. Therefore, the target model's training set is created
by merging these three sets. We also need to set aside some data points that were not used
for training the target model (``ATTACK_TRAIN_OUT`` and ``ATTACK_TEST_OUT``). Finally, we need data
for tuning and testing the target model itself (``TARGET_VALID``, ``TARGET_TEST``).
Note that we first create these finer granularity splits, and then merge them to form the
appropriate train and test sets for the target model and the attack model.

.. code-block:: python

    from guardian_ai.privacy_estimation.dataset import DataSplit

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


Register Target Model
---------------------

List of all the target models to try on this dataset. See ``guardian_ai.privacy_estimation.models.py``
for the target models currently supported, but one can easily configure new target models by
subclassing the ``TargetModel`` class. Any sklearn based classifier that implements ``.predict_proba``
method is supported. Typically, we train each of the target models once, and then run multiple
attacks against it to see which one performs the best, thus giving us the worst case
risk for that target model.

.. code-block:: python

    from guardian_ai.privacy_estimation.model import (
        RandomForestTargetModel,
        GradientBoostingTargetModel,
        LogisticRegressionTargetModel,
        SGDTargetModel,
        MLPTargetModel
    )

    target_models = []
    target_models.append(RandomForestTargetModel())
    target_models.append(RandomForestTargetModel(n_estimators=1000))
    target_models.append(GradientBoostingTargetModel())
    target_models.append(GradientBoostingTargetModel(n_estimators=1000))
    target_models.append(LogisticRegressionTargetModel())
    target_models.append(SGDTargetModel())
    target_models.append(MLPTargetModel())
    target_models.append(MLPTargetModel(hidden_layer_sizes=(800,)))



Register Attacks
----------------

Specify which attacks you would like to run. To get an estimate of the worst case risk,
run all the attacks and see which one performs the best. See attack modules to see the
description of these attacks.

.. code-block:: python

    from guardian_ai.privacy_estimation.attack import AttackType

    attacks = []
    attacks.append(AttackType.LossBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedLossBasedBlackBoxAttack)
    attacks.append(AttackType.ConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.MerlinAttack)
    attacks.append(AttackType.CombinedBlackBoxAttack)
    attacks.append(AttackType.CombinedWithMerlinBlackBoxAttack)
    attacks.append(AttackType.MorganAttack)


Setup Threshold Grids
---------------------

Setup threshold grids for the threshold based attacks we plan to run. Loss threshold
grid depends on the datasets and models. The confidence and merlin ratios are
always in the range of 0-1, but where the values are concentrated again depends on the
dataset and the models.

.. code-block:: python

    threshold_grids = {
        AttackType.LossBasedBlackBoxAttack.name: [-0.0001, -0.001, -0.01, -0.05, -0.1, -0.3,
                                                -0.5, -0.7,-0.9, -1.0, -1.5, -10, -50, -100],
        AttackType.ConfidenceBasedBlackBoxAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                                                0.99, 0.999, 1.0],
        AttackType.MerlinAttack.name: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0]
    }


Initiate ``AttackRunner``
-------------------------

``AttackRunner`` is responsible for training all the target models.

.. code-block:: python

    from guardian_ai.privacy_estimation.attack_runner import AttackRunner

    attack_runner = AttackRunner(dataset,
                                target_models,
                                attacks,
                                threshold_grids
                                )

    attack_runner.train_target_models()


Cache
-----

Cache can be helpful for running the Morgan and Combined attacks, since they use information
from other attacks, which might be expensive to compute.

.. code-block:: python

    cache_input = AttackType.MorganAttack in attacks \
                or AttackType.CombinedBlackBoxAttack \
                or AttackType.CombinedWithMerlinBlackBoxAttack in attacks


Run Attacks
-----------

Run specified attacks on the given target models and record success metrics.

.. code-block:: python

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






************************************
Evaluating Externally Trained Models
************************************

This section outlines how to assess the privacy risk of a model trained outside the Guardian AI framework


Step 1: Load Your Data
----------------------

Load the data used to train your model and a similar dataset not used in training ( CSV files, Dataframes )

.. code-block:: python

    df_x_in = pd.read_csv("in_data.csv")  # Features from training data
    df_y_in = pd.read_csv("in_labels.csv", header=None).squeeze()  # Labels from training data
    df_x_out = pd.read_csv("out_data.csv")  # Features from non-training data
    df_y_out = pd.read_csv("out_labels.csv", header=None).squeeze()  # Labels from non-training data


Step 2: Prepare Attack Splits
-----------------------------

Use the ``prepare_attack_data_for_pretrained_model`` method to create attack-specific data splits:

.. code-block:: python

    from guardian_ai.privacy_estimation.dataset import ClassificationDataset, DataSplit

    dataset = ClassificationDataset("your_dataset_name")
    dataset.prepare_attack_data_for_pretrained_model(
        data_split_seed=42,
        dataset_split_ratios={
            DataSplit.ATTACK_TRAIN_IN: 0.3,
            DataSplit.ATTACK_TEST_IN: 0.7,
            DataSplit.ATTACK_TRAIN_OUT: 0.3,
            DataSplit.ATTACK_TEST_OUT: 0.7,
        },
        df_x_in=df_x_in,
        df_y_in=df_y_in,
        df_x_out=df_x_out,
        df_y_out=df_y_out
    )


Step 3: Wrap Your Model
-----------------------

Wrap your pretrained model to make it compatible with the framework:

.. code-block:: python

    from guardian_ai.privacy_estimation.model import TargetModel

    class ExternalTargetModel(TargetModel):
        """
        Wrapper for external pretrained models.
        """
        def __init__(self, model):
            self.model = model

        def get_model(self):
            return self.model

        def get_model_name(self):
            return "external_model"

        def get_prediction_probs(self, X):
            return self.model.predict_proba(X)


Step 4: Register Attacks and Run Evaluation
-------------------------------------------

Instantiate the attack runner and execute the evaluation:

.. code-block:: python

    # Initialize attack runner
    attack_runner = AttackRunner(
        dataset=dataset,
        target_models=[ExternalTargetModel(your_external_model)],
        attacks=[
            AttackType.LossBasedBlackBoxAttack,
            AttackType.ConfidenceBasedBlackBoxAttack,
            AttackType.MerlinAttack
        ],
        threshold_grids={AttackType.MerlinAttack.name: [0.001, 0.01, 0.1]}
    )

    results = attack_runner.run_attack(
        target_model=ExternalTargetModel(your_external_model),
        attack_type=AttackType.MerlinAttack,
        metric_functions=["precision", "recall", "f1", "accuracy"],
        cache_input=True
    )


Notes:
------

1.	Data Preprocessing: Ensure ``df_x_in`` and ``df_x_out`` are preprocessed identically to how they were during the model's training
2.	Split Ratios: The sum of ``ATTACK_TRAIN_IN`` + ``ATTACK_TEST_IN`` and ``ATTACK_TRAIN_OUT`` + ``ATTACK_TEST_OUT`` must equal ``1.0``
3.	Model Compatibility: The external model must support a ``predict_proba`` method
