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
lists of recommendations. The attacks implemented for recommender systems assume that the adversary 
has access to a small amount of data reflecting user-item interactions from the same distribution 
as the training data of the target recommender model, as well as access to the recommender outputs.
While this assumption may seem strong, it mirrors realistic scenarios where adversaries gain limited 
access to such data through creation of false users and webscraping.


Configuration
-------------
Note: inputs for the User IDs and Item IDs must be continuous, ranging from 0 - # Users and 0 - # Items respectively.

Load Data
---------
.. code-block:: python
input_features = pd.read_csv('ratings.csv', names=['userID', "itemID", "rating", 'timestamp'])

dataset = CFDataset("Example Data")
dataset.load_data_from_df(input_features)
dataset_split_ratios = {
        CFDataSplit.ATTACK_TRAIN_IN: 0.2,
        CFDataSplit.ATTACK_TRAIN_OUT: 0.2,
        CFDataSplit.TARGET_TRAIN_MEMBERS: 0.2,
        CFDataSplit.TARGET_NON_MEMBERS: 0.2,
        CFDataSplit.ITEM_DATASET:0.2
    }
dataset.get_item_features(dataset_split_ratios)
item_vectors = dataset.perform_matrix_factorization(50)

dataset.create_shadow_target_dataset()


Setup Target and Shadow Models
---------
.. code-block:: python
from guardian_ai.privacy_estimation.recommender_model import MLPTargetModel, GMFTargetModel, NCFTargetModel
from guardian_ai.privacy_estimation.attack import AttackType

target_models = []
target_models.append(NCFTargetModel(10, [64,32,16,8], 50, 20, 64, 0.001))
target_models.append(MLPTargetModel(10, [64,32,16,8], 50, 64, 0.01))
target_models.append(GMFTargetModel(10, 50, 5, 64, 0.01)) 

shadow_models = []
shadow_models.append(NCFTargetModel(10, [64,32,16,8], 50, 20, 64, 0.001))
shadow_models.append(MLPTargetModel(10, [64,32,16,8], 50, 64, 0.01))
shadow_models.append(GMFTargetModel(10, 50, 5, 64, 0.01)) 

attacks = []
attacks.append(AttackType.CollaborativeFilteringAttack)

Run Attack
---------
.. code-block:: python
from guardian_ai.privacy_estimation.attack_runner import AttackRunner
attack_runner = AttackRunner( dataset, target_models, attacks, None, shadow_models)
attack_runner.train_collaborative_filtering_models()

metric_functions = ["precision", "recall", "f1", "accuracy"]
result_attacks = []
for target_model in attack_runner.target_models:
    for shadow_model in attack_runner.shadow_models:
       for attack_type in attack_runner.attacks:
            result_attack = attack_runner.run_attack(
                target_model, attack_type, metric_functions, None, None, item_vectors, shadow_model
         )
            result_attacks.append(result_attack)

