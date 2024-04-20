from guardian_ai.privacy_estimation.dataset import (
    CFDataset,
	CFDataSplit,
)

from guardian_ai.privacy_estimation.attack import AttackType
from guardian_ai.privacy_estimation.attack_runner import AttackRunner

from guardian_ai.privacy_estimation.recommender_model import MLPTargetModel, GMFTargetModel, NCFTargetModel
import pytest
import pandas as pd

@pytest.fixture(scope="module")
def dataset():
    input_features = pd.read_csv("tests/test_data/recommender_test_data.csv")
    dataset = CFDataset("dummy data")
    dataset.load_data_from_df(input_features)
    return dataset


@pytest.fixture(scope="module")
def dataset_split_ratios():
    dataset_split_ratios = {
        CFDataSplit.ATTACK_TRAIN_IN: 0.2,
        CFDataSplit.ATTACK_TRAIN_OUT: 0.2,
        CFDataSplit.TARGET_TRAIN_MEMBERS: 0.2,
        CFDataSplit.TARGET_NON_MEMBERS: 0.2,
        CFDataSplit.ITEM_DATASET:0.2
    }
    return dataset_split_ratios


@pytest.fixture(scope="module")
def target_models():
    target_models = []
    target_models.append(NCFTargetModel(10, [64,32,16,8], 50, 20, 64, 0.01))
    target_models.append(MLPTargetModel(10, [64,32,16,8], 50, 64, 0.01))
    target_models.append(GMFTargetModel(10, 50, 5, 64, 0.01)) 

    return target_models

@pytest.fixture(scope="module")
def shadow_models():
    shadow_models = []
    shadow_models.append(NCFTargetModel(10, [64,32,16,8], 50, 20, 64, 0.001))
    shadow_models.append(MLPTargetModel(10, [64,32,16,8], 50, 64, 0.01))
    shadow_models.append(GMFTargetModel(10, 50, 5, 64, 0.01)) 
    return shadow_models


@pytest.fixture(scope="module")
def attacks():
    attacks = []
    attacks.append(AttackType.CollaborativeFilteringAttack)
    return attacks

@pytest.fixture(scope="module")
def metric_functions():
    return ["precision", "recall", "f1", "accuracy"]


@pytest.fixture(scope="module")
def attack_runner(dataset, target_models, shadow_models, attacks):
    return AttackRunner(dataset, target_models, attacks, None, shadow_models)


def test_dummy_dataset(dataset):
    assert dataset.get_num_rows() == 1000209


def test_prepare_attack_shadow_target_data(dataset, dataset_split_ratios):
    dataset.get_item_features(dataset_split_ratios)
    dataset.create_shadow_target_dataset()
    dataset.prepare_target_and_attack_data(42, dataset_split_ratios)
    
    assert dataset is not None
    assert dataset.target_model_data is not None
    assert dataset.shadow_model_data is not None
    assert dataset.attack_model_data is not None
    assert dataset.item_features is not None

    assert 1620 < len(dataset.attack_model_data.y_membership_train) < 1700
    assert 810 < len(dataset.target_model_data.X_target_members) < 850
    assert 810 < len(dataset.shadow_model_data.X_target_members) < 850
    assert 200000 < len(dataset.item_features) < 300000


def test_run_attack(attack_runner, metric_functions, dataset, target_models, shadow_models, attacks):
    item_vectors = dataset.perform_matrix_factorization(50)
    attack_runner.train_collaborative_filtering_models()
    attack_results = []
    for target_model in target_models:
        for shadow_model in shadow_models:
            for attack_type in attacks:
                    result_attack = attack_runner.run_attack(
                        target_model, attack_type, metric_functions, None, None, item_vectors, shadow_model
                )
                    attack_results.append(result_attack)
    assert attack_results is not None

    for x in range(len(attack_results)):
        attack_result = float(attack_results[x].split()[4])
        assert 0.89 < attack_result < 1.0
   
