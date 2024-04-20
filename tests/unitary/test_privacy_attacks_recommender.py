import guardian_ai.privacy_estimation.attack
from guardian_ai.privacy_estimation.dataset import (
    CFDataset,
	CFDataSplit,
    TargetCFData,
    AttackModelData
)

from guardian_ai.privacy_estimation.attack import AttackType
from guardian_ai.privacy_estimation.attack_runner import AttackRunner
from guardian_ai.privacy_estimation.recommender_model import (
	NeuMF,
	MLP,
	GMF
)
import pytest
import pandas as pd


from tests.utils import get_dummy_dataset


@pytest.fixture(scope="module")
def dataset():
    input_features = pd.read_csv("ratings.csv")
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
    target_models.append(MLP())
    target_models.append(NeuMF())
    target_models.append(GMF())
    return target_models

@pytest.fixture(scope="module")
def shadow_models():
    shadow_models = []
    shadow_models.append(MLP())
    shadow_models.append(NeuMF())
    shadow_models.append(GMF())
    return shadow_models


@pytest.fixture(scope="module")
def attacks():
    attacks = []
    attacks.append(AttackType.CFAttack)
    return attacks

@pytest.fixture(scope="module")
def metric_functions():
    return ["precision", "recall", "f1", "accuracy"]


@pytest.fixture(scope="module")
def attack_runner(dataset, target_models, shadow_models, attacks):
    return AttackRunner( dataset, target_models, attacks, shadow_models)


def test_dummy_dataset(dataset):
    assert dataset.get_num_rows() == 500


def test_prepare_target_and_attack_data(dataset, dataset_split_ratios):
    dataset.prepare_target_and_attack_data(42, dataset_split_ratios)
    assert len(dataset.splits) == 5
    target_model_data = dataset.target_model_data
    dataset.get_item_features(dataset_split_ratios)
    attack_model_data = dataset.attack_model_data
    assert target_model_data is not None
    assert attack_model_data is not None
    assert dataset.item_features is not None
    assert target_model_data.X_target_train.get_shape() == (200, 30)
    assert attack_model_data.X_attack_test.get_shape() == (199, 30)


def test_run_attack(attack_runner, metric_functions, dataset):
    attack_runner.train_collaborative_filtering_models()
    target_result_string_0 = attack_runner.target_model_result_strings[
        attack_runner.target_recommenders[0].get_model_name()
    ]
    target_result_string_1 = attack_runner.target_model_result_strings[
        attack_runner.target_recommenders[1].get_model_name()
    ]
    target_result_string_2 = attack_runner.target_model_result_strings[
        attack_runner.target_recommenders[2].get_model_name()
    ]
    
    shadow_result_string_0 = attack_runner.shadow_model_result_strings[
        attack_runner.shadow_recommenders[0].get_model_name()
    ]
    shadow_result_string_1 = attack_runner.shadow_model_result_strings[
        attack_runner.shadow_recommenders[1].get_model_name()
    ]
    shadow_result_string_2 = attack_runner.shadow_model_result_strings[
        attack_runner.shadow_recommenders[2].get_model_name()
    ]


    #target_result_string_0_test_f1 = target_result_string_0.split()[2]
    #assert 0.4648744113029828 == pytest.approx(float(target_result_string_0_test_f1))

    #target_result_string_1_test_f1 = target_result_string_1.split()[2]
    #assert 0.4733890801770782 == pytest.approx(float(target_result_string_1_test_f1))

    #target_result_string_2_test_f1 = target_result_string_2.split()[2]
    #assert 0.46529411764705875 == pytest.approx(float(target_result_string_2_test_f1))

    item_features = dataset.item_features
    result_attacks = []
    for target_model in attack_runner.target_models:
        for shadow_model in attack_runner.shadow_models:
           for attack_type in attack_runner.attacks:
                result_attack = attack_runner.run_attack_recommender(
                    target_model, shadow_model, attack_type, metric_functions, item_features, shadow_model
             )
                result_attacks.append(result_attack)

