#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import guardian_ai.privacy_estimation.attack
from guardian_ai.privacy_estimation.dataset import (
    ClassificationDataset,
    DataSplit,
    TargetModelData,
    AttackModelData,
)
from guardian_ai.privacy_estimation.attack import AttackType
from guardian_ai.privacy_estimation.attack_runner import AttackRunner
from guardian_ai.privacy_estimation.model import (
    RandomForestTargetModel,
    LogisticRegressionTargetModel,
    MLPTargetModel,
)
import pytest
import pandas as pd


from tests.utils import get_dummy_dataset


@pytest.fixture(scope="module")
def dataset():
    input_features, target = get_dummy_dataset(n_samples=500, n_features=5, n_classes=2)
    dataset = ClassificationDataset("dummy_data")
    dataset.load_data_from_df(input_features, target)
    return dataset


@pytest.fixture(scope="module")
def dataset_split_ratios():
    dataset_split_ratios = {
        DataSplit.ATTACK_TRAIN_IN: 0.1,  # fraction of datapoints for training the
        # attack model, included in target model training set
        DataSplit.ATTACK_TRAIN_OUT: 0.1,  # fraction of datapoints for training the
        # attack model, not included in target model training set
        DataSplit.ATTACK_TEST_IN: 0.2,  # fraction of datapoints for evaluating the
        # attack model, included in target model training set
        DataSplit.ATTACK_TEST_OUT: 0.2,  # fraction of datapoints for evaluating the
        # attack model, not included in target model training set
        DataSplit.TARGET_ADDITIONAL_TRAIN: 0.1,  # fraction of datapoints included in
        # target model training set, not used in the attack training or testing
        DataSplit.TARGET_VALID: 0.1,  # fraction of datapoints for tuning the target model
        DataSplit.TARGET_TEST: 0.2  # fraction of datapoints for evaluating the
        # target model
    }
    return dataset_split_ratios


@pytest.fixture(scope="module")
def target_models():
    target_models = []
    target_models.append(RandomForestTargetModel())
    target_models.append(LogisticRegressionTargetModel())
    target_models.append(MLPTargetModel())
    return target_models


@pytest.fixture(scope="module")
def attacks():
    attacks = []
    attacks.append(AttackType.LossBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedLossBasedBlackBoxAttack)
    attacks.append(AttackType.ConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.ExpectedConfidenceBasedBlackBoxAttack)
    attacks.append(AttackType.MerlinAttack)
    attacks.append(AttackType.CombinedBlackBoxAttack)
    attacks.append(AttackType.CombinedWithMerlinBlackBoxAttack)
    attacks.append(AttackType.MorganAttack)
    return attacks


@pytest.fixture(scope="module")
def threshold_grids():
    threshold_grids = {
        AttackType.LossBasedBlackBoxAttack.name: [
            -0.0001,
            -0.001,
            -0.01,
            -0.05,
            -0.1,
            -0.3,
            -0.5,
            -0.7,
            -0.9,
            -1.0,
            -1.5,
            -10,
            -50,
            -100,
        ],
        AttackType.ConfidenceBasedBlackBoxAttack.name: [
            0.001,
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.99,
            0.999,
            1.0,
        ],
        AttackType.MerlinAttack.name: [
            0.001,
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.99,
            0.999,
            1.0,
        ],
    }
    return threshold_grids


@pytest.fixture(scope="module")
def metric_functions():
    return ["precision", "recall", "f1", "accuracy"]


@pytest.fixture(scope="module")
def attack_runner(dataset, target_models, attacks, threshold_grids):
    return AttackRunner(dataset, target_models, attacks, threshold_grids)


def test_dummy_dataset(dataset):
    assert dataset.get_num_rows() == 500


def test_prepare_target_and_attack_data(dataset, dataset_split_ratios):
    dataset.prepare_target_and_attack_data(42, dataset_split_ratios)
    assert len(dataset.splits) == 7
    target_model_data = dataset.target_model_data
    attack_model_data = dataset.attack_model_data
    assert target_model_data is not None
    assert attack_model_data is not None
    assert target_model_data.X_target_train.get_shape() == (200, 30)
    assert attack_model_data.X_attack_test.get_shape() == (199, 30)


def test_run_attack(attack_runner, metric_functions):
    cache_input = (
        AttackType.MorganAttack in attack_runner.attacks
        or AttackType.CombinedBlackBoxAttack in attack_runner.attacks
    )

    attack_runner.train_target_models()
    target_result_string_0 = attack_runner.target_model_result_strings[
        attack_runner.target_models[0].get_model_name()
    ]
    target_result_string_1 = attack_runner.target_model_result_strings[
        attack_runner.target_models[1].get_model_name()
    ]
    target_result_string_2 = attack_runner.target_model_result_strings[
        attack_runner.target_models[2].get_model_name()
    ]

    target_result_string_0_test_f1 = target_result_string_0.split()[2]
    assert 0.4648744113029828 == pytest.approx(float(target_result_string_0_test_f1))

    target_result_string_1_test_f1 = target_result_string_1.split()[2]
    assert 0.4733890801770782 == pytest.approx(float(target_result_string_1_test_f1))

    target_result_string_2_test_f1 = target_result_string_2.split()[2]
    assert 0.46529411764705875 == pytest.approx(float(target_result_string_2_test_f1))

    result_attacks = []
    for target_model in attack_runner.target_models:
        for attack_type in attack_runner.attacks:
            result_attack = attack_runner.run_attack(
                target_model, attack_type, metric_functions, cache_input=cache_input
            )
            result_attacks.append(result_attack)

    attack_result_0_accuracy = float(result_attacks[0].split()[4])
    assert 0.8190954773869347 == pytest.approx(attack_result_0_accuracy)

    attack_result_1_accuracy = float(result_attacks[1].split()[4])
    assert 0.8743718592964824 == pytest.approx(attack_result_1_accuracy)

    attack_result_2_accuracy = float(result_attacks[2].split()[4])
    assert 0.8341708542713567 == pytest.approx(attack_result_2_accuracy)

    attack_result_3_accuracy = float(result_attacks[3].split()[4])
    assert 0.8241206030150754 == pytest.approx(attack_result_3_accuracy)

    attack_result_4_accuracy = float(result_attacks[4].split()[4])
    assert 0.7989949748743719 == pytest.approx(attack_result_4_accuracy)

    attack_result_5_accuracy = float(result_attacks[5].split()[4])
    assert 0.8944723618090452 == pytest.approx(attack_result_5_accuracy)

    attack_result_6_accuracy = float(result_attacks[6].split()[4])
    assert 0.9296482412060302 == pytest.approx(attack_result_6_accuracy)

    attack_result_7_accuracy = float(result_attacks[7].split()[4])
    assert 0.8894472361809045 == pytest.approx(attack_result_7_accuracy)

    attack_result_8_accuracy = float(result_attacks[8].split()[4])
    assert 0.507537688442211 == pytest.approx(attack_result_8_accuracy)

    attack_result_9_accuracy = float(result_attacks[9].split()[4])
    assert 0.5376884422110553 == pytest.approx(attack_result_9_accuracy)

    attack_result_10_accuracy = float(result_attacks[10].split()[4])
    assert 0.5025125628140703 == pytest.approx(attack_result_10_accuracy)

    attack_result_11_accuracy = float(result_attacks[11].split()[4])
    assert 0.49246231155778897 == pytest.approx(attack_result_11_accuracy)

    attack_result_12_accuracy = float(result_attacks[12].split()[4])
    assert 0.5025125628140703 == pytest.approx(attack_result_12_accuracy)

    attack_result_13_accuracy = float(result_attacks[13].split()[4])
    assert 0.4824120603015075 == pytest.approx(attack_result_13_accuracy)

    attack_result_14_accuracy = float(result_attacks[14].split()[4])
    assert 0.5025125628140703 == pytest.approx(attack_result_14_accuracy)

    attack_result_15_accuracy = float(result_attacks[15].split()[4])
    assert 0.507537688442211 == pytest.approx(attack_result_15_accuracy)

    attack_result_16_accuracy = float(result_attacks[16].split()[4])
    assert 0.6482412060301508 == pytest.approx(attack_result_16_accuracy)

    attack_result_17_accuracy = float(result_attacks[17].split()[4])
    assert 0.6331658291457286 == pytest.approx(attack_result_17_accuracy)

    attack_result_18_accuracy = float(result_attacks[18].split()[4])
    assert 0.5025125628140703 == pytest.approx(attack_result_18_accuracy)

    attack_result_19_accuracy = float(result_attacks[19].split()[4])
    assert 0.5226130653266332 == pytest.approx(attack_result_19_accuracy)

    attack_result_20_accuracy = float(result_attacks[20].split()[4])
    assert 0.6432160804020101 == pytest.approx(attack_result_20_accuracy)

    attack_result_21_accuracy = float(result_attacks[21].split()[4])
    assert 0.6331658291457286 == pytest.approx(attack_result_21_accuracy)

    attack_result_22_accuracy = float(result_attacks[22].split()[4])
    assert 0.6381909547738693 == pytest.approx(attack_result_22_accuracy)

    attack_result_23_accuracy = float(result_attacks[23].split()[4])
    assert 0.628140703517588 == pytest.approx(attack_result_23_accuracy)
