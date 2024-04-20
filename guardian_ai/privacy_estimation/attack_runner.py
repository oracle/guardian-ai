#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/
from typing import Union
from guardian_ai.privacy_estimation.dataset import (
    ClassificationDataset,
    TargetModelData,
    AttackModelData,
)
from guardian_ai.privacy_estimation.dataset import CFDataset, TargetCFData
from guardian_ai.privacy_estimation.attack import (
    AttackType,
    LossBasedBlackBoxAttack,
    ConfidenceBasedBlackBoxAttack,
    ExpectedLossBasedBlackBoxAttack,
    ExpectedConfidenceBasedBlackBoxAttack,
    ThresholdClassifier,
)
from guardian_ai.privacy_estimation.combined_attacks import (
    CombinedBlackBoxAttack,
    CombinedWithMerlinBlackBoxAttack,
)
from guardian_ai.privacy_estimation.merlin_attack import MerlinAttack
from guardian_ai.privacy_estimation.morgan_attack import MorganAttack, MorganClassifier
from guardian_ai.privacy_estimation.collaborative_filtering_attack import CFAttack
from guardian_ai.privacy_estimation.model import TargetModel
from guardian_ai.privacy_estimation.recommender_model import CFModel
from typing import List, Dict
from sklearn.linear_model import LogisticRegression


class AttackRunner:
    """
    Class that can run the specified attacks against specified target models using the
    given dataset
    """
    def __init__(
            self,
            dataset: Union[ClassificationDataset, CFDataset],
            target_models: Union[List[CFModel], List[TargetModel]],
            attacks: List[AttackType],
            threshold_grids,
            shadow_models: List[CFModel] = None,
    ):
        """
        Initialize AttackRunner.

        Parameters
        ----------
        dataset: ClassificationDataset
            Dataset that has been split and prepared for running the attacks
        target_models: List[CFModel] | List[TargetModel]
            Target models to run the attacks against
        attacks: Dict[str:List[float]],
            List of attacks to run. Use the pattern AttackType.LossBasedBlackBoxAttack.name

        Returns
        -------
        AttackRunner
        """
        self.dataset = dataset
        assert self.dataset.target_model_data is not None
        assert self.dataset.attack_model_data is not None
        self.target_models = target_models
        self.shadow_models = shadow_models
        self.attacks = attacks
        self.threshold_grids = threshold_grids
        self.target_model_result_strings = {}
        self.shadow_model_result_strings = {}
        self.attack_cache = {}
    
    def train_collaborative_filtering_models(self):
        for target_model in self.target_models:
            assert isinstance(target_model, CFModel)
            print("Target Model: " + target_model.get_model_name())
            target_model_data: TargetCFData = self.dataset.target_model_data
            user_item_df = target_model.reindex(target_model_data.X_target_members,
                                                target_model_data.y_target_members)
            train, test = target_model.train_test_split(user_item_df)
            target_model.train_model(train, test)
            print("Target Model Train Evaluation: ")
            (hit_rate, ndcg) = target_model.metrics(test)
            print("Hit Rate:\t%f" % hit_rate,
                  "NDCG:\t%f" % ndcg, sep='\n')
            result_string = (
                    target_model.get_model_name()
                    + "\t"
                    + str(hit_rate)
                    + "\t"
                    + str(ndcg)
            )
            
            self.target_model_result_strings[
                target_model.get_model_name()
            ] = result_string
        
        for shadow_model in self.shadow_models:
            assert isinstance(shadow_model, CFModel)
            print("Target Model: " + shadow_model.get_model_name())
            shadow_model_data: TargetCFData = self.dataset.shadow_model_data
            user_item_df = shadow_model.reindex(shadow_model_data.X_target_members,
                                                shadow_model_data.y_target_members)
            train, test = shadow_model.train_test_split(user_item_df)
            shadow_model.train_model(train, test)
            print("Shadow Model Train Evaluation: ")
            (hit_rate, ndcg) = shadow_model.metrics(test)
            print("Hit Rate:\t%f" % hit_rate,
                  "NDCG:\t%f" % ndcg, sep='\n')
            result_string = (
                    target_model.get_model_name()
                    + "\t"
                    + str(hit_rate)
                    + "\t"
                    + str(ndcg)
            )
            
            self.shadow_model_result_strings[
                shadow_model.get_model_name()
            ] = result_string
    
    def train_target_models(self):
        for target_model in self.target_models:
            assert isinstance(target_model, TargetModel)
            print("Target Model: " + target_model.get_model_name())
            target_model_data: TargetModelData = self.dataset.target_model_data
            classifier = target_model.train_model(
                target_model_data.X_target_train, target_model_data.y_target_train
            )
            print("Target Model Train Evaluation: ")
            target_model.test_model(
                target_model_data.X_target_train, target_model_data.y_target_train
            )
            train_f1 = target_model.get_f1(
                target_model_data.X_target_train, target_model_data.y_target_train
            )
            print("Target Model Test Evaluation: ")
            target_model.test_model(
                target_model_data.X_target_test, target_model_data.y_target_test
            )
            test_f1 = target_model.get_f1(
                target_model_data.X_target_test, target_model_data.y_target_test
            )
            
            result_string = (
                    target_model.get_model_name()
                    + "\t"
                    + str(train_f1)
                    + "\t"
                    + str(test_f1)
            )
            
            self.target_model_result_strings[
                target_model.get_model_name()
            ] = result_string
    
    def _get_attack_object(
            self,
            attack_type: AttackType,
            target_model: TargetModel,  # need this for Morgan Attack
            use_cache: bool = False,
    ):
        """
        Instantiate the attack object of the specified attack_type. Some complex attack
        types may require training simpler attacks first if they have not been cached.

        Parameters
        ----------
        attack_type: AttackType
            Type of the attack to instantiate
        target_model: TargetModel
            Target model is required to train simpler attacks as needed
        use_cache: bool
            Use attacks previously cached

        Returns
        -------
        Attack
            Attack object
        """
        
        attack = None
        if attack_type == AttackType.LossBasedBlackBoxAttack:
            attack = LossBasedBlackBoxAttack(ThresholdClassifier())
        elif attack_type == AttackType.ExpectedLossBasedBlackBoxAttack:
            attack = ExpectedLossBasedBlackBoxAttack(LogisticRegression())
        elif attack_type == AttackType.ConfidenceBasedBlackBoxAttack:
            attack = ConfidenceBasedBlackBoxAttack(ThresholdClassifier())
        elif attack_type == AttackType.ExpectedConfidenceBasedBlackBoxAttack:
            attack = ExpectedConfidenceBasedBlackBoxAttack(LogisticRegression())
        elif attack_type == AttackType.MerlinAttack:
            attack = MerlinAttack(ThresholdClassifier())
        elif attack_type == AttackType.CollaborativeFilteringAttack:
            attack = CFAttack(LogisticRegression())
        elif attack_type == AttackType.CombinedBlackBoxAttack:
            if use_cache:
                loss_attack = self.attack_cache[AttackType.LossBasedBlackBoxAttack]
                confidence_attack = self.attack_cache[
                    AttackType.ConfidenceBasedBlackBoxAttack
                ]
                attack = CombinedBlackBoxAttack(
                    LogisticRegression(),
                    loss_attack=loss_attack,
                    confidence_attack=confidence_attack,
                )
            else:
                attack = CombinedBlackBoxAttack(LogisticRegression())
        elif attack_type == AttackType.CombinedWithMerlinBlackBoxAttack:
            if use_cache:
                loss_attack = self.attack_cache[AttackType.LossBasedBlackBoxAttack]
                confidence_attack = self.attack_cache[
                    AttackType.ConfidenceBasedBlackBoxAttack
                ]
                merlin_attack = self.attack_cache[AttackType.MerlinAttack]
                attack = CombinedWithMerlinBlackBoxAttack(
                    LogisticRegression(),
                    loss_attack=loss_attack,
                    confidence_attack=confidence_attack,
                    merlin_attack=merlin_attack,
                )
            else:
                merlin_attack = MerlinAttack(ThresholdClassifier())
                """
                Note that we don't need to train the Merlin attack for this to work. We just
                need the noise parameters etc. from Merlin attack to calculate the ratio
                """
                attack = CombinedWithMerlinBlackBoxAttack(
                    LogisticRegression(), merlin_attack=merlin_attack
                )
        elif attack_type == AttackType.MorganAttack:
            if use_cache:
                loss_attack = self.attack_cache[AttackType.LossBasedBlackBoxAttack]
                merlin_attack = self.attack_cache[AttackType.MerlinAttack]
            else:
                attack_model_data = self.dataset.attack_model_data
                # tune the loss-based attack and get the lower loss based threshold
                loss_attack = LossBasedBlackBoxAttack(ThresholdClassifier())
                loss_attack.train_attack_model(
                    target_model,
                    attack_model_data.X_attack_train,
                    attack_model_data.y_attack_train,
                    attack_model_data.y_membership_train,
                    self.threshold_grids[AttackType.LossBasedBlackBoxAttack.name],
                )
                # Similarly, train Merlin attack too
                merlin_attack = MerlinAttack(ThresholdClassifier())
                merlin_attack.train_attack_model(
                    target_model,
                    attack_model_data.X_attack_train,
                    attack_model_data.y_attack_train,
                    attack_model_data.y_membership_train,
                    self.threshold_grids[AttackType.MerlinAttack.name],
                )
            # careful, don't just cache the inputs here, because you'll also need to cache the test set by running
            # eval. Might be better to just use fresh values.
            
            loss_lower_threshold = loss_attack.attack_model.threshold
            merlin_threshold = merlin_attack.attack_model.threshold
            
            attack = MorganAttack(
                MorganClassifier(
                    loss_lower_threshold=loss_lower_threshold,
                    merlin_threshold=merlin_threshold,
                ),
                loss_attack=loss_attack,
                merlin_attack=merlin_attack,
            )
        else:
            raise Exception("This attack type is not supported.")
        return attack
    
    def run_attack(
            self,
            target_model: Union[TargetModel, CFModel],
            attack_type: AttackType,
            metric_functions: List[str],
            print_roc_curve: bool = False,
            cache_input: bool = False,
            features: List[List[float]] = None,
            shadow_model: CFModel = None
    ):
        """
        Instantiate the specified attack, trains and evaluates it, and prints out the result of
        the attack to an output result file, if provided.

        Parameters
        ----------
        target_model: TargetModel | CFModel
            Target model being attacked.
        attack_type: AttackType
            Type of the attack to run
        metric_functions: List[str]
            List of metric functions that we care about for evaluating the
            success of these attacks. Supports all sklearn.metrics that are relevant to binary
            classification, since the attack model is almost always a binary classifier.
        print_roc_curve: bool
            Print out the values of the tpr and fpr. Only works for
            trained attack classifiers for now.
        cache_input: bool
            Should we cache the input values - useful for expensive feature
            calculations like the merlin ratio.
        features: List[List[float]]
            Feature vectors representing the items - required when a recommender model
            is being attacked
        shadow_model: CFModel
            Trained shadow model that is used to generate the input features of an attack
            model - required when a recommender model is being attacked

        Returns
        -------
        str
            Result string
        """
        
        # figure out if we can use any of the previously cached values
        loss_exists = AttackType.LossBasedBlackBoxAttack in self.attack_cache.keys()
        confidence_exists = (
                AttackType.ConfidenceBasedBlackBoxAttack in self.attack_cache.keys()
        )
        merlin_ratio_exists = AttackType.MerlinAttack in self.attack_cache.keys()
        
        use_cache = False
        if attack_type == AttackType.MorganAttack:
            use_cache = loss_exists and merlin_ratio_exists
        if attack_type == AttackType.CombinedBlackBoxAttack:
            use_cache = loss_exists and confidence_exists
        if attack_type == AttackType.CombinedWithMerlinBlackBoxAttack:
            use_cache = loss_exists and confidence_exists and merlin_ratio_exists
        # Now, get the attack object
        attack = self._get_attack_object(attack_type, target_model, use_cache)
        
        # And, get the data needed to run the attack
        attack_model_data: AttackModelData = self.dataset.attack_model_data
        # train the attack
        if attack_type == AttackType.CollaborativeFilteringAttack:
            attack.train_attack_model(
                shadow_model,
                attack_model_data.X_attack_train,
                attack_model_data.y_attack_train,
                attack_model_data.y_membership_train,
                features=features
            )
        else:
            attack.train_attack_model(
                target_model,
                attack_model_data.X_attack_train,
                attack_model_data.y_attack_train,
                attack_model_data.y_membership_train,
                threshold_grid=self.threshold_grids.get(attack.name, None),
                cache_input=cache_input,
                use_cache=use_cache,
            )
        
        if cache_input:  # then cache the full attack
            self.attack_cache[attack.name] = attack
        
        # Evaluate the attack
        print(
            "Running "
            + attack.name
            + " against target model "
            + target_model.get_model_name()
        )
        print("Attack Metrics:")
        
        attack_metrics = attack.evaluate_attack(
            target_model,
            attack_model_data.X_attack_test,
            attack_model_data.y_attack_test,
            attack_model_data.y_membership_test,
            metric_functions,
            print_roc_curve=print_roc_curve,
            cache_input=cache_input,
            features=features
        )
        
        # Prepare the result string
        result_str = attack.name
        for i in range(len(attack_metrics)):
            result_str = result_str + "\t" + str(attack_metrics[i])
        result_str = result_str + "\n"
        print(result_str)
        return result_str
