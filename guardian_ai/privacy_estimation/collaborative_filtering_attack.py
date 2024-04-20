from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from guardian_ai.privacy_estimation.attack import BlackBoxAttack, AttackType
from guardian_ai.privacy_estimation.model import TargetModel


class CFAttack(BlackBoxAttack):
    """
    This is the base class for the collaborative filtering model attack. It has a base estimator,
    which is a binary classifier that decides whether an attack data point was part of the original training data for the target model
    or not. It's black box because this type of attack can only access the prediction API of
    the target model and does not have access to the model parameters.
    """
    
    def __init__(
            self,
            attack_model: BaseEstimator,
            name: str = "collaborative filtering attack",
    ):
        """
        Initialize the attack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
        name: str
            Name of this attack for reporting purposes.

        """
        self.name = name
        self.attack_model = attack_model
        self.X_membership_train = None  # Useful for caching the feature values for the attack (e.g. Morgan attack)
        self.X_membership_test = None
    
    def transform_attack_data(
            self,
            target_model: TargetModel,
            X_attack,
            y_attack,
            y_membership,
            split_type: str = None,
            use_cache: bool = False,
            features: List[List[float]] = None,
    ):
        """
        This is the central method in designing the attack, and captures the attacker's
        hypothesis about the membership of a data point in the training dataset of the target
        model. Its job is to derive signals from the original data that might be relevant to
        determining membership. Takes a dataset in the original format and converts it to the
        input variable for the attack. Think of it as feature engineering for building
        the attack model, which is essentially a binary classifier.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack: {array-like, sparse matrix} of shape (n_users, n_items)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y_attack: ndarray of shape (n_users, n_items)
            Vector containing the output labels of the attack data points (not membership label).
        y_membership: array of shape (n_users, 1)
            Vector containing the membership labels.
        split_type: str
            Use information cached from running the loss based and merlin attacks.
        use_cache: bool
            Using the cache or not.
        features: List[List[float]]
            Feature vectors of the items - required when the collaborative filtering model
            is being attacked
        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features for the attack model, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        """
        user_attributes = []
        for user, items, label in zip(X_attack, y_attack, y_membership):
            recommendations = target_model.get_predictions(user,
                                                           items) if label == 1 else target_model.get_most_popular(
                items)
            top_k = len(recommendations)
            interaction_vector = np.mean([features[m] for m in items], axis=0)
            recommendation_vector = np.zeros(len(features[0]))
            for j in recommendations:
                weight = top_k - j  # Weight decreases as rank increases
                recommendation_vector += weight * features[j]
            recommendation_vector /= sum(range(1, top_k + 1))
            vector_shadow = [i - w for i, w in zip(interaction_vector, recommendation_vector)]
            user_attributes.append(vector_shadow)
            X_membership = np.array(user_attributes)
        return X_membership
