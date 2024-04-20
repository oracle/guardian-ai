from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from guardian_ai.privacy_estimation.attack import BlackBoxAttack, AttackType
from guardian_ai.privacy_estimation.recommender_model import CFModel


class CFAttack(BlackBoxAttack):
    """
    This is the base class for the collaborative filtering model attack. It has a base estimator,
    which is a binary classifier that decides whether an attack data point was part of the
    original training data for the target model or not. It's black box because this type of attack
    can only access the prediction API of the target model and does not have access to the model parameters.
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
    
    @staticmethod
    def create_index_map(X, y):
        user_ids = sorted(list(set(X)))
        item_ids = sorted(list(set(y)))
        user_map = [{original_id: index for index, original_id in enumerate(user_ids)},
                    {index: original_id for index, original_id in enumerate(user_ids)}]
        item_map = [{original_id: index for index, original_id in enumerate(item_ids)},
                    {index: original_id for index, original_id in enumerate(item_ids)}]
        return user_map, item_map
    
    def transform_attack_data(
            self,
            target_model: CFModel,
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
        X_attack: pd.DataFrame
            pd.DataFrame containing a single column ``userID`` with user ids.
        y_attack: pd.DataFrame
            pd.DataFrame containing a single column ``interactions`` with items that the user interacted with.
        y_membership: array-like of shape (n_samples,)
            An array containing the membership labels, where 1 indicates membership in the training set, and 0
            indicates non-membership.
        split_type: str
            Use information cached from running the loss based and merlin attacks.
        use_cache: bool
            Using the cache or not.
        features: List[List[float]]
            A list of feature vectors representing items. This is required when attacking a recommender model.
        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features for the attack model, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        """
        users = X_attack.flatten().tolist()
        membership = y_membership
        user_attributes = []
        member_items = []
        member_users = []
        all_items = []
        items = y_attack.interactions.tolist()
        for user,item, label in zip(users, items, membership):
            if label == 1:
                member_items.append(item)
                member_users.append(user)
            all_items.append(item)
        member_item_list = [item for sublist in member_items for item in sublist]
        user_map, item_map = self.create_index_map(member_users, member_item_list)
        all_items_mapped = [item_map[0][item] for item in member_item_list]
        for user, items, label in zip(users, all_items, membership):
            if label == 1:
                user_mapped = user_map[0][user]
                items_mapped = [item_map[0][i] for i in items]
                recommendations_mapped = target_model.get_predictions_user(all_items_mapped, user_mapped, items_mapped)
                recommendations = [item_map[1][rec] for rec in recommendations_mapped]
            else:
                recommendations = target_model.get_most_popular(y_attack, items)
                
            top_k = len(recommendations)
            interaction_vector = np.mean([features[m] for m in items], axis=0)
            recommendation_vector = np.zeros(len(features[0]))
            for j in recommendations:
                weight = top_k - j
                recommendation_vector += weight * features[j]
            recommendation_vector /= sum(range(1, top_k + 1))
            vector_shadow = [i - w for i, w in zip(interaction_vector, recommendation_vector)]
            user_attributes.append(vector_shadow)
            X_membership = np.array(user_attributes)
        return X_membership
        
