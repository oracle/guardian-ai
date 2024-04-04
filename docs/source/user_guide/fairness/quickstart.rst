.. _quick-start-fairness:

Measurement with a Fairness Metric
==================================

Measure the Compliance of a Model with a Fairness Metric
--------------------------------------------------------

  .. code-block:: python

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from guardian_ai.fairness.metrics import ModelStatisticalParityScorer

    dataset = fetch_openml(name='adult', as_frame=True)
    df, y = dataset.data, dataset.target

    # Several of the columns are incorrectly labeled as category type in the original dataset
    numeric_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y.map({'>50K': 1, '<=50K': 0}).astype(int),
        train_size=0.7,
        random_state=0
    )

    sklearn_model = Pipeline(
        steps=[
            ("preprocessor", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", RandomForestClassifier()),
            ]
    )
    sklearn_model.fit(X_train, y_train)

    y_proba = sklearn_model.predict_proba(X_test)
    score = roc_auc_score(y_test, y_proba[:, 1])
    print(f'Score on test data: {score:.2f}')

    fairness_score = ModelStatisticalParityScorer(protected_attributes='sex')
    parity_test = fairness_score(sklearn_model, X_test)
    print(f'Statistical parity of the model (lower is better): {parity_test:.2f}')


Measure the Compliance of the True Labels of a Dataset with a Fairness Metric
-----------------------------------------------------------------------------

  .. code-block:: python

    from guardian_ai.fairness.metrics import DatasetStatisticalParityScorer
    from guardian_ai.fairness.metrics import dataset_statistical_parity
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    dataset = fetch_openml(name='adult', as_frame=True, version=1)
    df, y = dataset.data, dataset.target

    # Several of the columns are incorrectly labeled as category type in the original dataset
    numeric_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].astype(int)


    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y.map({'>50K': 1, '<=50K': 0}).astype(int),
        train_size=0.7,
        random_state=0
    )

    DSPS = DatasetStatisticalParityScorer(protected_attributes='sex')
    parity_test_data = DSPS(X=X_test, y_true=y_test)
    subgroups = X_test[['sex']]
    parity_test_data = dataset_statistical_parity(y_test, subgroups)
    print(f'Statistical parity of the test data (lower is better): {parity_test_data:.2f}')


Bias Mitigation
===============

.. code:: python

    from guardian_ai.fairness.bias_mitigation import ModelBiasMitigator
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder

    dataset = fetch_openml(name='adult', as_frame=True)
    df, y = dataset.data, dataset.target

    # Several of the columns are incorrectly labeled as category type in the original dataset
    numeric_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.75, random_state=12345
    )

    sklearn_model = Pipeline(
        steps=[
            ("preprocessor", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", RandomForestClassifier()),
            ]
    )
    sklearn_model.fit(X_train, y_train)

    bias_mitigated_model = ModelBiasMitigator(
        sklearn_model,
        protected_attribute_names="sex",
        fairness_metric="statistical_parity",
        accuracy_metric="balanced_accuracy",
    )

    bias_mitigated_model.fit(X_val, y_val)
    bias_mitigated_model.predict_proba(X_test)
    bias_mitigated_model.predict(X_test)
    bias_mitigated_model.tradeoff_summary_
    bias_mitigated_model.show_tradeoff(hide_inadmissible=False)
