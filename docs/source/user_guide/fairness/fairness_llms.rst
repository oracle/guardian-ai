****************
Measuring Bias in LLMs 
****************

**Load The Data**

.. code:: python
    import os
    from guardian_ai.fairness.llm.dataloader import BOLDLoader, HolisticBiasLoader

    # Define the root path of the repository
    repo_root = "..."  # Replace with the actual path to the root of your repository

    # Load the BOLD dataset (reference: https://arxiv.org/abs/2101.11718)
    bold_dataset_path = os.path.join(repo_root, "data", "BOLD")
    bold_loader = BOLDLoader(path_to_dataset=bold_dataset_path)

    # Select the subset of the BOLD dataset based on a protected attribute
    # Options: ["gender", "political_ideology", "profession", "race", "religious_ideology"]
    bold_dataset_info = bold_loader.get_dataset("race", sample_size=5) # Remove sample size to load full dataset

    # Extract relevant data from the dataset
    # The returned dictionary contains:
    # - "dataframe" (pd.DataFrame): the dataset as a DataFrame
    # - "prompt_column" (str): column name containing text prompts
    # - "protected_attributes_columns" (List[str]): column names for protected attributes
    bold = bold_dataset_info["dataframe"]

    # Load the Holistic Bias dataset (reference: https://arxiv.org/abs/2205.09209)
    holistic_dataset_path = os.path.join(repo_root, "data", "holistic_bias")
    holistic_loader = HolisticBiasLoader(path_to_dataset=holistic_dataset_path)

    # Select the subset of the Holistic Bias dataset for the "ability" attribute
    holistic_dataset_info = holistic_loader.get_dataset("ability", sample_size=5) # Remove sample size to load full dataset

    # Extract the dataset as a DataFrame
    holistic_bias = holistic_dataset_info["dataframe"]


**Generating Prompt Completions**
You can use any LLM or service to generate completions.
The resulting array of completions should be a list of lists, 
where each outer list contains completions (in the example below, 25) corresponding to a prompt in the dataset.

.. code:: python
    from transformers import pipeline

    # Initialize the text generation pipeline with the desired Hugging Face model
    pipe = pipeline("text-generation", model="<insert Hugging Face model ID here>")

    # Generate completions
    completions = []
    for prompt in bold[bold_dataset_info["prompt_column"]]:
        # Generate 25 completions per prompt and append them to the completions list
        completions.append(
            [generation['generated_text'] for generation in
            pipe(prompt, num_return_sequences=25)])


**Obtaining Classification Scores**

.. code:: python

    from guardian_ai.fairness.llm.classifier import ToxigenRoberta
    classifier = ToxigenRoberta()
    classifier_scores = []
    for completion_set in completions:
        classifier_scores.append(classifier.score(completion_set))

**Disparity Score Calculation**

.. code:: python
    from guardian_ai.fairness.llm.dataloader import BOLDLoader
    from guardian_ai.fairness.llm.evaluation import BiasEvaluator
    from guardian_ai.fairness.llm.metrics import DisparityScorer, ExpectedMaximumNegativityScorer
    
    group_scorer = ExpectedMaximumNegativityScorer()
    disparity_scorer = DisparityScorer()
    bias_evaluator = BiasEvaluator(group_scorer, disparity_scorer)

    disparity_score, group_scores = bias_evaluator(
        dataframe=dataframe,
        prompt_column=bold_dataset_info["prompt_column"],
        protected_attributes_columns=bold_dataset_info["protected_attributes_columns"],
        classifier_scores=classifier_scores,
    )

    disparity_score