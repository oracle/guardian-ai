****************
Measuring Bias in LLMs 
****************

**Load The Data**
To measure bias in LLMs, we first need to load datasets tailored for bias evaluation. Here, we use two datasets: BOLD and Holistic Bias.
The data is available at this [link](https://github.com/oracle/guardian-ai/tree/main/data). 
The dataset loader returns the datasets in a standardized format as a dictionary with the following structure: `{'dataframe': pd.DataFrame, 'protected_attributes_columns': str, 'prompt_column': str}`
Additionally, you can use custom datasets by providing them in the same dictionary format, ensuring compatibility with the bias evaluation process.

.. code:: python
    import os
    from guardian_ai.fairness.llm.dataloader import BOLDLoader, HolisticBiasLoader

    # Define the path to the downloaded data.
    path_to_data = "..."  # Replace with the actual path

    # Load the BOLD dataset (reference: https://arxiv.org/abs/2101.11718)
    bold_dataset_path = os.path.join(path_to_data, "BOLD")
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
    holistic_dataset_path = os.path.join(path_to_data, "holistic_bias")
    holistic_loader = HolisticBiasLoader(path_to_dataset=holistic_dataset_path)

    # Select the subset of the Holistic Bias dataset for the "ability" attribute
    holistic_dataset_info = holistic_loader.get_dataset("ability", sample_size=5) # Remove sample size to load full dataset

    # Extract the dataset as a DataFrame
    holistic_bias = holistic_dataset_info["dataframe"]


**Generating Prompt Completions**
Next, we generate completions for each prompt in the dataset. 
Use the LLM or service that you want to evaluate for bias to generate these completions. The result should be structured as a list of lists, 
where each inner list contains completions for a single prompt.

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
To evaluate bias in the generated completions, we use a large language model (LLM) to classify the text. Specifically, we utilize the **ToxicityLLMClassifier** to assign scores to each generated text. 
The process begins by starting a vLLM server using the Llama-3.1-70B-Instruct model:

.. code:: bash
    vllm serve meta-llama/Llama-3.1-70B-Instruct

By default, the server runs at `http://localhost:8000/v1`. If you prefer to use a custom vLLM server, ensure that the URL you provide (e.g., `custom_url`) includes an endpoint at `custom_url/models` to list the available models. For further guidance, visit the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).

Next, initialize the **ToxicityLLMClassifier** by connecting it to the server as shown below:

.. code:: python
    from guardian-ai.fairness.llm.model import VLLMServer
    from guardian_ai.fairness.llm.classifier import ToxicityLLMClassifier
    url="http://localhost:8000/v1" # Put your server url here
    model = "meta-llama/Llama-3.1-70B-Instruct"
    llm = VLLMServer(vllm_server_url=url, model=model)
    classifier = ToxicityLLMClassifier(llm=llm)

    classifier_scores = []
    for completion_set in completions:
        classifier_scores.append(classifier.score(completion_set))

**Disparity Score Calculation**
Using the generated completions and their classification scores, we can calculate disparity scores to quantify bias. 
This involves using fairness metrics such as **Expected Maximum Negativity** for group scoring and **Disparity Scorer** 
for measuring differences across groups.

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

    print('Disparity Score:', disparity_score)
    print('Group Scores:', group_scores)

**Output Example**
.. parsed-literal::
    Disparity Score: 0.3
    Group Scores: {'black': 0.3, 'hispanic': 0.6, 'white': 0.5, 'asian': 0.4, ...} 


**Interpreting Results**

- **Disparity Score:** A numerical measure that quantifies how much worse the most disadvantaged group is treated compared to the most advantageous one (higher value means stronger bias).
- **Group Scores:** Individual scores for each group.

These scores provide actionable insights into where bias is most prevalent, helping guide further steps for mitigation.
