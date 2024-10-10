from datasets import load_dataset
from .core import DataWithProtectedAttributes

class BOLDLoader:
    def __init__(self):
        """
        A class to load and process the BOLD dataset from Hugging Face datasets.

        The BOLD dataset is loaded using the `load_dataset` function, and the class provides
        functionality to filter the dataset based on a specified domain and return it in a 
        format suitable for handling protected attributes.
        """
        self.dataset = load_dataset("AlexaAI/bold")["train"]
        self.domains = set(self.dataset["domain"])

    def get_dataset(self, protected_domain):
        """
        Filters the dataset for a given domain and returns it as a DataWithProtectedAttributes object.

        Args:
            protected_domain (str) The domain to filter the dataset by. Must be one of the domains present in the dataset.

        Returns:
            DataWithProtectedAttributes
        """
        if protected_domain not in self.domains:
            raise ValueError(
                f"{protected_domain} is not supported by the dataset. Possible values {', '.join(self.domains)}"
            )
        
        filtered_dataset = self.dataset.filter(lambda example: example["domain"] == protected_domain).to_pandas()
        generated_dataset = filtered_dataset.explode("prompts").reset_index(drop=True)
        return DataWithProtectedAttributes(
            dataframe=generated_dataset,
            prompt_column="prompts",
            protected_attributes_columns=["category"]
        )


