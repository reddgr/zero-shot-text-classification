import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, HBox, VBox, Output, Text, HTML
import textwrap
from datasets import Dataset, DatasetDict, ClassLabel, concatenate_datasets, load_dataset

class NLILabelingWidget:
    def __init__(self, candidate_labels):
        self.labeled_data = pd.DataFrame(columns=["text", "category", "label"])
        self.label_map = {0: "contradiction",1: "neutral",2: "entailment"}
        self.session_complete = False
        self.candidate_labels = candidate_labels
        self.top_category = None
        self.second_category = None
        self.bottom_category = None
        self.text = None
    
    def manual_labeling(self, df, classifier):
        """
        Manual labeling function for user interaction. Returns labeled_data when the session ends.
        """

        output = Output()

        spinner = HTML(
            value="""
            <div style="display: flex; justify-content: center; align-items: center; margin: 10px;">
                <div style="
                    border: 4px solid rgba(0, 0, 0, 0.1);
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    border-left-color: black;
                    animation: spin 1s linear infinite;
                "></div>
            </div>
            <style>
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            </style>
            """
        )
        spinner.layout.display = 'none'  # Initially hidden

        current_index = {"value": 0}  # Track the current index in df

        # Create buttons
        correct_button = Button(description="CORRECT", button_style="success")
        pass_button = Button(description="NEUTRAL", button_style="info")
        wrong_button = Button(description="WRONG", button_style="danger")
        skip_button = Button(description="SKIP", button_style="info")
        end_button = Button(description="END SESSION", button_style="warning")
        text_box = Text(description="Other entailment:", placeholder="Enter category")
        text_box_cont = Text(description="Other contradiction:", placeholder="Enter category")
        text_box_n = Text(description="Other neutral:", placeholder="Enter category")

        # Bind button actions
        def on_correct_button_clicked(_):
            label_text(nli_label=2, category=self.top_category)  # Entailment
            label_text(nli_label=0, category=self.bottom_category)  # Contradiction
            add_entailment_record_if_applicable()
            add_contradiction_record_if_applicable()
            text_box.value = ""  # Clear the text box
            text_box_cont.value = ""  # Clear the text box
            text_box_n.value = ""  # Clear the text box
            current_index["value"] += 1
            display_text()

        def on_wrong_button_clicked(_):
            label_text(nli_label=0, category=self.top_category)  # Contradiction
            add_entailment_record_if_applicable()
            add_contradiction_record_if_applicable()
            text_box.value = ""  # Clear the text box
            text_box_cont.value = ""  # Clear the text box
            text_box_n.value = ""  # Clear the text box
            current_index["value"] += 1
            display_text()

        def on_neutral_button_clicked(_):
            label_text(nli_label=1, category=self.top_category)  # Neutral
            label_text(nli_label=0, category=self.bottom_category)  # Contradiction
            add_entailment_record_if_applicable()
            add_contradiction_record_if_applicable()
            add_neutral_record_if_applicable()
            text_box.value = ""  # Clear the text box
            text_box_cont.value = ""  # Clear the text box
            text_box_n.value = ""  # Clear the text box
            current_index["value"] += 1
            display_text()
        
        def on_skip_button_clicked(_):
            text_box.value = ""  # Clear the text box
            text_box_cont.value = ""  # Clear the text box
            text_box_n.value = ""  # Clear the text box
            current_index["value"] += 1
            display_text()

        def on_end_button_clicked(_):
            output.clear_output(wait=True)
            with output:
                print("### Labeling Session Ended ###")
                print(f"Candidate labels: {self.candidate_labels}\nTotal labels recorded: {len(self.labeled_data)}")
                print("Labeled data:")
                display(self.labeled_data)
                self.session_complete = True

        # Add an additional entailment record if the user has entered a custom category
        def add_entailment_record_if_applicable():
            """
            If the user has entered a category in the text box, add an entailment record
            with the entered category.
            """
            custom_category = text_box.value.strip()  # Get the entered text
            if custom_category:
                # text = df.iloc[current_index["value"]]["text"]
                self.labeled_data = pd.concat(
                    [self.labeled_data, pd.DataFrame({"text": [self.text], "category": [custom_category], "label": [2]})],
                    ignore_index=True,
                )

        # Add an additional contradiction record if the user has entered a custom category
        def add_contradiction_record_if_applicable():
            """
            If the user has entered a category in the text box, add a contradiction record
            with the entered category.
            """
            custom_category = text_box_cont.value.strip()
            if custom_category:
                # text = df.iloc[current_index["value"]]["text"]
                self.labeled_data = pd.concat(
                    [self.labeled_data, pd.DataFrame({"text": [self.text], "category": [custom_category], "label": [0]})],
                    ignore_index=True,
                )

        # Add an additional neutral record if the user has entered a custom category
        def add_neutral_record_if_applicable():
            """
            If the user has entered a category in the text box, add a neutral record
            with the entered category.
            """
            custom_category = text_box_n.value.strip()
            if custom_category:
                # text = df.iloc[current_index["value"]]["text"]
                self.labeled_data = pd.concat(
                    [self.labeled_data, pd.DataFrame({"text": [self.text], "category": [custom_category], "label": [1]})],
                    ignore_index=True,
                )

        correct_button.on_click(on_correct_button_clicked)
        wrong_button.on_click(on_wrong_button_clicked)
        pass_button.on_click(on_neutral_button_clicked)
        skip_button.on_click(on_skip_button_clicked)
        end_button.on_click(on_end_button_clicked)

        # Display the interface once
        display(VBox([
            output,
            HBox([correct_button, pass_button, wrong_button]),
            HBox([skip_button]),
            spinner,
            text_box,
            text_box_cont,
            text_box_n,
            HBox([end_button], layout={'justify_content': 'flex-end'}),
        ]))

        def display_text():
            """
            Function to display the current text and prediction.
            """
            spinner.layout.display = 'block' # Show the spinner
            output.clear_output(wait=True)  # Clear the output area for the current example
            with output:
                if current_index["value"] >= len(df):
                    print("### Labeling Complete ###")
                    print("Labeled data:")
                    display(self.labeled_data)
                    self.session_complete = True
                    return
                self.text = df.iloc[current_index["value"]]["text"]
                result = classifier(self.text, self.candidate_labels, multi_label=False)

                # Display top three labels with their probabilities
                wrapped_text = textwrap.fill(self.text, width=120)
                self.top_category = result["labels"][0] # Updating the global variable so it can be used by label_text function
                top_category_score = result["scores"][0]
                self.second_category = result["labels"][1] # Updating the global variable so it can be used by label_text function
                second_category_score = result["scores"][1]
                self.bottom_category = result["labels"][-1] # Updating the global variable so it can be used by label_text function
                bottom_category_score = result["scores"][-1]
                print(wrapped_text)
                print("\n### Top-2 classes (only top-1 will be entailment) ###")
                # print(" - ".join([f"{label}: {score:.3f}" for label, score in zip(result["labels"][:3], result["scores"][:3])]))
                print(f'{self.top_category} - {top_category_score:.2f} | {self.second_category} - {second_category_score:.2f}')
                print("\n### Less likely class (contradiction) ###")
                print(f'{self.bottom_category} - {bottom_category_score:.2f}')
            spinner.layout.display = 'none'  # Hide spinner after processing

        def label_text(nli_label, category):
            # text = df.iloc[current_index["value"]]["text"]
            # result = classifier(self.text, self.candidate_labels, multi_label=False)
            # highest_class = result["labels"][0]  # Class with the highest probability
            # 2 for entailment, 0 for contradiction, 1 for neutral
            self.labeled_data = pd.concat(
                [self.labeled_data, pd.DataFrame({"text": [self.text], "category": [category], "label": [nli_label]})],
                ignore_index=True,
            )

        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
        # Initialize by displaying the first text
        display_text()


    def cast_label_to_classlabel(self, dataset):
        class_label = ClassLabel(names=[self.label_map[i] for i in sorted(self.label_map.keys())])
        # Map the 'label' feature to the new ClassLabel feature
        def map_labels(example):
            example['label'] = class_label.str2int(self.label_map[example['label']])
            return example
        dataset = dataset.map(map_labels)
        dataset = dataset.cast_column("label", class_label)
        return dataset

    def update_dataset(self, dataset_name, split_name, hf_token, new_dataset_records=None):
        """
        Updates a HuggingFace dataset with the labeled data or a custom dataframe.

        Parameters:
        - dataset_name (str): The name of the dataset on the HuggingFace Hub.
        - split_name (str): The split of the dataset to update ('train' or 'test').
        - hf_token (str): The HuggingFace token for authentication.
        - new_dataset_records (Optional[pandas.DataFrame or Dataset], optional): A custom dataframe or dataset with new records to add. Defaults to None.
        """
        if not new_dataset_records:
            new_dataset_records = Dataset.from_pandas(self.labeled_data)
        else:
            new_dataset_records = new_dataset_records
        dataset = load_dataset(dataset_name, token=hf_token)
        new_dataset_records = self.cast_label_to_classlabel(new_dataset_records)
        updated_split = concatenate_datasets([dataset[split_name], new_dataset_records])
        updated_dataset = DatasetDict({
            'train': dataset['train'] if split_name == 'test' else updated_split,
            'test': dataset['test'] if split_name == 'train' else updated_split
        })
        updated_dataset.push_to_hub(dataset_name, token=hf_token)
        print(f"Successfully pushed {len(new_dataset_records)} records to {dataset_name} {split_name} split.")