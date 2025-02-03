from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import numpy as np
import os
from langdetect import detect
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, model_path, label_map, verbose = False):
        self.model_path = model_path
        self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)
        self.label_map = label_map
        if verbose: 
            self.print_device_information()
    
    def print_device_information(self):
        # Check device information
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_properties = torch.cuda.get_device_properties(0) if device.type == "cuda" else "CPU Device"

        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"Device Name: {device_properties.name}")
            # print(f"Compute Capability: {device_properties.major}.{device_properties.minor}")
            print(f"Total Memory: {device_properties.total_memory / 1e9:.2f} GB")

    def tokenize_and_trim(self, text):
        max_length = self.classifier.tokenizer.model_max_length
        inputs = self.classifier.tokenizer(text, truncation=True, max_length=max_length, return_tensors="tf")
        return self.classifier.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)


    def classify_dataframe_column(self, df, target_column, feature_suffix):

        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(self.tokenize_and_trim)

        results = []
        for text in tqdm(df[f'trimmed_{target_column}'].tolist(), desc="Classifying"):
            result = self.classifier(text)
            results.append(result[0])

        df[f'pred_label_{feature_suffix}'] = [self.label_map[int(result['label'].split('_')[-1])] for result in results]
        df[f'prob_{feature_suffix}'] = [result['score'] for result in results]
        df.drop(columns=[f'trimmed_{target_column}'], inplace=True)
        return df
    
    def test_model_predictions(self, df, target_column):
        """
        Tests model predictions on a given dataframe column and computes evaluation metrics.

        Args:
            df (pd.DataFrame): Input dataframe containing the data.
            target_column (str): The name of the column to classify.

        Requirements:
            - The dataframe must include a 'label' column for comparison with predictions.

        Returns:
            dict: A dictionary containing accuracy, F1 score, cross-entropy loss, 
                and the confusion matrix.
        """
        # Convert pandas dataframe to Dataset
        dataset = Dataset.from_pandas(df)

        # Define a processing function for tokenization and classification
        def process_data(batch):
            trimmed_text = self.tokenize_and_trim(batch[target_column])
            result = self.classifier(trimmed_text)
            score = result[0]['score']
            label = result[0]['label']
            return {
                'trimmed_text': trimmed_text,
                'predicted_prob_0': score if label == 'LABEL_0' else 1 - score,
                'predicted_prob_1': 1 - score if label == 'LABEL_0' else score,
            }

        # Apply processing with map
        processed_dataset = dataset.map(process_data, batched=False)

        # Convert back to pandas dataframe
        processed_df = processed_dataset.to_pandas()

        # Extract predicted probabilities and true labels
        predicted_probs = processed_df[['predicted_prob_0', 'predicted_prob_1']].values
        true_labels = df['label'].values

        # Calculate metrics
        accuracy = accuracy_score(true_labels, np.argmax(predicted_probs, axis=1))
        f1 = f1_score(true_labels, np.argmax(predicted_probs, axis=1), average='weighted')
        cross_entropy_loss = log_loss(true_labels, predicted_probs)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cross Entropy Loss: {cross_entropy_loss:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, np.argmax(predicted_probs, axis=1))
        cmap = plt.cm.Blues
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=cmap)
        plt.show()

        # Return metrics and probabilities for further inspection
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "cross_entropy_loss": cross_entropy_loss,
            "confusion_matrix": cm,
            "predicted_probs": predicted_probs  # Include reconstructed probabilities
        }
    
    
class LanguageDetector:
    def __init__(self, dataframe):
        """
        Initializes the LanguageDetector with the provided DataFrame.
        """
        self.dataframe = dataframe

    def detect_language_dataframe_column(self, target_column):
        """
        Detects the language of text in the specified column using langdetect and adds 
        a 'detected_language' column to the DataFrame.
        """
        def detect_language(text):
            try:
                return detect(text)
            except Exception:
                return None

        tqdm.pandas()
        self.dataframe['detected_language'] = self.dataframe[target_column].progress_apply(detect_language)

        return self.dataframe
    

# Classifier with Tensorflow backend
class TensorflowClassifier(Classifier):
    def __init__(self, model_path, label_map, verbose=False):
        super().__init__(model_path, label_map, verbose=False)
        self.is_tensorflow = False
        
        if self._is_tensorflow_model(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Adjust as per training tokenizer
            self.is_tensorflow = True
            if verbose:
                print("Loaded TensorFlow model.")
        else:
            if verbose:
                print("Fallback to HuggingFace pipeline.")

    def _is_tensorflow_model(self, model_path):
        return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb"))

    def classify(self, text):
        if self.is_tensorflow:
            inputs = self.tokenizer(text, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="np")
            logits = self.model.predict([inputs["input_ids"], inputs["attention_mask"]])
            probabilities = tf.nn.softmax(logits).numpy()
            label_id = np.argmax(probabilities, axis=-1).item()
            return {
                "label": f"LABEL_{label_id}",
                "score": probabilities.max()
            }
        else:
            return self.classifier(text)[0]

    def classify_dataframe_column(self, df, target_column, feature_suffix):
        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(
            lambda text: self.tokenizer.decode(
                self.tokenizer(text, truncation=True, max_length=self.tokenizer.model_max_length)["input_ids"],
                skip_special_tokens=True
            )
        )

        if self.is_tensorflow:
            results = [self.classify(text) for text in df[f'trimmed_{target_column}']]
        else:
            results = [self.classifier(text)[0] for text in df[f'trimmed_{target_column}']]

        df[f'pred_label_{feature_suffix}'] = [
            self.label_map[int(result['label'].split('_')[-1])] for result in results
        ]
        df[f'prob_{feature_suffix}'] = [result['score'] for result in results]
        df.drop(columns=[f'trimmed_{target_column}'], inplace=True)
        return df


class ZeroShotClassifier(Classifier):

    def __init__(self, model_path, tokenizer_path, candidate_labels):
        self.model_path = model_path
        self.candidate_labels = candidate_labels
        self.classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=tokenizer_path, clean_up_tokenization_spaces=True, device=0 if torch.cuda.is_available() else -1)

    def classify_text(self, text, top_n=None, multi_label=False):
        """
        Classify a single text using zero-shot classification with truncated scores.

        :param text: The text to classify
        :param multi_label: Whether to allow multi-label classification
        :return: Classification result as a dictionary with scores truncated to 3 decimals
        """
        classification_output = self.classifier(text, self.candidate_labels, multi_label=multi_label, clean_up_tokenization_spaces=True)
        classification_output['scores'] = [round(score, 3) for score in classification_output['scores']]
        if top_n is not None:
            classification_output = {
                'sequence': classification_output['sequence'],
                'labels': classification_output['labels'][:top_n],
                'scores': classification_output['scores'][:top_n]
            }
        return classification_output

    def classify_dataframe_column(self, df, target_column, feature_suffix, multi_label=False):
        """
        Classify the contents of a dataframe column using zero-shot classification.

        :param df: The dataframe to process
        :param target_column: The column containing text to classify
        :param feature_suffix: Suffix for the output columns
        :param multi_label: Whether to allow multi-label classification
        :return: The dataframe with classification results
        """
        tqdm.pandas()

        # Apply the classify_text method to each row
        results = df[target_column].progress_apply(
            lambda text: self.classify_text(text, multi_label=multi_label)
        )

        # Extract and store results
        df[f'top_class_{feature_suffix}'] = results.apply(lambda res: res['labels'][0])
        df[f'top_score_{feature_suffix}'] = results.apply(lambda res: res['scores'][0])
        df[f'full_results_{feature_suffix}'] = results.apply(lambda res: list(zip(res['labels'], res['scores'])))

        return df
    
    def test_zs_predictions(self, df, target_column='text', true_classes_column='category', plot_conf_matrix=True):
        """
        Tests model predictions on a given dataset column using the zero-shot classification pipeline.

        Args:
            df (pd.DataFrame): Input dataframe containing texts for zero-shot classification.
            target_column (str): The name of the column containing text to classify.
            true_classes_column (str): The column containing annotated classes.

        Returns:
            dict: A dictionary containing accuracy, F1 score, and confusion matrix.
        """
        # Progress bar for classification
        tqdm.pandas(desc=f"Zero-shot classification with {self.model_path}")
        
        # Function to classify each row
        def classify_row(row):
            classification_output = self.classifier(
                row[target_column],
                self.candidate_labels,
                multi_label=False,
                clean_up_tokenization_spaces=True,
            )
            return classification_output["labels"][0]

        # Apply classification with progress bar
        df = df.copy()
        df.loc[:, 'predicted_class'] = df.progress_apply(classify_row, axis=1)
        
        # Extract true and predicted classes
        true_classes = df[true_classes_column]
        predicted_classes = df['predicted_class']

        # Compute metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        f1 = f1_score(true_classes, predicted_classes, average="macro")
        cm = confusion_matrix(true_classes, predicted_classes, labels=self.candidate_labels)
        if plot_conf_matrix:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.candidate_labels)
            fig, ax = plt.subplots(figsize=(4, 4))
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            ax.set_title(f"Zero-shot classification with {self.model_path}", fontsize=10)
            ax.set_xlabel("Predicted label", fontsize=8) 
            ax.set_ylabel("True label", fontsize=8)   

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

            fig.text(
                0.5, 0.01, 
                f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}",
                ha="center",
                fontsize=10
            )
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust bottom margin
            plt.show()

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm,
            "detailed_results": df.to_dict(),  # Full dataframe with predictions
        }
    
    def test_zs_predictions_with_dataset(self, df, target_column='text', true_classes_column='category', plot_conf_matrix=True):
        dataset = Dataset.from_pandas(df)
        def classify_text(batch):
            classification_output = self.classifier(
                batch[target_column],
                self.candidate_labels,
                multi_label=False,
                clean_up_tokenization_spaces=True,
            )
            return {
                "predicted_class": classification_output["labels"][0],
                "predicted_scores": classification_output["scores"],
            }

        # Apply classification to the dataset
        classified_dataset = dataset.map(classify_text, batched=False)
        # classified_dataset = dataset.map(classify_text, batched=True, batch_size=16)

        # Extract true and predicted classes
        true_classes = classified_dataset[true_classes_column]
        predicted_classes = classified_dataset["predicted_class"]

        # Compute metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        f1 = f1_score(true_classes, predicted_classes, average="macro")

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Generate confusion matrix:
        cm = confusion_matrix(true_classes, predicted_classes, labels=self.candidate_labels)
        if plot_conf_matrix:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.candidate_labels)
            fig, ax = plt.subplots(figsize=(6, 6))  
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.xticks(rotation=45, ha="right") 
            plt.show()

        # Return metrics for further inspection
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm,
            "detailed_results": classified_dataset.to_dict(), 
        }
    
class MetricsComparison: 
    def __init__(self, base_classifier, fine_tuned_classifier, base_metrics, fine_tuned_metrics):
        self.base_classifier = base_classifier
        self.fine_tuned_classifier = fine_tuned_classifier
        self.base_metrics = base_metrics
        self.fine_tuned_metrics = fine_tuned_metrics

    def compare_conf_matrices(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Plot for base_classifier (left)
        disp1 = ConfusionMatrixDisplay(confusion_matrix=self.base_metrics["confusion_matrix"], 
                                       display_labels=self.base_classifier.candidate_labels)
        disp1.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
        axes[0].set_title(f"Zero-shot classification with {self.base_classifier.model_path}", fontsize=10)
        axes[0].set_xlabel("Predicted class", fontsize=8)
        axes[0].set_ylabel("True class", fontsize=8)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=8)

        fig.text(
            0.25, 0.01, 
            f"Accuracy: {self.base_metrics['accuracy']:.4f} | F1 Score: {self.base_metrics['f1_score']:.4f}",
            ha="center",
            fontsize=10
        )

        # Plot for zs_classifier (fine-tuned) (right)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=self.fine_tuned_metrics["confusion_matrix"], 
                                       display_labels=self.fine_tuned_classifier.candidate_labels)
        disp2.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
        axes[1].set_title(f"ZS classification with {self.fine_tuned_classifier.model_path}", fontsize=10)
        axes[1].set_xlabel("Predicted class", fontsize=8)
        axes[1].set_ylabel("True class", fontsize=8)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=8)

        fig.text(
            0.75, 0.01, 
            f"Accuracy: {self.fine_tuned_metrics['accuracy']:.4f} | F1 Score: {self.fine_tuned_metrics['f1_score']:.4f}",
            ha="center",
            fontsize=10
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
    
