import os
import pandas as pd
import textwrap
import random
from datasets import load_dataset
from IPython.display import display

class LMSYSChat1MHandler:
    def __init__(self, hf_token, streaming=False, verbose=True):
        self.hf_token = hf_token
        self.streaming = streaming
        self.lmsys_dataset = load_dataset(
            'lmsys/lmsys-chat-1m',
            revision="main",
            token=self.hf_token,
            streaming=self.streaming
        )
        self.verbose = verbose
        if verbose:
            print(self.lmsys_dataset)
        self.df_sample = None
        self.df_prompts = None
        self.unwrapped_turns_df = None

        if not self.streaming and verbose:
            print('Data is cached at:\n')
            for file_info in self.lmsys_dataset['train'].cache_files:
                filename = file_info['filename']
                file_size = os.path.getsize(filename)
                i = int((len(filename) - 41) / 2)
                print(f"Filename: {filename[:i]}*{filename[-41:]}\nSize: {file_size} bytes")

    def extract_df_sample(self, n_samples=None, conversation_ids=None):
        """
        Extracts a sample of conversations or specific conversations based on their conversation IDs.

        Parameters:
        - n_samples (int): Number of random samples to extract. Ignored if `conversation_ids` is provided.
        - conversation_ids (list): List of conversation IDs to extract. If provided, this takes precedence over `n_samples`.

        Returns:
        - pd.DataFrame: A DataFrame containing the extracted conversations.
        """
        if conversation_ids:
            # Filter conversations based on the provided conversation IDs
            df_sample = self.lmsys_dataset['train'].to_pandas()
            df_sample = df_sample[df_sample['conversation_id'].isin(conversation_ids)]
            print(f"Retrieved {len(df_sample)} conversations based on specified IDs")
        else:
            # Randomly sample conversations if no IDs are provided
            if not self.streaming:    
                df_sample = self.lmsys_dataset['train'].to_pandas().sample(n_samples)
                print(f"Retrieved {len(df_sample)} random conversations from lmsys/lmsys-chat-1m")
            else:
                # Take a sample from the streamed dataset
                streamed_samples = []
                for i, row in enumerate(self.lmsys_dataset['train']):
                    streamed_samples.append(row)
                    if i + 1 == n_samples:  # Collect only the desired number of samples
                        break
                # Shuffle and convert the collected samples to a Pandas DataFrame
                random.shuffle(streamed_samples)
                df_sample = pd.DataFrame(streamed_samples)
                
        self.df_sample = df_sample
        if self.verbose and len(df_sample) > 4:
            display(df_sample.head(2))
            print('...')
            display(df_sample.tail(2))
        return df_sample
    
    def parquet_sampling(self, n_samples):
        base_url = "https://huggingface.co/datasets/lmsys/lmsys-chat-1m/resolve/main/data/"
        data_files = [
            "train-00000-of-00006-4feeb3f83346a0e9.parquet",
            "train-00001-of-00006-4030672591c2f478.parquet",
            "train-00002-of-00006-1779b7cec9462180.parquet",
            "train-00003-of-00006-2fa862bfed56af1f.parquet",
            "train-00004-of-00006-18f4bdd50c103e71.parquet",
            "train-00005-of-00006-fe1acc5d10a9f0e2.parquet"
        ]
        sample_file = random.choice(data_files)
        print(f"Sampling from {sample_file}")
        data_files = {"train": base_url + sample_file}
        parquet_sample = load_dataset("parquet", data_files=data_files, split="train")
        df_sample = parquet_sample.to_pandas().sample(n_samples)
        print(f"Retrieved {len(df_sample)} random conversations from lmsys/lmsys-chat-1m/{sample_file}")
        self.df_sample = df_sample
        if self.verbose and len(df_sample) > 4:
            display(df_sample.head(2))
            print('...')
            display(df_sample.tail(2))
        return df_sample

    def add_turns_to_conversations(self):
        """
        Adds 'turn' keys to each conversation in the 'conversation' column of the dataframe.
        """
        self.df_sample['conversation'] = self.df_sample['conversation'].apply(
            lambda conv: Conversation(conv).add_turns()
        )
        df_with_turns = self.df_sample
        return df_with_turns
    
    def unwrap_turns(self):
        """
        Creates a dataframe where each row corresponds to a pair of user-assistant messages in a conversation and turn.
        The 'prompt' column contains the user's message, and the 'response' column contains the assistant's message.
        Each row includes a 'turn_id' column, which numbers the turns uniquely per conversation.
        """
        paired_data = []
        for _, row in self.df_sample.iterrows():
            conversation_id = row['conversation_id']
            row_data = row.to_dict()
            row_data.pop('conversation')  # Remove the 'conversation' field as it's being unwrapped

            current_prompt = None
            turn_id = None

            for message in row['conversation']:
                if message['role'] == 'user':
                    current_prompt = message['content']
                    turn_id = f"{conversation_id}{message['turn']:03}"  # Create turn_id
                elif message['role'] == 'assistant' and current_prompt is not None:
                    # Create a new row with the user-assistant pair
                    paired_row = {
                        **row_data,
                        'turn_n': message['turn'],
                        'prompt': current_prompt,
                        'response': message['content'],
                    }
                    paired_data.append(paired_row)
                    current_prompt = None  # Reset after pairing

        unwrapped_turns_df = pd.DataFrame(paired_data)
        unwrapped_turns_df.rename(columns={"turn": "conversation_turns"}, inplace=True) # The naming in the original dataset is ambiguous
        self.unwrapped_turns_df = unwrapped_turns_df
        return unwrapped_turns_df

    def extract_prompts(self, filter_language=None, min_char_length=20, max_char_length=500):
        """
        Extracts user prompts from the sample dataframe, optionally filtering by language and limiting the character length.

        Parameters:
        - filter_language (list of str or None): A list of specific languages to filter prompts by. If None, no language 
          filter is applied. Examples of valid values include ['English'], ['English', 'Portuguese'], or 
          ['Spanish', 'French', 'German'].
        - min_char_length (int): The minimum character length for user prompts to include. Defaults to 20.
        - max_char_length (int): The maximum character length for user prompts to include. Defaults to 500.

        Returns:
        - pd.DataFrame: A DataFrame containing extracted prompts with columns 'prompt' and 'language'.
        """
        df_sample = self.df_sample
        if filter_language:
            extracted_data = df_sample[df_sample['language'].isin(filter_language)].apply(
                lambda row: [
                    {'content': entry['content'], 'language': row['language']}
                    for entry in row['conversation']
                    if entry['role'] == 'user' and min_char_length <= len(entry['content']) <= max_char_length
                ], axis=1
            ).explode().dropna()
        else:
            extracted_data = df_sample.apply(
                lambda row: [
                    {'content': entry['content'], 'language': row['language']}
                    for entry in row['conversation']
                    if entry['role'] == 'user' and min_char_length <= len(entry['content']) <= max_char_length
                ], axis=1
            ).explode().dropna()

        df_prompts = pd.DataFrame(extracted_data.tolist())
        df_prompts.rename(columns={'content': 'prompt'}, inplace=True)
        self.df_prompts = df_prompts
        if self.verbose and len(df_sample) > 4:
            display(df_prompts.head(2))
            print('...')
            display(df_prompts.tail(2))
        return df_prompts
    
    def extract_prompt_sample(self):
        prompt_sample = self.df_prompts.sample(1)['prompt'].values[0]
        if self.verbose:
            wrapped_message = textwrap.fill(prompt_sample, width=120)
            print(wrapped_message)
        return prompt_sample
    
    def search_conversations(self, search_term):
        """
        Searches the dataset for a given string and returns a DataFrame with matching records.

        Parameters:
        - search_term (str): The string to search for in the dataset.

        Returns:
        - pd.DataFrame: A DataFrame containing conversations where the search term is found.
        """
        if self.streaming:
            raise ValueError("Search is not supported in streaming mode.")
        df = self.lmsys_dataset['train'].to_pandas()
        # Filter rows where the search term appears in the 'conversation' column
        matching_records = df[df['conversation'].apply(
            lambda conv: any(search_term.lower() in message['content'].lower() for message in conv)
        )]
        if self.verbose:
            print(f"Found {len(matching_records)} matching conversations for search term: '{search_term}'")
        return matching_records
        
    def print_language_counts(self, df):
        language_counts = df['language'].value_counts()
        print("Language Record Counts:")
        print(language_counts.to_frame('Count').reset_index().rename(columns={'index': 'Language'}))


class Conversation:
    def __init__(self, conversation_data):
        """
        Initializes the Conversation object with the conversation data.

        Parameters:
        - conversation_data (list): A list of dictionaries representing a conversation.
        """
        self.conversation_data = conversation_data

    def add_turns(self):
        """
        Adds a 'turn' key to each dictionary in the conversation,
        identifying the turn (pair of user and assistant messages).

        Returns:
        - list: The updated conversation with 'turn' keys added.
        """
        turn_counter = 0
        for message in self.conversation_data:
            if message['role'] == 'user':
                turn_counter += 1
            message['turn'] = turn_counter
        return self.conversation_data
    
    def pretty_print(self, user_prefix, assistant_prefix, width=80):
        """
        Prints the conversation with specified prefixes and wrapped text.

        Parameters:
        - user_prefix (str): Prefix to prepend to user messages.
        - assistant_prefix (str): Prefix to prepend to assistant messages.
        - width (int): Maximum characters per line for wrapping.
        """
        wrapper = textwrap.TextWrapper(width=width)
        
        for message in self.conversation_data:
            if message['role'] == 'user':
                prefix = user_prefix
            elif message['role'] == 'assistant':
                prefix = assistant_prefix
            else:
                continue  # Ignore roles other than 'user' and 'assistant'
            
            # Split on existing newlines, wrap each line, and join back with newlines
            wrapped_content = "\n".join(
                wrapper.fill(line) for line in message['content'].splitlines()
            )
            print(f"{prefix} {wrapped_content}\n")