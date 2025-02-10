import os
from tenacity import retry
from openai import AzureOpenAI
import json
from json_repair import repair_json
from datasets import load_dataset

from prompts import *


class ReferenceLevelFeedbackCollector():
    def __init__(self, teacher_name="gpt-4o-mini", seed_dataset_name="GAIR/lima", output_dir="./output_dir"):
        """
        Initialize the feedback collector with model and dataset configurations.

        Args:
            teacher_name (str): name of the teacher model used to collect feedback
            seed_dataset_name (str): name of the seed dataset used for reference samples
            output_dir (str): directory to save the feedback files
        """
        self.teacher_name = teacher_name

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_VERSION")
        )

        self.seed_dataset_name = seed_dataset_name
        self.seed_dataset = self.process_seed_dataset(seed_dataset_name)

    def process_seed_dataset(self, seed_dataset_name):
        """
        Preprocesses the seed dataset.

        Args:
            seed_dataset_name (str): name of dataset to load

        Returns:
            list: list of objects with instruction and response values
        """
        ds = load_dataset(seed_dataset_name)

        processed_data = []
        for elem in ds["train"]:
            conversations = elem["conversations"]
            if len(conversations)==2:
                instruction,response = conversations[0],conversations[1]

                processed_data.append({
                    "instruction": instruction,
                    "response": response
                })
        return processed_data

    @retry
    def azure_openai_completion(
        self,
        prompt,
        model_name,
        temperature,
        max_tokens,
        top_p,
        stop=None
    ):
        """
        Make an API call to Azure OpenAI for chat completion.

        Args:
            prompt (str): prompt for the model
            model_name (str): name of the model to use
            temperature (float): sampling temperature
            max_tokens (int): maximum number of tokens to generate
            top_p (float): nucleus sampling parameter
            stop (Optional[str]): stop sequence for text generation

        Returns:
            dict: Response containing generated text
        """
        messages = [{"role": "user", "content": prompt }]

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                logit_bias={"50256": -100},
            )
            response = response.choices[0].message.content
            return {"text": response.strip()}
        except Exception as e:
            if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry." in str(e):
                return {"text": ""}
            else:
                raise

    @retry
    def ask_gpt(self, prompt):
        """
        Submits a request to the teacher model

        Args:
            prompt (str): input prompt for the model

        Returns:
            str: Generated response
        """
        response = self.azure_openai_completion(
            prompt=prompt,
            model_name=self.teacher_name,
            temperature =1.0,
            max_tokens=4096,
            top_p=1.0
        )
        return response["text"]


    def collect_feedback(self):
        """
        Collects reference-level feedback on instructions and responses from the seed dataset.

        Returns:
            list: Collection of processed samples with their corresponding feedback
        """
        print(f"Beginning reference-level feedback collection for reference samples from {self.seed_dataset_name}\n\n")

        filepath = os.path.join(self.output_dir, "feedback.json")
        if os.path.exists(filepath):
            print(f"Loading pre-generated reference-level feedback from: {filepath}")
            with open(filepath, "r") as file: return json.load(file)

        samples_with_feedback = []
        for i,elem in enumerate(self.seed_dataset):
            instruction,response = elem["instruction"],elem["response"]

            instruction_feedback_prompt = get_instruction_feedback_prompt(instruction, response)
            response_feedback_prompt = get_response_feedback_prompt(instruction, response)

            print(f"Collecting feedback for reference sample {i}")

            try:
                instruction_feedback = self.ask_gpt(instruction_feedback_prompt)
                instruction_feedback = repair_json(instruction_feedback, return_objects=True)

                response_feedback = self.ask_gpt(response_feedback_prompt)
                response_feedback = repair_json(response_feedback, return_objects=True)["response_feedback"]


                samples_with_feedback.append({
                    "instruction": instruction,
                    "reference_response": response,

                    "instruction_feedback_subject": instruction_feedback["subject_areas"],
                    "instruction_feedback_skill": instruction_feedback["relevant_skills"],
                    "response_feedback": response_feedback
                })
            except Exception as e:
                print(f"Feedback collection failed: {e}")

        print(f"\n\nReference-level feedback collection completed, saving to: {filepath}")
        with open(filepath, "w") as file:
            json.dump(samples_with_feedback, file)

        return samples_with_feedback