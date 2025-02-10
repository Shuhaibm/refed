import json
import os
import random
import concurrent.futures

from openai import AzureOpenAI,OpenAI

import tqdm
from tenacity import retry

from openai import AzureOpenAI
import tiktoken
from json_repair import repair_json

from prompts import *

class ReferenceLevelFeedbackSynthesizer():
    def __init__(self, reference_samples_with_feedback, teacher_model="gpt-4o-mini", output_dir="./output_dir"):
        """
        Initialize the synthesizer with reference samples, feedback and teacher model.

        Args:
            reference_samples_with_feedback (list): reference samples and their feedback
            teacher_model (str): name of teacher model to use for data synthesis
            output_dir (str): directory to save the synthesized data
        """
        self.reference_samples_with_feedback = reference_samples_with_feedback
        self.teacher_model = teacher_model
        self.output_dir = output_dir

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_VERSION")
        )

        self.temperature,self.top_p = 1.0,1.0

        self.chatgpt_encoder = tiktoken.encoding_for_model(teacher_model)
        self.total_input_tokens_submitted_to_chatgpt,self.total_output_tokens_received_from_chatgpt = 0,0

        self.seen = set()
    
    def update_costs(self, input, output):
        """
        Update token counters for cost tracking.

        Args:
            input (str): input prompt sent to the model
            output (str): response received from the model
        """
        num_input_tokens = len(self.chatgpt_encoder.encode(input))
        self.total_input_tokens_submitted_to_chatgpt += num_input_tokens
        num_output_tokens = len(self.chatgpt_encoder.encode(output))
        self.total_output_tokens_received_from_chatgpt += num_output_tokens

    @retry
    def azure_openai_completion(self, prompt, model_name, temperature, max_tokens, top_p, stop=None):
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
        messages = [{"role": "user", "content": prompt}]

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
            self.update_costs(prompt, response)

            return {"text": response}
        except Exception as e:
            if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry." in str(e):
                print(f"Error in API cll: {e}")
                return {"text": ""}
            else:
                raise

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
            model_name=self.teacher_model,
            temperature =1.0,
            max_tokens=4096,
            top_p=1.0
        )
        return response["text"].strip()

    def synthesize_instructions(self, instruction, instruction_feedback):
        """
        Generate new instructions based on a reference instruction and its feedback.

        Args:
            instruction (str): original reference instruction
            instruction_feedback (str): instruction reference-level feedback

        Returns:
            list: newly synthesized instructions
        """
        synthesized_instrs = []
        if instruction_feedback == "": return synthesized_instrs

        instr_generation_prompt = get_instruction_generation_prompt(instruction, instruction_feedback)
 
        generated_instructions = self.ask_gpt(instr_generation_prompt)
        generated_instructions = repair_json(generated_instructions, return_objects=True)["instructions"]

        return generated_instructions

    def synthesize_responses(self, reference_instruction, reference_response, instructions, response_feedback):
        """
        Generate and improve responses for synthesized instructions.

        Args:
            reference_instruction (str): original instruction used as reference
            reference_response (str): original response used as reference
            instructions (list): list of synthesized instructions
            response_feedback (str): response reference-level feedback

        Returns:
            list: synthesized instruction-response pairs along with analysis and explanations
        """
        synthesized_instr_response_pairs = []
        for instruction in instructions:
            if instruction == "" or response_feedback == "": continue

            response_generation_prompt = get_response_generation_prompt(instruction, reference_instruction, reference_response)
            response = self.ask_gpt(response_generation_prompt)
            response = repair_json(response, return_objects=True)["response"]
            if response == "": continue


            improved_response_prompt = get_improved_response_prompt(instruction, response, response_feedback)
            improved_response = self.ask_gpt(improved_response_prompt)
            improved_response = repair_json(improved_response, return_objects=True)
            if improved_response == "": continue

            synthesized_instr_response_pairs.append({
                "instruction": instruction,
                "response": response,

                "analysis": improved_response["analysis"],
                "implementation_strategy": improved_response["implementation_strategy"],
                "improved_response": improved_response["improved_response"],
            })
        
        return synthesized_instr_response_pairs

    def synthesize_data(self, num_samples_to_generate):
        """
        Uses reference-level feedback to synthesize new instruction-response pairs.
        Uses parallel processing to generate samples efficiently.

        Args:
            num_samples_to_generate (int): target number of samples to synthesize

        Returns:
            list: collection of synthesized instruction-response pairs
        """
        synthesized_data = []
        filepath = os.path.join(self.output_dir, "synthesized_data.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as file: synthesized_data = json.load(file)

        print(f"Loaded {len(synthesized_data)} synthesized data")
        progress_bar = tqdm.tqdm(total=num_samples_to_generate)
        if synthesized_data: progress_bar.update(len(synthesized_data))

        def process_single_request():
            unseen_samples = [elem for elem in self.reference_samples_with_feedback if elem["instruction"] not in self.seen]
            response_with_feedback = random.choice(unseen_samples)
            instruction,reference_response = response_with_feedback["instruction"],response_with_feedback["reference_response"]
            instruction_feedback_subject,instruction_feedback_skill,response_feedback = response_with_feedback["instruction_feedback_subject"],response_with_feedback["instruction_feedback_skill"],response_with_feedback["response_feedback"]
            curr_synthesized_data = []

            if instruction in self.seen: return curr_synthesized_data
            self.seen.add(instruction)

            for feature in [instruction_feedback_subject, instruction_feedback_skill]:
                # 1. Synthesize instructions for skill/subject feedback
                synthesized_instructions = self.synthesize_instructions(instruction, feature)

                # 2. Synthesize responses for skill/subject feedback
                curr_synthesized_data += self.synthesize_responses(instruction, reference_response, synthesized_instructions, response_feedback)

            return curr_synthesized_data
        
        request_idx = 0
        curr_goal = 5000
        while len(synthesized_data) < num_samples_to_generate:
            request_idx += 1
            num_workers = 40
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                if len(self.seen) == len(self.reference_samples_with_feedback): self.seen = set()

                futures = [executor.submit(process_single_request) for _ in range(num_workers)]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        curr_synthesized_data = future.result()
                        synthesized_data += curr_synthesized_data
                        
                        print(f"Generated {len(curr_synthesized_data)} instruction response pairs")
                        progress_bar.update(len(curr_synthesized_data))

                        if len(synthesized_data) >= curr_goal:
                            curr_goal += 5000
                            print(f"\n\n\nSaving generated data to {filepath}")
                            with open(filepath, 'w') as f:
                                json.dump(synthesized_data, f, indent=4)
                            print(f"\n\n\nTotal cost for synthesis:\n    Total input tokens: {self.total_input_tokens_submitted_to_chatgpt}\n    Total output tokens: {self.total_output_tokens_received_from_chatgpt}\n    Total cost: {self.total_input_tokens_submitted_to_chatgpt*0.150/1000000 + self.total_input_tokens_submitted_to_chatgpt*0.600/1000000}")


                    except Exception as e:
                        print(f"Error processing request {request_idx}: {e}")

        
        print(f"\n\n\nSaving generated data to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(synthesized_data, f, indent=4)

        
        print(f"\n\n\nTotal cost for synthesis:\n    Total input tokens: {self.total_input_tokens_submitted_to_chatgpt}\n    Total output tokens: {self.total_output_tokens_received_from_chatgpt}\n    Total cost: {self.total_input_tokens_submitted_to_chatgpt*0.150/1000000 + self.total_input_tokens_submitted_to_chatgpt*0.600/1000000}")

        return synthesized_data
