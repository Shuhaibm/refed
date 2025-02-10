def get_instruction_feedback_prompt(instruction, response):
    return f"""# Task
Analyze the following instruction to extract key features that make it effective for instruction tuning. This analysis will help create high-quality instruction-response pairs for training language models to better follow instructions.

# Context
The analysis will be used to create training data for instruction tuning language models. Focus on qualities that help language models learn to:
- Better understand user intentions
- Recognize instruction patterns
- Generate appropriate responses

# Input
{{
  "instruction": {instruction},
  "reference_response": {response},
}}

# Output Format
```json
{{
  "subject_areas": str, # This should be a description of the relevant subject areas and domains the instruction covers
  "relevant_skills": str # This should be a description of the relevant skills required to provide a good response to the instruction
}}

# Analysis Guidelines
- Consider what makes this instruction clear and actionable
- Identify all relevant domains and skills
- Note structural elements that enhance instruction clarity

Output only a JSON object, in the format specified
"""

def get_response_feedback_prompt(instruction, response):
    return f"""# Task
Analyze the instruction-response pair and provide detailed feedback on how well it addresses the instruction. The feedback should:
- Highlight the specific qualities that make the response effective
- Provide actionable feedback for improvement

# Input
{{
  "instruction": {instruction},
  "reference_response": {response},
}}

# Evaluation Criteria
## Content Quality
- Accuracy and factual correctness
- Quality and depth of coverage

## Communication
- Clarity and comprehensiveness
- Logical flow, organization, and structure
- Appropriate quality and depth
- Engagement and tone

## Instruction Alignment
- How will it addresses the instruction
- Appropriate scope and focus
- Match with implied user needs

# Output Format
{{
  "response_feedback" : str # Feedback describing strengths of the response and how it can be improved
}}

Output only a JSON object, in the format specified."""

def get_instruction_generation_prompt(instruction, instruction_feedback):
  return f"""# Task
Generate 10 new instructions based on the provided instruction feature and sample. Each instruction should:
- Be of similar complexity and length to the sample instruction
- Be practical and reasonable to answer
- Be diverse and high-quality

# Sample Instruction:
{instruction}

# Instruction Features:
{instruction_feedback}

# Output Format 
```json
{{
  "instructions": list # List of 10 distinct instructions. Each instruction should be a single string.
}}
```

Output only a JSON object, in the format specified."""      

def get_response_generation_prompt(instruction, reference_instruction, reference_response):
  return f"""# Task
I will provide an instruction. Generate a high-quality, helpful response to the instruction. The response should demonstrate expertise, clear reasoning, and natural language use.

# Response Requirements
- Directly address all aspects of the instruction
- Response should demonstrate clear reasoning and expertise
- Use clear, natural language
- Include examples or evidence when relevant
- Show step-by-step reasoning where appropriate
- Maintain appropriate length and detail level
- Use proper formatting (lists, paragraphs) as needed

Here is an example of a response to an instruction:
# Sample Input Instruction:
{reference_instruction}
# Sample Response:
{reference_response}

# Output Format
{{
  "response": "The complete response text here"
}}

# Input
{{
  "instruction": {instruction},
}}

Generate a properly formatted JSON response, as specified by the Output Format, that addresses this instruction.
"""

def get_improved_response_prompt(instruction, response, response_feedback):
   return f"""# Task
Given an instruction-response pair and feedback, generate an improved version of the response by applying the feedback. The feedback was given for a similar but different instruction-response pair. Not all aspects of the feedback may be directly applicable, so make sure to only apply relevant aspects of the feedback.

# Input
{{
  "instruction": {instruction},
  "original_response": {response},
  "feedback": {response_feedback}
}}


# Quality Assessment Process
1. Analyze Original Response
- Core strengths and effective elements
- Structure and organization
- Depth and comprehensiveness
- Alignment with instruction

2. Evaluate Feedback
- Identify feedback points that are relevant to improving this response, and ignore points that are not relevant
- Identify actionable improvement suggestions
- Assess potential impact of each change
- Check alignment with original instruction
- Validate that suggested changes maintain or enhance quality

3. Improvement Strategy
- Prioritize changes with highest impact
- Preserve effective elements of the original response
- Ensure feedback applied enhance the response and do not remove valuable elements

# Output Format 
{{
    "analysis": {{
        "original_strengths": ["list of key effective elements to preserve"],
        "improvement_opportunities": ["list of specific areas that will benefit from enhancement"],
        "relevant_feedback": ["list of feedback points that are relevant and beneficial"]
    }},
    "implementation_strategy": {{
        "planned_changes": ["identify what feedback will be applied"],
        "rationale": "explain how this feedback will improve the original response"
    }},
    "improved_response": "The revised and improved response"
}}

Output only a JSON object, in the format specified."""
            