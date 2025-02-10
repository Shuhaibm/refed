# REFED (Reference-Level Feedback Enhanced Data) Dataset

<span style="font-variant: small-caps;"><strong>RE</strong>ference-Level <strong>F</strong>eedback <strong>E</strong>nhanced <strong>D</strong>ata</span> (<span style="font-variant: small-caps;">REFED</span>) is our dataset synthesized using the reference-level feedback framework. Using gpt-4o-mini as the teacher model and the LIMA training dataset (1K samples) as reference data, we synthesized this dataset for less than $20.

Data samples will follow this format:
```
{
    "instruction": str,
    "response": str,
    "analysis": {
        "original_strengths": list[str],
        "improvement_opportunities": list[str],
        "relevant_feedback": list[str]
    },
    "implementation_strategy": {
        "planned_changes": list[str],
        "rationale": str
    },
    "improved_response": str
}
```