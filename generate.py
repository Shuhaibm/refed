import argparse
import json

from ReferenceLevelFeedbackCollector import ReferenceLevelFeedbackCollector
from ReferenceLevelFeedbackSynthesizer import ReferenceLevelFeedbackSynthesizer

def main(teacher_model, seed_dataset_name, size, output_dir):
    """
    Runs the reference-level feedback guided data synthesis pipeline.
    
    Args:
        teacher_name (str): name of the teacher model used to collect feedback
        seed_dataset_name (str): name of the seed dataset used for reference samples
        output_dir (str): directory to save the feedback files
        size (int): target number of samples to synthesize
    """

    # Reference-level feedback collection
    referenceLevelFeedbackCollector = ReferenceLevelFeedbackCollector(teacher_model, seed_dataset_name, output_dir)
    reference_samples_with_feedback = referenceLevelFeedbackCollector.collect_feedback()

    # Data synthesis with reference-level feedback
    referenceLevelFeedbackSynthesizer = ReferenceLevelFeedbackSynthesizer(reference_samples_with_feedback, teacher_model, output_dir)
    referenceLevelFeedbackSynthesizer.synthesize_data(size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str)
    parser.add_argument("--seed_dataset_name", type=str)
    parser.add_argument("--size", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    print(args)

    main(args.teacher_model, args.seed_dataset_name, int(args.size), args.output_dir)