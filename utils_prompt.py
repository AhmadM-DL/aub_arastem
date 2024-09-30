class PromptFactory:
    def __init__(self):
        self.prompt_types = {
            0: self._zero_shot
        }

    def get_prompt_function(self, n_shots):
        if n_shots in self.prompt_types:
            return self.prompt_types[n_shots]
        else:
            raise ValueError(f"Unknown prompt type: {n_shots}")

    def _zero_shot(self, question, options, subject, level):
        intro = f"You are an expert in {subject} at the {level} level. Analyze the given multiple-choice question and provide the correct answer using this approach:"
        cot = f"1. Carefully read the question and options \n \
                   2. Identify core {subject} concepts and required knowledge \n \
                   3. Analyze each option for relevance, accuracy, and consistency \n \
                   4. Consider {subject}-specific context and factors \n \
                   5. Use elimination and comparative analysis \n \
                   6. Select the most accurate answer \n \
                   Maintain objectivity, consider {subject}-specific sensitivities, and base your decision on verifiable facts and sound logical reasoning within {subject} at the {level}."
        question = f"Question: {question}"
        options = []
        for i, option in enumerate(options):
            options+= f"{i} - {option}"
        options = "\n".join(options)
        outro = "Correct option number is: "
        return "\n".join([intro, cot, question, options, outro])

    