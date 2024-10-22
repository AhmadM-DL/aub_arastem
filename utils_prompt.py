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

    def _zero_shot(self, question, question_options, subject, level):
        intro = f"You are an expert in {subject} at the {level} level. Analyze the given multiple-choice question and provide the correct answer using this approach:"
        cot = f"""1. Carefully read the question and options \n2. Identify core {subject} concepts and required knowledge \n3. Analyze each option for relevance, accuracy, and consistency \n4. Consider {subject}-specific context and factors \n5. Use elimination and comparative analysis \n6. Select the most accurate answer \nMaintain objectivity, consider {subject}-specific sensitivities, and base your decision on verifiable facts and sound logical reasoning within {subject} at the {level}."""
        question = f"Question: {question}"
        options = []
        mapping = {0:"A", 1:"B", 2:"C", 3:"D"}
        for i, option in enumerate(question_options):
            options.append(f"{mapping[i]} - {option}")
        options = "Options:\n" + "\n".join(options)
        outro = "Correct option is: "
        return "\n".join([intro, cot, question, options, outro])

    