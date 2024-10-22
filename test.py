from utils_prompt import PromptFactory

if __name__ == '__main__':
    q = {'question': 'إختر تعريف مصطلح الكيمياء التالي: الأكسدة',
        'option_0': 'فرق الجهد بين قطب المادة وقطب الهيدروجين القياسي',
        'option_1': 'عملية يتم فيها فقد إلكترونات، وينتج عنها زيادة عدد التأكسد للعنصر',
        'option_2': 'هي الأيونات الموجبة والسالبة للمركبات الأيونية الموجودة داخل الخلية على هيئة مصاهير أو محاليل إلكتروليتية',
        'option_3': 'هو المادة التي تختزل والتي تحتوي على عنصر ينقص عدد تأكسده أثناء التفاعل الكيميائي',
        'correct_option': 1,
        'level': 'collage',
        'subject': 'chemistry',
        'resource': 'book(https://platform.almanhal.com/Details/Book/53455)'
        }
    prompt_factory = PromptFactory()
    prompt_generator = prompt_factory.get_prompt_function(n_shots=0)

    question = q["question"]
    options = [q[f"option_{i}"] for i in [0, 1, 2, 3] if f"option_{i}" in q and q[f"option_{i}"] != "" ]
    prompt = prompt_generator(question, options, subject=q["subject"], level=q["level"])

    print(prompt)