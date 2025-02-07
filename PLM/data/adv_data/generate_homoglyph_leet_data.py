import json
import numpy as np
from tqdm import tqdm
import datasets
import shap,torch,csv
import scipy as sp
from datasets.dataset_dict import DatasetDict
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

metric = evaluate.load("accuracy")
seed = 98

import itertools
import random
from typing import List



"""
---homoglyphs---
Visual letter perturbation based on image-based character embedding space https://arxiv.org/pdf/1903.11508v1.pdf.

"""

letter_mappings = {
    "a": [
        "а",
        "ạ",
        "ȧ",
        "ḁ",
        "ā",
        "ą",
        "ä",
        "ӓ",
        "ã",
        "á",
        "ẚ",
        "â",
        "à",
        "ả",
        "ậ",
        "ǎ",
        "ă",
        "ӑ",
        "ȃ",
        "ắ",
        "ặ",
        "ằ",
        "ȁ",
        "ɑ",
        "ầ",
        "α",
        "⍺",
        "ẳ",
        "å",
    ],
    "b": ["ḅ", "ḇ", "þ", "ϸ", "ƀ", "ƅ", "ɓ", "ƃ", "ƥ", "ᖯ", "ᑲ", "ᑿ", "Ь"],
    "c": [
        "ᴄ",
        "с",
        "ⅽ",
        "ϲ",
        "ċ",
        "ç",
        "ҫ",
        "ć",
        "ς",
        "ҁ",
        "ĉ",
        "ƈ",
        "ϛ",
        "ḉ",
    ],
    "d": [
        "ⅾ",
        "ḍ",
        "ḏ",
        "ḓ",
        "ḑ",
        "ɖ",
        "đ",
        "ժ",
        "₫",
        "ƌ",
        "ď",
        "ᴅ",
        "ɗ",
        "ᑯ",
        "ð",
        "ᕷ",
        "ɒ",
        "ȡ",
        "ᴆ",
        "ᑻ",
    ],
    "e": [
        "е",
        "ẹ",
        "ė",
        "ȩ",
        "є",
        "ē",
        "ę",
        "ḛ",
        "ë",
        "ё",
        "ǝ",
        "ə",
        "ә",
        "ẽ",
        "ɘ",
        "è",
        "ḙ",
        "ɞ",
        "ɐ",
        "ѐ",
        "ê",
        "é",
        "ẻ",
        "ͼ",
        "ε",
        "ԑ",
        "ɛ",
        "ệ",
        "ě",
        "ĕ",
        "ӗ",
        "ȇ",
        "ȅ",
        "ề",
        "ℯ",
        "ḝ",
        "ἐ",
        "ἑ",
        "ᴇ",
        "ḕ",
    ],
    "f": ["ƒ", "ẜ", "ḟ", "ẝ", "ʄ", "ϝ"],
    "g": ["ց", "ɡ", "ģ", "ġ", "ḡ", "ǵ", "ĝ", "ǥ", "ǧ", "ğ", "ɠ"],
    "h": [
        "հ",
        "һ",
        "ḥ",
        "ẖ",
        "ḫ",
        "ի",
        "ɦ",
        "ḩ",
        "ɧ",
        "ћ",
        "ħ",
        "Һ",
        "ℎ",
        "ⱨ",
    ],
    "i": [
        "і",
        "ⅰ",
        "ị",
        "į",
        "ı",
        "ו",
        "إ",
        "ߊ",
        "ἰ",
        "ľ",
        "ⵏ",
        "ἱ",
        "⍳",
        "ι",
        "ɩ",
        "ⵑ",
        "ɨ",
        "í",
        "⌊",
        "ḭ",
        "ᶅ",
        "ǃ",
        "ί",
        "ί",
        "î",
        "ĩ",
        "׀",
        "ℹ",
    ],
    "j": [
        "ј",
        "ϳ",
        "ȷ",
        "ɉ",
        "ĵ",
        "ǰ",
        "ز",
        "ڗ",
        "ۈ",
        "ژ",
        "ڙ",
        "ڒ",
        "ڑ",
        "ⅉ",
    ],
    "k": [
        "ķ",
        "ḳ",
        "ḵ",
        "ƙ",
        "ᴋ",
        "ĸ",
        "ҟ",
        "κ",
        "ⱪ",
        "к",
        "ҝ",
        "ԟ",
        "ќ",
        "ḱ",
    ],
    "l": [
        "ⅼ",
        "ا",
        "ļ",
        "ḷ",
        "ŀ",
        "ḽ",
        "ł",
        "ⵏ",
        "ɭ",
        "ᶅ",
        "Ɩ",
        "ȴ",
        "ⱡ",
        "լ",
    ],
    "m": ["ⅿ", "ṃ", "ṁ", "ḿ", "ⴇ", "ⴊ", "ⴔ", "ⴜ"],
    "n": [
        "ո",
        "ņ",
        "ṇ",
        "ṉ",
        "ח",
        "ṅ",
        "ṋ",
        "η",
        "ƞ",
        "ŋ",
        "ɳ",
        "п",
        "ῃ",
        "ה",
        "ɲ",
        "ñ",
        "ռ",
        "ǹ",
        "һ",
        "ń",
        "ἠ",
        "ἡ",
        "ᾐ",
        "ᾑ",
        "ῆ",
        "ᴨ",
        "ⴖ",
        "ὴ",
        "ῇ",
        "ň",
        "ῂ",
        "ή",
        "ή",
        "ɴ",
        "ἢ",
        "ἣ",
        "ῄ",
        "ᴎ",
        "ͷ",
        "и",
        "ᾒ",
        "ᾓ",
        "ἤ",
    ],
    "o": [
        "ο",
        "օ",
        "ᴏ",
        "о",
        "ọ",
        "ȯ",
        "ơ",
        "σ",
        "ǫ",
        "ợ",
        "ὀ",
        "ὁ",
        "ō",
        "ø",
        "ϙ",
        "ӧ",
        "ö",
        "ɵ",
        "ѳ",
        "ө",
        "õ",
        "ʊ",
        "ວ",
        "ǭ",
        "ό",
        "ό",
        "ó",
        "ô",
        "ὸ",
        "ò",
        "ỏ",
        "ớ",
        "ὂ",
        "ὃ",
        "໐",
        "ס",
        "ộ",
        "ǿ",
        "ờ",
        "ở",
        "ǒ",
        "⌀",
        "ອ",
        "ŏ",
    ],
    "p": [
        "р",
        "⍴",
        "ρ",
        "ṗ",
        "ϸ",
        "þ",
        "ῤ",
        "ῥ",
        "ṕ",
        "ƥ",
        "ƿ",
        "ҏ",
        "բ",
        "ᵽ",
    ],
    "q": ["ԛ", "ɋ", "ɖ", "ϥ", "զ"],
    "r": [
        "ŗ",
        "ṛ",
        "ṟ",
        "ṙ",
        "г",
        "ᴦ",
        "ṝ",
        "ɽ",
        "ɼ",
        "ґ",
        "ŕ",
        "Ւ",
        "ѓ",
        "ř",
        "ȓ",
        "Ի",
    ],
    "s": ["ѕ", "ș", "ṣ", "ṡ", "ṩ", "ʂ", "ş", "ś", "ŝ", "š", "ṥ"],
    "t": ["ț", "ṭ", "ṯ", "ṱ", "ţ", "ƭ", "ƫ", "ŧ", "ť", "ʈ", "է", "Է", "ե"],
    "u": [
        "ս",
        "ṳ",
        "ụ",
        "ū",
        "ư",
        "ṵ",
        "ü",
        "ự",
        "ʋ",
        "ų",
        "ũ",
        "ṷ",
        "ú",
        "ⴎ",
        "μ",
        "µ",
        "û",
        "ù",
        "υ",
        "ủ",
        "ບ",
        "ᴜ",
        "ʊ",
        "ữ",
        "կ",
        "վ",
        "մ",
        "ứ",
        "џ",
        "ǔ",
        "ů",
        "ŭ",
        "ừ",
        "ߎ",
        "ử",
        "ὑ",
        "ȗ",
        "ὐ",
        "ט",
        "ȕ",
        "ῡ",
        "ϋ",
        "և",
        "ű",
    ],
    "v": ["ⅴ", "ᴠ", "ṿ", "ѵ", "ү", "ν", "ṽ", "ⱱ", "ⱴ", "γ", "ѷ", "ע"],
    "w": ["ᴡ", "ԝ", "ẉ", "ẇ", "ẅ", "ẁ", "ŵ", "ẃ", "ẘ", "ⱳ", "ա", "ⴍ"],
    "x": ["ⅹ", "х", "ẋ", "ӿ", "ẍ", "ҳ", "ӽ"],
    "y": [
        "у",
        "ỵ",
        "ү",
        "ẏ",
        "ȳ",
        "ӯ",
        "ÿ",
        "ӱ",
        "ỹ",
        "ƴ",
        "ў",
        "ý",
        "ŷ",
        "ỷ",
        "ỳ",
        "ӳ",
        "ẙ",
    ],
    "z": ["ᴢ", "ẓ", "ẕ", "ż", "ƶ", "ȥ", "ʐ", "ź", "ẑ", "ʑ", "ƨ", "ž"],
    "A": [
        "Α",
        "А",
        "Ạ",
        "ᾼ",
        "Ḁ",
        "Ἀ",
        "Ἁ",
        "ᾈ",
        "ᾉ",
        "Ȧ",
        "Ά",
        "Ά",
        "Ⱥ",
        "Ą",
        "Ả",
        "ᕕ",
        "À",
        "Ẵ",
        "Ằ",
        "Ắ",
        "Ᾰ",
        "Á",
        "Ȃ",
        "Ã",
        "Ӓ",
        "Ä",
        "Ẳ",
        "₳",
        "Ȁ",
        "Ẫ",
        "Â",
        "ᕖ",
        "Ậ",
        "Ᾱ",
        "Ā",
        "Ǎ",
        "Å",
        "Å",
        "4",
        "Ӑ",
        "Ă",
        "Ặ",
        "Ầ",
        "Ấ",
        "Ǡ",
        "Ǟ",
        "Ẩ",
        "Ὰ",
    ],
    "B": ["Β", "В", "Ḅ", "Ḇ", "Ḃ", "8", "Ƀ", "ß", "β", "฿"],
    "C": [
        "Ⅽ",
        "Ϲ",
        "С",
        "Ҫ",
        "Ç",
        "Ҁ",
        "ᑕ",
        "Ċ",
        "ᕦ",
        "ᑢ",
        "Ϛ",
        "ⵛ",
        "ⵎ",
        "Ȼ",
        "ᑖ",
        "Ć",
        "ᕧ",
        "ᑤ",
        "₵",
        "ᕩ",
    ],
    "D": ["Ⅾ", "ᗞ", "Ḏ", "Ḍ", "Ḓ", "Ḑ", "ↁ", "Ḋ", "Ð", "Ɖ", "Đ", "Ď", "Ɒ"],
    "E": [
        "Ε",
        "ⴹ",
        "Е",
        "Ẹ",
        "ⵟ",
        "Ȩ",
        "Ḛ",
        "Ę",
        "Ḙ",
        "Ė",
        "Ɛ",
        "Ԑ",
        "ℇ",
        "Є",
        "Ẻ",
        "È",
        "Ӗ",
        "Ĕ",
        "Ѐ",
        "É",
        "Ȇ",
        "Ƹ",
        "Ḝ",
        "Ẽ",
        "⋿",
        "Ё",
        "Ë",
        "∃",
        "ⴺ",
        "Ǝ",
        "£",
        "Ȅ",
        "Ễ",
        "Ê",
        "Ệ",
        "Σ",
        "ⵉ",
    ],
    "F": ["Ϝ", "Ḟ", "ߓ", "Ғ"],
    "G": ["Ģ", "Ǥ", "Ԍ", "Ġ", "Ğ", "Ǵ", "Ḡ", "Ĝ"],
    "H": [
        "Η",
        "Н",
        "ᕼ",
        "Ḥ",
        "Ḫ",
        "ῌ",
        "Ḣ",
        "Ḩ",
        "Ӈ",
        "Ң",
        "Ⱨ",
        "Ḧ",
        "Ĥ",
        "Ȟ",
        "Ӊ",
        "ℍ",
    ],
    "I": [
        "Ι",
        "Ӏ",
        "ⵏ",
        "І",
        "Ⅰ",
        "Ị",
        "ߊ",
        "ⵑ",
        "ا",
        "ǃ",
        "Į",
        "ⅼ",
        "ӏ",
        "إ",
        "׀",
        "Ḭ",
    ],
    "J": ["Ј", "Ɉ", "յ", "Ĵ", "ȷ"],
    "K": [
        "Κ",
        "K",
        "Ķ",
        "Ḳ",
        "Ḵ",
        "₭",
        "Ƙ",
        "Ҝ",
        "Ḱ",
        "К",
        "Ԟ",
        "Ҟ",
        "Ϗ",
        "қ",
        "Ǩ",
        "ⱪ",
        "ҟ",
        "Ⱪ",
        "ԟ",
        "ҝ",
        "κ",
        "к",
        "Ќ",
        "ⴿ",
    ],
    "L": ["ᒪ", "Ⅼ", "Ļ", "Լ", "Ḷ", "Ḻ", "Ŀ", "ᒷ", "Ľ", "Ḽ", "ʟ", "ւ"],
    "M": ["Ⅿ", "Μ", "М", "Ṃ", "Ṁ", "Ϻ", "Ḿ", "Ɱ", "Ӎ", "ᴍ", "м"],
    "N": ["Ν", "Ņ", "Ṉ", "Ṇ", "Ṋ", "Ṅ", "Ǹ", "Ń", "Ñ", "Ň", "Ɲ", "Ŋ"],
    "O": [
        "Ο",
        "Օ",
        "О",
        "Ọ",
        "Ǫ",
        "Ơ",
        "Θ",
        "Ϙ",
        "Ợ",
        "Ɵ",
        "ϴ",
        "Ө",
        "Ѳ",
        "Ȯ",
        "Q",
        "Ԛ",
        "Ό",
        "Ό",
        "Ὀ",
        "Ò",
        "Ŏ",
        "⊙",
        "Ờ",
        "Ȏ",
        "Ó",
        "Ỏ",
        "Ȍ",
        "⊝",
        "Õ",
        "Ṍ",
        "Ṏ",
        "Ӧ",
        "Ö",
        "⊘",
        "⊖",
        "Ớ",
        "Ở",
        "⊜",
        "⊛",
        "Ỡ",
        "⊚",
        "ʘ",
        "Ǿ",
        "Ô",
        "Ỗ",
        "Ộ",
        "Ӫ",
        "⊗",
        "⊕",
        "Փ",
        "Ṑ",
        "Ō",
        "Ṓ",
        "Ǒ",
        "Ǭ",
        "Ὁ",
        "Ø",
        "Ѻ",
    ],
    "P": [
        "Ρ",
        "Р",
        "Ҏ",
        "ᑭ",
        "ᕈ",
        "Ṗ",
        "ᑷ",
        "₱",
        "Ᵽ",
        "ᑮ",
        "ᑹ",
        "Ṕ",
        "ᕉ",
        "ᒆ",
    ],
    "Q": ["Ԛ", "ℚ"],
    "R": ["Ŗ", "Ṛ", "Ṟ", "Ɽ", "Ɍ", "Ṙ", "Ŕ", "Ȓ", "Ṝ", "Ȑ", "℞", "Ř", "Я"],
    "S": [
        "Ѕ",
        "Ș",
        "Ṣ",
        "Ş",
        "Ṡ",
        "Ṩ",
        "5",
        "Ś",
        "Ƽ",
        "Ŝ",
        "Ṥ",
        "Ṧ",
        "Š",
        "Ƨ",
    ],
    "T": [
        "Т",
        "Τ",
        "Ț",
        "Ṭ",
        "Ṯ",
        "Ṱ",
        "Ţ",
        "Ŧ",
        "Ʈ",
        "Ṫ",
        "Ҭ",
        "₮",
        "Ԏ",
        "Ƭ",
        "₸",
        "Ť",
        "ⴶ",
        "ͳ",
        "ߠ",
        "☨",
        "⊺",
        "☦",
    ],
    "U": [
        "Ս",
        "ᑌ",
        "Ṳ",
        "Ụ",
        "Ų",
        "Ṵ",
        "ᑘ",
        "Ṷ",
        "ⵡ",
        "Ư",
        "Ự",
        "Џ",
        "Ա",
        "Ù",
        "Ʉ",
        "ᕟ",
        "ᕞ",
        "Ŭ",
        "Ȗ",
        "Ú",
        "Ủ",
        "Ȕ",
        "Ů",
        "Ṹ",
        "Ũ",
        "ᑧ",
        "Ü",
        "Ǘ",
        "Ǜ",
        "Ǚ",
        "Ừ",
        "Ц",
        "Ứ",
        "Ử",
        "Û",
        "∪",
        "Ū",
        "Ǔ",
        "Մ",
        "⊎",
        "⊍",
        "⊌",
    ],
    "V": ["Ⅴ", "ᐯ", "ⴸ", "Ṿ", "ᐻ", "Ѵ", "∨", "⊽", "Ѷ", "⋎", "℣", "Ṽ"],
    "W": ["Ԝ", "Ẉ", "Ẇ", "Ẁ", "Ⱳ", "₩", "Ẅ", "Ẃ", "Ŵ", "Ɯ"],
    "X": [
        "Х",
        "Χ",
        "ⵝ",
        "Ⅹ",
        "Ӿ",
        "ⵅ",
        "ⴴ",
        "Ẋ",
        "ⴳ",
        "Ẍ",
        "Ӽ",
        "Ҳ",
        "✘",
        "☓",
        "ⴵ",
    ],
    "Y": [
        "Ү",
        "Υ",
        "Ỵ",
        "Ẏ",
        "Ƴ",
        "Ỷ",
        "Ý",
        "Ῠ",
        "Ỳ",
        "Ɏ",
        "Ỹ",
        "Ÿ",
        "Ϋ",
        "Ŷ",
        "Ȳ",
        "Ῡ",
        "ϒ",
    ],
    "Z": ["Ζ", "Ẓ", "Ẕ", "Ƶ", "Ż", "Ȥ", "Ź", "ƻ", "Ẑ", "Ž"],
}

# Now, let's generate a dictionary that maps every letter to every other letter in its family.
# That way we aren't stuck with only modifying the standard letters in the Latin alphabet
# But instead we can also visually attack nonstandard letters such as 'Ṃ' or 'ǵ' or 'ℚ'

complete_mappings = {}
for key_letter, letter_family in letter_mappings.items():
    for letter in letter_family:
        complete_mappings[letter] = list(
            set(letter_family) - {letter} | {key_letter}
        )

    # Add back in the original mapping
    complete_mappings[key_letter] = letter_family

letter_mappings = complete_mappings
class VisualAttackLetters():
    # tasks = [
    #     TaskType.TEXT_CLASSIFICATION,
    #     TaskType.TEXT_TO_TEXT_GENERATION,
    #     TaskType.TEXT_TAGGING,
    # ]
    languages = [
        "sq",
        "hy",
        "bg",
        "ca",
        "hr",
        "cs",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "is",
        "it",
        "la",
        "lv",
        "lt",
        "mk",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "es",
        "sv",
        "tr",
        "uk",
    ]
    # keywords = [
    #     "morphological",
    #     "external-knowledge-based",
    #     "visual",
    #     "high-generations",
    # ]

    def __init__(
        self, seed: int = 0, max_outputs: int = 1, perturb_pct: float = 0.50
    ) -> None:
        """
        In order to generate multiple different perturbations, you should set seed=None
        """
        # super().__init__(seed=seed, max_outputs=max_outputs)
        self.perturb_pct = perturb_pct
        self.seed=seed
        self.max_outputs=max_outputs

    def homo_50(self, sentence: str) -> List[str]:
        random.seed(self.seed)
        sentence_list = list(sentence)
        l=len(sentence_list)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # print("sen list", sentence_list, l,k_fifty)
        perturbed_texts = []
        # Perturb the input sentence max_output times
        
        cnt=1
        for idx, letter in enumerate(sentence_list):
           
            if(cnt>k_fifty):
                
                break
                
            else:
                if (
                    letter in letter_mappings
                    and random.random() < self.perturb_pct
                ):
                    sentence_list[idx] = random.choice(
                        letter_mappings[letter]
                    )  
                cnt=cnt+1

        perturbed_texts="".join(sentence_list)

        return perturbed_texts

    def homo_25(self, sentence: str) -> List[str]:
        random.seed(self.seed)
        sentence_list = list(sentence)
        l=len(sentence_list)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # print("sen list", sentence_list, l,k_fifty)
        perturbed_texts = []
        # Perturb the input sentence max_output times
        
        cnt=1
        for idx, letter in enumerate(sentence_list):
           
            if(cnt>k_twntyfive):
                
                break
                
            else:
                if (
                    letter in letter_mappings
                    and random.random() < self.perturb_pct
                ):
                    sentence_list[idx] = random.choice(
                        letter_mappings[letter]
                    )  
                cnt=cnt+1

        perturbed_texts="".join(sentence_list)

        return perturbed_texts
    





"""
https://github.com/GEM-benchmark/NL-Augmenter
Leet speak letter perturbation based on https://simple.wikipedia.org/wiki/Leet, excluding the space > 0.
"""

leet_letter_mappings = {
    "!": "1",
    "7": "1",
    "A": "4",
    "B": "8",
    "C": "0",
    "D": "0",
    "E": "3",
    "G": "6",
    "I": "1",
    "J": "9",
    "L": "7",
    "N": "11",
    "O": "0",
    "S": "5",
    "T": "7",
    "X": "8",
    "Z": "2",
    "i": "1",
    "b": "6",
    "e": "3",
    "g": "9",
    "h": "4",
    "j": "7",
    "m": "3",
    "o": "0",
    "w": "3",
    "y": "4",
    "|": "1",
    "Θ": "0",
    "ε": "3",
    "ω": "3",
    "∈": "3",
    "∩∩": "3",
}


class LeetLetters():
    # tasks = [
    #     TaskType.TEXT_CLASSIFICATION,
    #     TaskType.TEXT_TO_TEXT_GENERATION,
    #     TaskType.TEXT_TAGGING,
    # ]
    # languages = ["en"]

    def __init__(
        self, seed: int = 0, max_outputs: int = 1, max_leet: float = 0.5
    ) -> None:
        # super().__init__(seed=seed, max_outputs=max_outputs)
        self.max_leet = max_leet
        self.max_outputs=max_outputs
       
            
    def leet_50(self, sentence: str) -> List[str]:
        # random.seed(self.seed)
        max_leet_replacements = int(self.max_leet * len(sentence))
        perturbed_texts = []
       
        l=len(sentence)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # Perturb the input sentence max_output times
        # for _ in itertools.repeat(None, self.max_outputs):
            # Determine what to replace
        leet_candidates = [];cnt=1;sentence_list = list(sentence);cnt=1
        for idx, letter in enumerate(sentence):
            if(cnt>k_fifty):
                break
            else:
                # print(idx, letter,cnt)
                if letter in leet_letter_mappings:
                    sentence_list[idx] = str(leet_letter_mappings[letter]);cnt=cnt+1
                    # print("re",leet_letter_mappings[letter])
                    # leet_candidates.append((idx, leet_letter_mappings[letter]))
            # leet_replacements = random.choices(
            #     leet_candidates, k=max_leet_replacements
            # )
            # Conduct replacement
            # sentence_list = list(sentence)
            # for idx, leet in leet_replacements:
            #     sentence_list[idx] = str(leet)
            
        perturbed_texts="".join(sentence_list)
        return perturbed_texts
        
    def leet_25(self, sentence: str) -> List[str]:
        # random.seed(self.seed)
        max_leet_replacements = int(self.max_leet * len(sentence))
        perturbed_texts = []
       
        l=len(sentence)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # Perturb the input sentence max_output times
        # for _ in itertools.repeat(None, self.max_outputs):
            # Determine what to replace
        leet_candidates = [];cnt=1;sentence_list = list(sentence);cnt=1
        for idx, letter in enumerate(sentence):
            if(cnt>k_twntyfive):
                break
            else:
                # print(idx, letter,cnt)
                if letter in leet_letter_mappings:
                    sentence_list[idx] = str(leet_letter_mappings[letter]);cnt=cnt+1
                    
            
        perturbed_texts="".join(sentence_list)
        

        return perturbed_texts
    
def read_data(path, to_dataset=True):
    '''
    read csv and return list of instances
    '''
    rows = []; label_dict = {'SUPPORTED': 0, 'REFUTED': 1, 'NEI': 2}
    # read data
    with open(path, 'r') as f:
        j_list = list(f)

    for j_str in j_list:
        row_dict = json.loads(j_str)
        
        rows.append({
            'id': row_dict['id'],
            'text': row_dict['gold_evidence_text'].strip() + '[SEP]' + row_dict['claim'].strip(),
            'label': label_dict[row_dict['label']],
            'claim': row_dict['claim'].strip(),
            'evidence':  row_dict['gold_evidence_text'].strip()

        })
    if to_dataset:
        # to object
        rows = datasets.Dataset.from_list(rows)

    return rows



      
if __name__ == "__main__":
    # dataset
    rows = read_data("./fever_test.jsonl", to_dataset=False)

   
    functionss2=['leet_25','leet_50','homo_25','homo_50']
    # functionss2=['homo_50']

    ll=LeetLetters()
    
    va=VisualAttackLetters()

  
   
    prev_pred = []; gold = [];leet_adv_pred=[];homo_adv_pred=[]
    for func in functionss2:
        aa=func
        with open(aa+"_adv_train.csv", "w", newline='') as f:
            # Create a CSV writer object
            writer = csv.writer(f)	    
            # Write the header row
            writer.writerow(["id", "actual_claim", "gold_evidence_text", "adv_claim", "label"])
        pass
    stop=0
    for row in tqdm(rows, total=len(rows)):
        stop=stop+1
        print(stop)
        act=row['claim'];homo_claim=row['claim'];leet_claim=row['claim']

        for func in functionss2:
            aa=func
            if("leet" in func):
                c=ll
            elif("homo" in func):
                c=va
            
            
            aa=func
           
            func = getattr(c, func, None)
            
            pert=1; budget1 = 0.50; perturb_ratio=0
            homo_claim=va.homo_50(act) 
            
           
            with open(aa+"_adv_train.csv", "a", newline='') as f:
                # Create a CSV writer object
                writer = csv.writer(f) 
                # Write the header row
                writer.writerow([row['id'], act, row['evidence'], homo_claim, row['label']])
            pass
           

       
