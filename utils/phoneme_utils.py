import re
import json

class Korean:
    unicode_base_code, unicode_onset_offset, unicode_nucleus_offset = 44032, 588, 28
    phoneme_onset_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    phoneme_nucleus_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    phoneme_coda_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    phoneme_double_consonant_dict = {
        'ㄲ': ['ㄱ', 'ㄱ'], 'ㄳ': ['ㄱ', 'ㅅ'], 'ㄵ': ['ㄴ', 'ㅈ'], 'ㄶ': ['ㄴ', 'ㅎ'],
        'ㄸ': ['ㄷ', 'ㄷ'], 'ㄺ': ['ㄹ', 'ㄱ'], 'ㄻ': ['ㄹ', 'ㅁ'], 'ㄼ': ['ㄹ', 'ㅂ'],
        'ㄽ': ['ㄹ', 'ㅅ'], 'ㄾ': ['ㄹ', 'ㅌ'], 'ㄿ': ['ㄹ', 'ㅍ'], 'ㅀ': ['ㄹ', 'ㅎ'],
        'ㅃ': ['ㅂ', 'ㅂ'], 'ㅄ': ['ㅂ', 'ㅅ'], 'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ']
    }

    @staticmethod
    def is_korean(char):
        code = ord(char)
        return 0xAC00 <= code <= 0xD7A3

    @staticmethod
    def decompose_syllable(char):
        base = ord(char) - Korean.unicode_base_code
        onset = base // Korean.unicode_onset_offset
        nucleus = (base % Korean.unicode_onset_offset) // Korean.unicode_nucleus_offset
        coda = base % Korean.unicode_nucleus_offset
        o = Korean.phoneme_onset_list[onset]
        n = Korean.phoneme_nucleus_list[nucleus]
        c = Korean.phoneme_coda_list[coda] if coda > 0 else None
        return o, n, c

    @staticmethod
    def text_to_phoneme_sequence(text, phoneme2index):
        seq = []
        for char in text:
            if not Korean.is_korean(char): continue
            o, n, c = Korean.decompose_syllable(char)
            if o in Korean.phoneme_double_consonant_dict:
                for p in Korean.phoneme_double_consonant_dict[o]:
                    seq.append(phoneme2index.get(p, 0))
            elif o:
                seq.append(phoneme2index.get(o, 0))
            if n: seq.append(phoneme2index.get(n, 0))
            if c in Korean.phoneme_double_consonant_dict:
                for p in Korean.phoneme_double_consonant_dict[c]:
                    seq.append(phoneme2index.get(p, 0))
            elif c:
                seq.append(phoneme2index.get(c, 0))
        return seq

def clean_korean_text(text: str) -> str:
    return re.sub(r"[^가-힣]", "", text)

def convert_prompt_to_phoneme_sequence(prompt: str, phoneme2index: dict, korean: Korean) -> list:
    cleaned = clean_korean_text(prompt)

    if not cleaned:
        print(f"[SKIP] Prompt cleaned to empty: '{prompt}'")
        return []

    phoneme_seq = korean.text_to_phoneme_sequence(cleaned, phoneme2index)

    if not phoneme_seq:
        print(f"[SKIP] Phoneme sequence empty after mapping: '{cleaned}' ← '{prompt}'")
        return []

    return phoneme_seq