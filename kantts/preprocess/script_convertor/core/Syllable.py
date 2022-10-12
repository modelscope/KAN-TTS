import xml.etree.ElementTree as ET

from .XmlObj import XmlObj


class Syllable(XmlObj):
    def __init__(self):
        self.m_phone_list = []
        self.m_tone = None
        self.m_language = None
        self.m_breaklevel = None

    def PronunciationText(self):
        return " ".join([str(phone) for phone in self.m_phone_list])

    def PhoneCount(self):
        return len(self.m_phone_list)

    def ToneText(self):
        return str(self.m_tone.value)

    def Save(self):
        pass

    def Load(self):
        pass

    def GetPhoneMeta(
        self, phone_name, word_pos, syll_pos, tone_text, single_syllable_word=False
    ):
        #  Special case: word with single syllable, the last phone's word_pos should be "word_end"
        if word_pos == "word_begin" and syll_pos == "s_end" and single_syllable_word:
            word_pos = "word_end"
        elif word_pos == "word_begin" and syll_pos not in [
            "s_begin",
            "s_both",
        ]:  # FIXME: keep accord with Engine logic
            word_pos = "word_middle"
        elif word_pos == "word_end" and syll_pos not in ["s_end", "s_both"]:
            word_pos = "word_middle"
        else:
            pass

        return "{{{}$tone{}${}${}}}".format(phone_name, tone_text, syll_pos, word_pos)

    def SaveMetafile(self, word_pos, single_syllable_word=False):
        syllable_phone_cnt = len(self.m_phone_list)

        meta_line_list = []

        for idx, phone in enumerate(self.m_phone_list):
            if syllable_phone_cnt == 1:
                syll_pos = "s_both"
            elif idx == 0:
                syll_pos = "s_begin"
            elif idx == len(self.m_phone_list) - 1:
                syll_pos = "s_end"
            else:
                syll_pos = "s_middle"
            meta_line_list.append(
                self.GetPhoneMeta(
                    phone,
                    word_pos,
                    syll_pos,
                    self.ToneText(),
                    single_syllable_word=single_syllable_word,
                )
            )

        return " ".join(meta_line_list)


class SyllableList(XmlObj):
    def __init__(self, syllables):
        self.m_syllable_list = syllables

    def __len__(self):
        return len(self.m_syllable_list)

    def __index__(self, index):
        return self.m_syllable_list[index]

    def PronunciationText(self):
        return " - ".join(
            [syllable.PronunciationText() for syllable in self.m_syllable_list]
        )

    def ToneText(self):
        return "".join([syllable.ToneText() for syllable in self.m_syllable_list])

    def Save(self, parent_node):
        syllable_node = ET.SubElement(parent_node, "syllable")
        syllable_node.set("syllcount", str(len(self.m_syllable_list)))

        phone_node = ET.SubElement(syllable_node, "phone")
        phone_node.text = self.PronunciationText()

        tone_node = ET.SubElement(syllable_node, "tone")
        tone_node.text = self.ToneText()

        return

    def Load(self):
        pass
