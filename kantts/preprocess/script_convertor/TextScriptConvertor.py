import logging
import os
from tqdm import tqdm
from .core.Script import Script
from .core.ScriptItem import ScriptItem
from .core.ScriptSentence import ScriptSentence
from .core.SyllableFormatter import (
    ZhCNSyllableFormatter,
    ZhHKSyllableFormatter,
    PinYinSyllableFormatter,
    WuuShanghaiSyllableFormatter,
    SichuanSyllableFormatter,
    EnXXSyllableFormatter,
)
from .core.ScriptWord import SpokenWord, SpokenMark, WrittenWord, WrittenMark
from .core.PhoneSet import PhoneSet
from .core.PosSet import PosSet
from .core.core_types import Language, BreakLevel
from .core.utils import (
    RegexID,
    RegexSentence,
    RegexForeignLang,
    RegexNeutralTone,
    format_prosody,
)
import argparse
import re
from bitstring import BitArray

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)


class TextScriptConvertor:
    def __init__(
        self,
        phoneset_path,
        posset_path,
        target_lang,
        foreign_lang,
        f2t_map_path,
        s2p_map_path,
        m_emo_tag_path,
        m_speaker,
    ):
        self.m_f2p_map = {}
        self.m_s2p_map = {}
        self.m_phoneset = PhoneSet(phoneset_path)
        self.m_posset = PosSet(posset_path)
        self.m_target_lang = Language.parse(target_lang)
        self.m_foreign_lang = Language.parse(foreign_lang)
        self.m_emo_tag_path = m_emo_tag_path
        self.m_speaker = m_speaker

        self.LoadF2TMap(f2t_map_path)
        self.LoadS2PMap(s2p_map_path)

        if m_emo_tag_path is not None:
            self.m_emo_dict = self.parse_emo_tag(m_emo_tag_path)
        else:
            self.m_emo_dict = {}

        self.m_target_lang_syllable_formatter = self.InitSyllableFormatter(
            self.m_target_lang
        )
        self.m_foreign_lang_syllable_formatter = self.InitSyllableFormatter(
            self.m_foreign_lang
        )

    def parse_emo_tag(self, emo_tag_path):
        with open(emo_tag_path, "r") as f:
            lines = f.readlines()
            emo_tag = {}
            for line in lines:
                line = line.strip()
                elements = line.split()
                if len(elements) != 2:
                    logging.error(
                        "TextScriptConvertor.parse_emo_tag: invalid line: %s", line
                    )
                    continue
                key = elements[0]
                value = elements[1]
                if key in emo_tag:
                    logging.warning(
                        "TextScriptConvertor.parse_emo_tag: duplicate key: %s", key
                    )
                emo_tag[key] = value
        return emo_tag

    def parse_sentence(self, sentence, line_num):
        script_item = ScriptItem(self.m_phoneset, self.m_posset)
        script_sentence = ScriptSentence(self.m_phoneset, self.m_posset)
        script_item.m_scriptSentence_list.append(script_sentence)

        written_sentence = script_sentence.m_writtenSentence
        spoken_sentence = script_sentence.m_spokenSentence

        position = 0

        sentence = sentence.strip()

        #  Get ID
        match = re.search(RegexID, sentence)
        if match is None:
            logging.error(
                "TextScriptConvertor.parse_sentence:invalid line: %s,\
                    line ID is needed",
                line_num,
            )
            return None
        else:
            sentence_id = match.group("ID")
            script_item.m_id = sentence_id
            position += match.end()

        prevSpokenWord = SpokenWord()

        prevWord = False
        lastBreak = False

        for m in re.finditer(RegexSentence, sentence[position:]):
            if m is None:
                logging.error(
                    "TextScriptConvertor.parse_sentence:\
                    invalid line: %s, there is no matched pattern",
                    line_num,
                )
                return None

            if m.group("Word") is not None:
                wordName = m.group("Word")
                written_word = WrittenWord()
                written_word.m_name = wordName
                written_sentence.AddHost(written_word)

                spoken_word = SpokenWord()
                spoken_word.m_name = wordName
                prevSpokenWord = spoken_word
                prevWord = True
                lastBreak = False
            elif m.group("Break") is not None:
                breakText = m.group("BreakLevel")
                if len(breakText) == 0:
                    breakLevel = BreakLevel.L1
                else:
                    breakLevel = BreakLevel.parse(breakText)
                if prevWord:
                    prevSpokenWord.m_breakText = breakText
                    spoken_sentence.AddHost(prevSpokenWord)

                if breakLevel != BreakLevel.L1:
                    spokenMark = SpokenMark()
                    spokenMark.m_breakLevel = breakLevel
                    spoken_sentence.AddAccompany(spokenMark)

                lastBreak = True

            elif m.group("PhraseTone") is not None:
                #  TODO: PhraseTonePattern
                pass
            elif m.group("POS") is not None:
                POSClass = m.group("POSClass")
                if prevWord:
                    prevSpokenWord.m_pos = POSClass
                prevWord = False
            elif m.group("Mark") is not None:
                markText = m.group("Mark")

                writtenMark = WrittenMark()
                writtenMark.m_punctuation = markText
                written_sentence.AddAccompany(writtenMark)
            else:
                logging.error(
                    "TextScriptConvertor.parse_sentence:\
                invalid line: %s, matched pattern is unrecognized",
                    line_num,
                )
                return None

        if not lastBreak:
            prevSpokenWord.m_breakText = "4"
            spoken_sentence.AddHost(prevSpokenWord)

        spoken_word_cnt = len(spoken_sentence.m_spoken_word_list)
        spoken_mark_cnt = len(spoken_sentence.m_spoken_mark_list)
        if (
            spoken_word_cnt > 0
            and spoken_sentence.m_align_list[spoken_word_cnt - 1] == spoken_mark_cnt
        ):
            spokenMark = SpokenMark()
            spokenMark.m_breakLevel = BreakLevel.L4
            spoken_sentence.AddAccompany(spokenMark)

        written_sentence.BuildSequence()
        spoken_sentence.BuildSequence()
        written_sentence.BuildText()
        spoken_sentence.BuildText()

        script_sentence.m_text = written_sentence.m_text
        script_item.m_text = written_sentence.m_text

        return script_item

    def FormatSyllable(self, pron, syllable_list):
        isForeign = RegexForeignLang.search(pron) is not None
        if self.m_foreign_lang_syllable_formatter is not None and isForeign:
            return self.m_foreign_lang_syllable_formatter.Format(
                self.m_phoneset, pron, syllable_list
            )
        else:
            return self.m_target_lang_syllable_formatter.Format(
                self.m_phoneset, pron, syllable_list
            )

    def GetWordProns(self, pronText):
        prons = pronText.split("/")
        res = []

        for pron in prons:
            if re.search(RegexForeignLang, pron):
                res.append(pron.strip())
            else:
                res.extend(pron.strip().split(" "))
        return res

    def IsErHuaYin(self, pron):
        pron = RegexNeutralTone.sub("5", pron)
        pron = pron[:-1]

        return pron[-1] == "r" and pron != "er"

    def parse_pronunciation(self, script_item, pronunciation, line_num):
        spoken_sentence = script_item.m_scriptSentence_list[0].m_spokenSentence

        wordProns = self.GetWordProns(pronunciation)

        wordIndex = 0
        pronIndex = 0
        succeed = True

        while pronIndex < len(wordProns):
            language = Language.Neutral
            syllable_list = []

            pron = wordProns[pronIndex].strip()

            succeed = self.FormatSyllable(pron, syllable_list)
            if not succeed:
                logging.error(
                    "TextScriptConvertor.parse_pronunciation:\
                        invalid line: %s, error pronunciation: %s,\
                        syllable format error",
                    line_num,
                    pron,
                )
                return False
            language = syllable_list[0].m_language

            if wordIndex < len(spoken_sentence.m_spoken_word_list):
                if language in [Language.EnGB, Language.EnUS]:
                    spoken_sentence.m_spoken_word_list[
                        wordIndex
                    ].m_syllable_list.extend(syllable_list)
                    wordIndex += 1
                    pronIndex += 1
                elif language in [
                    Language.ZhCN,
                    Language.PinYin,
                    Language.ZhHK,
                    Language.WuuShanghai,
                    Language.Sichuan,
                ]:
                    charCount = len(
                        spoken_sentence.m_spoken_word_list[wordIndex].m_name
                    )
                    if (
                        language in [Language.ZhCN, Language.PinYin, Language.Sichuan]
                        and self.IsErHuaYin(pron)
                        and "儿" in spoken_sentence.m_spoken_word_list[wordIndex].m_name
                    ):
                        spoken_sentence.m_spoken_word_list[
                            wordIndex
                        ].m_name = spoken_sentence.m_spoken_word_list[
                            wordIndex
                        ].m_name.replace(
                            "儿", ""
                        )
                        charCount -= 1
                    if charCount == 1:
                        spoken_sentence.m_spoken_word_list[
                            wordIndex
                        ].m_syllable_list.extend(syllable_list)
                        wordIndex += 1
                        pronIndex += 1
                    else:
                        #  FIXME(Jin): Just skip the first char then match the rest char.
                        i = 1
                        while i >= 1 and i < charCount:
                            pronIndex += 1
                            if pronIndex < len(wordProns):
                                pron = wordProns[pronIndex].strip()
                                succeed = self.FormatSyllable(pron, syllable_list)
                                if not succeed:
                                    logging.error(
                                        "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                                                error pronunciation: %s, syllable format error",
                                        line_num,
                                        pron,
                                    )
                                    return False
                                if (
                                    language
                                    in [
                                        Language.ZhCN,
                                        Language.PinYin,
                                        Language.Sichuan,
                                    ]
                                    and self.IsErHuaYin(pron)
                                    and "儿"
                                    in spoken_sentence.m_spoken_word_list[
                                        wordIndex
                                    ].m_name
                                ):
                                    spoken_sentence.m_spoken_word_list[
                                        wordIndex
                                    ].m_name = spoken_sentence.m_spoken_word_list[
                                        wordIndex
                                    ].m_name.replace(
                                        "儿", ""
                                    )
                                    charCount -= 1
                            else:
                                logging.error(
                                    "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                                            error pronunciation: %s, Word count mismatch with Pron count",
                                    line_num,
                                    pron,
                                )
                                return False
                            i += 1
                        spoken_sentence.m_spoken_word_list[
                            wordIndex
                        ].m_syllable_list.extend(syllable_list)
                        wordIndex += 1
                        pronIndex += 1
                else:
                    logging.error(
                        "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                                unsupported language: %s",
                        line_num,
                        language.name,
                    )
                    return False

            else:
                logging.error(
                    "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                            error pronunciation: %s, word index is out of range",
                    line_num,
                    pron,
                )
                return False
        if pronIndex != len(wordProns):
            logging.error(
                "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                        error pronunciation: %s, pron count mismatch with word count",
                line_num,
                pron,
            )
            return False

        if wordIndex != len(spoken_sentence.m_spoken_word_list):
            logging.error(
                "TextScriptConvertor.parse_pronunciation: invalid line: %s, \
                        error pronunciation: %s, word count mismatch with word index",
                line_num,
                pron,
            )
            return False

        return True

    def LoadF2TMap(self, file_path):
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                elements = line.split("\t")
                if len(elements) != 2:
                    logging.error(
                        "TextScriptConvertor.LoadF2TMap: invalid line: %s", line
                    )
                    continue
                key = elements[0]
                value = elements[1]
                value_list = value.split(" ")
                if key in self.m_f2p_map:
                    logging.error(
                        "TextScriptConvertor.LoadF2TMap: duplicate key: %s", key
                    )
                self.m_f2p_map[key] = value_list

    def LoadS2PMap(self, file_path):
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                elements = line.split("\t")
                if len(elements) != 2:
                    logging.error(
                        "TextScriptConvertor.LoadS2PMap: invalid line: %s", line
                    )
                    continue
                key = elements[0]
                value = elements[1]
                if key in self.m_s2p_map:
                    logging.error(
                        "TextScriptConvertor.LoadS2PMap: duplicate key: %s", key
                    )
                self.m_s2p_map[key] = value

    def InitSyllableFormatter(self, targetLang):
        if targetLang == Language.ZhCN:
            if len(self.m_s2p_map) == 0:
                logging.error(
                    "TextScriptConvertor.InitSyllableFormatter: ZhCN syllable to phone map is empty"
                )
                return None
            return ZhCNSyllableFormatter(self.m_s2p_map)
        elif targetLang == Language.PinYin:
            if len(self.m_s2p_map) == 0:
                logging.error(
                    "TextScriptConvertor.InitSyllableFormatter: PinYin syllable to phone map is empty"
                )
                return None
            return PinYinSyllableFormatter(self.m_s2p_map)
        elif targetLang == Language.ZhHK:
            if len(self.m_s2p_map) == 0:
                logging.error(
                    "TextScriptConvertor.InitSyllableFormatter: ZhHK syllable to phone map is empty"
                )
                return None
            return ZhHKSyllableFormatter(self.m_s2p_map)
        elif targetLang == Language.WuuShanghai:
            if len(self.m_s2p_map) == 0:
                logging.error(
                    "TextScriptConvertor.InitSyllableFormatter: WuuShanghai syllable to phone map is empty"
                )
                return None
            return WuuShanghaiSyllableFormatter(self.m_s2p_map)
        elif targetLang == Language.Sichuan:
            if len(self.m_s2p_map) == 0:
                logging.error(
                    "TextScriptConvertor.InitSyllableFormatter: Sichuan syllable to phone map is empty"
                )
                return None
            return SichuanSyllableFormatter(self.m_s2p_map)
        elif targetLang == Language.EnGB:
            formatter = EnXXSyllableFormatter(Language.EnGB)
            if len(self.m_f2p_map) != 0:
                formatter.m_f2t_map = self.m_f2p_map
            return formatter
        elif targetLang == Language.EnUS:
            formatter = EnXXSyllableFormatter(Language.EnUS)
            if len(self.m_f2p_map) != 0:
                formatter.m_f2t_map = self.m_f2p_map
            return formatter
        else:
            logging.error(
                "TextScriptConvertor.InitSyllableFormatter: unsupported language: %s",
                targetLang,
            )
            return None

    def process(self, textScriptPath, outputXMLPath, outputMetafile):
        script = Script(self.m_phoneset, self.m_posset)
        formatted_lines = format_prosody(textScriptPath)
        line_num = 0
        for line in tqdm(formatted_lines):
            if line_num % 2 == 0:
                sentence = line.strip()
                item = self.parse_sentence(sentence, line_num)
            else:
                if item is not None:
                    pronunciation = line.strip()
                    res = self.parse_pronunciation(item, pronunciation, line_num)
                    if res:
                        script.m_items.append(item)

            line_num += 1

        script.Save(outputXMLPath)
        logging.info("TextScriptConvertor.process:\nSave script to: %s", outputXMLPath)

        meta_lines = script.SaveMetafile()
        speaker = self.m_speaker

        meta_lines_tagged = []
        for line in meta_lines:
            line_id, line_text = line.split("\t")
            emo = self.m_emo_dict.get(line_id, "emotion_neutral")
            syll_items = line_text.split(" ")
            syll_items_tagged = []
            for syll_item in syll_items:
                syll_item_tagged = syll_item[:-1] + "$" + emo + "$" + speaker + "}"
                syll_items_tagged.append(syll_item_tagged)
            meta_lines_tagged.append(line_id + "\t" + " ".join(syll_items_tagged))
        with open(outputMetafile, "w") as f:
            for line in meta_lines_tagged:
                f.write(line + "\n")

        logging.info(
            "TextScriptConvertor.process:\nSave metafile to: %s", outputMetafile
        )

    @staticmethod
    def turn_text_into_bytes(plain_text_path, output_meta_file_path, speaker):
        meta_lines = []
        with open(plain_text_path, "r") as in_file:
            for text_line in in_file:
                [sentence_id, sentence] = text_line.strip().split("\t")
                sequence = []
                for character in sentence:
                    hex_string = character.encode("utf-8").hex()
                    i = 0
                    while i < len(hex_string):
                        byte_hex = hex_string[i : i + 2]
                        bit_array = BitArray(hex=byte_hex)
                        integer = bit_array.uint
                        if integer > 255:
                            logging.error(
                                "TextScriptConverter.turn_text_into_bytes: invalid byte conversion in sentence {} \
                                        character {}: (uint) {} - (hex) {}".format(
                                    sentence_id,
                                    character,
                                    integer,
                                    character.encode("utf-8").hex(),
                                )
                            )
                            continue
                        sequence.append(
                            "{{{}$emotion_neutral${}}}".format(integer, speaker)
                        )
                        i += 2
                if sequence[-1][1:].split("$")[0] not in ["33", "46", "63"]:
                    sequence.append("{{46$emotion_neutral${}}}".format(speaker))
                meta_lines.append("{}\t{}\n".format(sentence_id, " ".join(sequence)))
        with open(output_meta_file_path, "w") as out_file:
            out_file.writelines(meta_lines)


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("TextScriptConvertor.main: start")

    if args.speaker is None:
        speaker = os.path.dirname(os.path.dirname(args.text_script_path))
    else:
        speaker = args.speaker

    convertor = TextScriptConvertor(
        args.phoneset_path,
        args.posset_path,
        args.language,
        args.foreignLang,
        args.f2t_map_path,
        args.s2p_map_path,
        args.emo_tag_path,
        speaker,
    )

    convertor.process(args.text_script_path, args.output_xml_path, args.output_metafile)

    logging.info("TextScriptConvertor.main: end")


#  TODO(jin): add emotional style; add speaker info;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextScriptConvertor")
    parser.add_argument("--language", type=str, required=True, help="target language")
    parser.add_argument(
        "--foreignLang", type=str, required=True, help="foreign language"
    )
    parser.add_argument(
        "--phoneset_path", type=str, required=True, help="phoneset path"
    )
    parser.add_argument("--posset_path", type=str, required=True, help="posset path")
    parser.add_argument("--f2t_map_path", type=str, required=True, help="f2t map path")
    parser.add_argument("--s2p_map_path", type=str, required=True, help="s2p map path")
    parser.add_argument(
        "--text_script_path", type=str, required=True, help="input text script path"
    )
    parser.add_argument(
        "--output_xml_path", type=str, required=True, help="output xml path"
    )
    parser.add_argument(
        "--output_metafile", type=str, required=True, help="output metafile path"
    )
    parser.add_argument(
        "--emo_tag_path", type=str, default=None, help="emotion tag path"
    )
    parser.add_argument("--speaker", type=str, default=None, help="speaker")
    args = parser.parse_args()

    main(args)
