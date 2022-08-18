from .XmlObj import XmlObj

import xml.etree.ElementTree as ET


#  TODO(jin): Not referenced, temporarily commented
class WrittenSentence(XmlObj):
    def __init__(self, posset):
        self.m_written_word_list = []
        self.m_written_mark_list = []
        self.m_posset = posset
        self.m_align_list = []
        self.m_alignCursor = 0
        self.m_accompanyIndex = 0
        self.m_sequence = ""
        self.m_text = ""

    def AddHost(self, writtenWord):
        self.m_written_word_list.append(writtenWord)
        self.m_align_list.append(self.m_alignCursor)

    def LoadHost(self):
        pass

    def SaveHost(self):
        pass

    def AddAccompany(self, writtenMark):
        self.m_written_mark_list.append(writtenMark)
        self.m_alignCursor += 1
        self.m_accompanyIndex += 1

    def SaveAccompany(self):
        pass

    def LoadAccompany(self):
        pass

    #  Get the mark span corresponding to specific spoken word
    def GetAccompanySpan(self, host_index):
        if host_index == -1:
            return (0, self.m_align_list[0])

        accompany_begin = self.m_align_list[host_index]
        accompany_end = (
            self.m_align_list[host_index + 1]
            if host_index + 1 < len(self.m_written_word_list)
            else len(self.m_written_mark_list)
        )

        return (accompany_begin, accompany_end)

    # TODO: iterable
    def GetElements(self):
        accompany_begin, accompany_end = self.GetAccompanySpan(-1)
        res_lst = [
            self.m_written_mark_list[i] for i in range(accompany_begin, accompany_end)
        ]

        for j in range(len(self.m_written_word_list)):
            accompany_begin, accompany_end = self.GetAccompanySpan(j)
            res_lst.extend([self.m_written_word_list[j]])
            res_lst.extend(
                [
                    self.m_written_mark_list[i]
                    for i in range(accompany_begin, accompany_end)
                ]
            )

        return res_lst

    def BuildSequence(self):
        self.m_sequence = " ".join([str(ele) for ele in self.GetElements()])

    def BuildText(self):
        self.m_text = "".join([str(ele) for ele in self.GetElements()])


class SpokenSentence(XmlObj):
    def __init__(self, phoneset):
        self.m_spoken_word_list = []
        self.m_spoken_mark_list = []
        self.m_phoneset = phoneset
        self.m_align_list = []
        self.m_alignCursor = 0
        self.m_accompanyIndex = 0
        self.m_sequence = ""
        self.m_text = ""

    def __len__(self):
        return len(self.m_spoken_word_list)

    def AddHost(self, spokenWord):
        self.m_spoken_word_list.append(spokenWord)
        self.m_align_list.append(self.m_alignCursor)

    def SaveHost(self):
        pass

    def LoadHost(self):
        pass

    def AddAccompany(self, spokenMark):
        self.m_spoken_mark_list.append(spokenMark)
        self.m_alignCursor += 1
        self.m_accompanyIndex += 1

    def SaveAccompany(self):
        pass

    #  Get the mark span corresponding to specific spoken word
    def GetAccompanySpan(self, host_index):
        if host_index == -1:
            return (0, self.m_align_list[0])

        accompany_begin = self.m_align_list[host_index]
        accompany_end = (
            self.m_align_list[host_index + 1]
            if host_index + 1 < len(self.m_spoken_word_list)
            else len(self.m_spoken_mark_list)
        )

        return (accompany_begin, accompany_end)

    # TODO: iterable
    def GetElements(self):
        accompany_begin, accompany_end = self.GetAccompanySpan(-1)
        res_lst = [
            self.m_spoken_mark_list[i] for i in range(accompany_begin, accompany_end)
        ]

        for j in range(len(self.m_spoken_word_list)):
            accompany_begin, accompany_end = self.GetAccompanySpan(j)
            res_lst.extend([self.m_spoken_word_list[j]])
            res_lst.extend(
                [
                    self.m_spoken_mark_list[i]
                    for i in range(accompany_begin, accompany_end)
                ]
            )

        return res_lst

    def LoadAccompany(self):
        pass

    def BuildSequence(self):
        self.m_sequence = " ".join([str(ele) for ele in self.GetElements()])

    def BuildText(self):
        self.m_text = "".join([str(ele) for ele in self.GetElements()])

    def Save(self, parent_node):
        spoken_node = ET.SubElement(parent_node, "spoken")
        spoken_node.set("wordcount", str(len(self.m_spoken_word_list)))

        text_node = ET.SubElement(spoken_node, "text")
        text_node.text = self.m_sequence

        #  TODO: spoken mark might be used
        for word in self.m_spoken_word_list:
            word.Save(spoken_node)

    def SaveMetafile(self):
        meta_line_list = [word.SaveMetafile() for word in self.m_spoken_word_list]

        return " ".join(meta_line_list)


class ScriptSentence(XmlObj):
    def __init__(self, phoneset, posset):
        self.m_phoneset = phoneset
        self.m_posset = posset
        self.m_writtenSentence = WrittenSentence(posset)
        self.m_spokenSentence = SpokenSentence(phoneset)
        self.m_text = ""

    def Save(self, parent_node):
        if len(self.m_spokenSentence) > 0:
            self.m_spokenSentence.Save(parent_node)

    def SaveMetafile(self):
        if len(self.m_spokenSentence) > 0:
            return self.m_spokenSentence.SaveMetafile()
        else:
            return ""
