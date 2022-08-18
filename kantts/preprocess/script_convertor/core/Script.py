from .XmlObj import XmlObj

import xml.etree.ElementTree as ET
from xml.dom import minidom


class Script(XmlObj):
    def __init__(self, phoneset, posset):
        self.m_phoneset = phoneset
        self.m_posset = posset
        self.m_items = []

    def Save(self, outputXMLPath):
        root = ET.Element("script")

        root.set("uttcount", str(len(self.m_items)))
        root.set("xmlns", "http://schemas.alibaba-inc.com/tts")
        for item in self.m_items:
            item.Save(root)

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(
            indent="  ", encoding="utf-8"
        )
        with open(outputXMLPath, "wb") as f:
            f.write(xmlstr)

    def SaveMetafile(self):
        meta_lines = []

        for item in self.m_items:
            meta_lines.append(item.SaveMetafile())

        return meta_lines
