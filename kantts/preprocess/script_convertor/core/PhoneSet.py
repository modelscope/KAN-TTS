import xml.etree.ElementTree as ET
import logging

from .XmlObj import XmlObj
from .Phone import Phone


class PhoneSet(XmlObj):
    def __init__(self, phoneset_path):
        self.m_phone_list = []
        self.m_id_map = {}
        self.m_name_map = {}
        self.Load(phoneset_path)

    def Load(self, file_path):
        #  alibaba tts xml namespace
        ns = "{http://schemas.alibaba-inc.com/tts}"

        phoneset_root = ET.parse(file_path).getroot()
        for phone_node in phoneset_root.findall(ns + "phone"):
            phone = Phone()
            phone.Load(phone_node)
            self.m_phone_list.append(phone)
            if phone.m_id in self.m_id_map:
                logging.error("PhoneSet.Load: duplicate id: %d", phone.m_id)
            self.m_id_map[phone.m_id] = phone

            if phone.m_name in self.m_name_map:
                logging.error("PhoneSet.Load duplicate name name: %s", phone.m_name)
            self.m_name_map[phone.m_name] = phone

    def Save(self):
        pass


#  if __name__ == "__main__":
#      import os
#      import sys
#
#      phoneset = PhoneSet()
#      phoneset.Load(sys.argv[1])
#
#      for phone in phoneset.m_phone_list:
#          print(phone)
#          print(phone.m_id)
#          print(phone.m_name)
#          print(phone.m_cv_type)
#          print(phone.m_if_type)
#          print(phone.m_uv_type)
#          print(phone.m_ap_type)
#          print(phone.m_am_type)
#          print(phone.m_bnd)
