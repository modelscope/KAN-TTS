import ttsfrd

ENG_LANG_MAPPING = {
    "PinYin": "zh-cn",
    "English": "en-us",
    "British": "en-gb",
    "ZhHK": "hk_cantonese",
    "Sichuan": "sichuan",
    "Japanese": "japanese",
    "WuuShangHai": "shanghai",
    "Indonesian": "indonesian",
    "Malay": "malay",
    "Filipino": "filipino",
    "Vietnamese": "vietnamese",
    "Korean": "korean",
    "Russian": "russian",
}


def text_to_mit_symbols(texts, resources_dir, speaker, lang="PinYin"):
    fe = ttsfrd.TtsFrontendEngine()
    fe.initialize(resources_dir)
    fe.set_lang_type(ENG_LANG_MAPPING[lang])

    symbols_lst = []
    for idx, text in enumerate(texts):
        text = text.strip()
        res = fe.gen_tacotron_symbols(text)
        res = res.replace("F7", speaker)
        sentences = res.split("\n")
        for sentence in sentences:
            arr = sentence.split("\t")
            # skip the empty line
            if len(arr) != 2:
                continue
            sub_index, symbols = sentence.split("\t")
            symbol_str = "{}_{}\t{}\n".format(idx, sub_index, symbols)
            symbols_lst.append(symbol_str)

    return symbols_lst
