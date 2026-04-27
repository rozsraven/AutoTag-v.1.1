import html
import re
import os
from lxml import etree

def convert_unicode_punctuation_in_xml(file_path):
    #print(f"Cleaning: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        #print(f"❌ Error reading file: {e}")
        return file_path

    original = content
    content = html.unescape(content)

    replacements = {
        "&": "&amp;",  # Ampersand
        "§": "&#x00A7;",   # Section
        "’": "&#x2019;",   # Right single quote
        "‘": "&#x2018;",   # Left single quote
        "“": "&#x201C;",   # Left double quote
        "”": "&#x201D;",   # Right double quote
        "–": "&#x2013;",   # En dash
        "—": "&#x2014;",   # Em dash
        " ": "&#x2009;",   # Thin space
        "*+": '"',
    }
    for u, ent in replacements.items():
        content = content.replace(u, ent)

    content = re.sub(r'(\w+)=“([^”]*?)”', r'\1="\2"', content)
    content = re.sub(r"(\w+)=‘([^’]*?)’", r"\1='\2'", content)

    if content != original:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            #print("✔ Cleaned and saved (attributes fixed, XML parser-safe).")
        except Exception as e:
            print(f"❌ Error writing file: {e}")
    else:
        print("ℹ No replacements needed; file unchanged.")

    return file_path

def convert_punctuation_in_folder(folder_path, file_extension=".xml"):
    """
    Reads all XML files in a folder and replaces all '&' in text nodes
    with '&amp;' safely.
    """
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(file_extension.lower()):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(file_path, parser)
            root = tree.getroot()
        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")
            continue

        def encode_ampersands(elem):
            """Recursively encode & in text and tail of each element"""
            if elem.text:
                elem.text = elem.text.replace(" & ", " &amp; ")
            if elem.tail:
                elem.tail = elem.tail.replace(" & ", " &amp; ")
            for child in elem:
                encode_ampersands(child)

        encode_ampersands(root)

        try:
            tree.write(file_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
            print(f"✔ Encoded '&' in: {filename}")
        except Exception as e:
            print(f"❌ Error writing {filename}: {e}")