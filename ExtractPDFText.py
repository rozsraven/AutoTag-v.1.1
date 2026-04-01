import csv
import html
import io
import os
import re
import traceback
import xml.etree.ElementTree as xml
from glob import glob
from itertools import tee, zip_longest
from xml.etree.ElementTree import Comment
from xml.sax.saxutils import escape, quoteattr

import bs4
import cv2
# import layoutparser as lp
import numpy as np
import pandas as pd
import pytesseract
import tika
from bs4 import BeautifulSoup
from dateutil.parser import parse
from PIL import Image
from pytesseract import Output
from tika import parser

from opencv_process import discard_large_components, replace_redactions, split_body_and_footnotes
from pdf_ocr_preprocessor import prepare_pdf_images_for_ocr

# set tika server before importing tika
# os.environ['TIKA_SERVER_JAR'] = "file:///C:/Users/GOXAN/Documents/Auto%20Tag/Main/tika-server/tika-server-1.9.jar"
fnstyle_pattern = r"[^.*]*font-size: 7.*?}"
emphasis_pattern =  r"[^.]*italic.*?\}"
user_patterns = r"C:\Users\GOXAN\Documents\Auto Tag\Main\cite.patterns"
custom_config = fr'-c preserve_interword_spaces=1 --oem 1 --psm 1 -l eng'
furtherOrderPattern = re.compile("(F+U+R+T+H+E+R+\s+O+R+D+E+R+:)")
orderPattern = re.compile("(O+R+D+E+R+:)(?!<)")

tika.TikaClientOnly = True

headerplate = "<?xml version=\'1.0\' encoding=\'UTF-8\'?><!DOCTYPE ad:decision-grp PUBLIC \"-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML\" \"D:\\Pub383\\DTD\\DTD\\admin-interp-pubV005-0000\\admin-interp-pubV005-0000.dtd\"><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?>"
footerplate = "<?Pub Caret -2?>"
endMarker = "</ad:decision-grp>"
redactionMarkers = ["xx", "XX", "Xx", "xX"]

def aaoConversion(filecount, file, workDirFolder):
    debugLogs = ""
    htmlFilename = file.replace(".pdf", ".html")
    
    if not os.path.isfile(htmlFilename):
        debugLogs += f"Missing HTML file for {file}"
        return (filecount, "Failed", 0, "Missing HTML file.", debugLogs)
    
    debugLogs += "Start conversion to XML... \n"
    filename = os.path.basename(file)
    print(f"Processing {filename}")
    remarks = []
    
    try:
        year = filename[filename.index("_")-4:filename.index("_")]
        # convert pdf to image
        images = pdfToImage(file, filename, workDirFolder)
        # separate FN from body and save in different folders
        images, fnImages = separateBodyFromFootnote(images)
        rawBody = extractText(images)
        rawFootnotes = extractText(fnImages)
        # replace redaction markers
        for marker in redactionMarkers:
            rawBody = rawBody.replace(marker,"<core:emph typestyle=\"bf\">[redacted]</core:emph>")
            rawFootnotes = rawFootnotes.replace(marker,"<core:emph typestyle=\"bf\">[redacted]</core:emph>")
        
        caseName, caseDate, coreparas, typeOfCase, fileNum = processTikaString(rawBody, filename)

        footnoteTexts = flatten(rawFootnotes)
        cleanedFootnotes = cleanFootnotes(footnoteTexts)
        coreparas = continuityFix(coreparas)
        generatePDFXML(cleanedFootnotes, caseName = caseName, filename = filename, year = year, fileNum = fileNum, decisionDate = caseDate, typeofcases = typeOfCase, coreParas = coreparas, path = workDirFolder, htmlFile=htmlFilename)    
        status = "Successful"
    except Exception as e:
        status= "Failed"
        debugLogs += str(traceback.format_exc()) + "\n"

    if remarks != []:
        remarks = list(set(remarks))
        remarks = "Has " + ", ".join(remarks) + "."
    else:
        remarks = "None."
    
    debugLogs += f"Successfully converted {file} \n"
    flagCount = 0

    return (filecount, status, flagCount, remarks, debugLogs)

def fix_entities(xml_str):
    with open("replace_patterns.csv", mode='r', encoding="utf-8-sig") as infile:
        reader = csv.reader(infile)
        # skip first line
        next(reader)
        entities = dict((rows[0],rows[1]) for rows in reader)

    for entity in entities.keys():    
        xml_str = xml_str.replace(entity, entities[entity])

    return xml_str    
    
def processDataframeStr(data, imgCount, part="Text"):
    footnoteTexts = []
    coreparas = []
    # text = ""
    # clean up blanks
    df = pd.DataFrame(data)
    # df.to_csv(f'data_{part}{str(imgCount)}.csv', index=False)
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    # sort blocks vertically
    sorted_blocks = df1.groupby(
        'block_num').first().sort_values('top').index.tolist()
    # each block of text is either of a different format or a different paragraph
    # print(sorted_blocks)
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        char_w = (sel.width/sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        # mark the start of each block# text += "<start block>"
        for ix, ln in curr.iterrows():
            # add new line when necessary
            # mark each new line
            if prev_par != ln['par_num']:
                # NOTE if the new para is from a new paragraph, append the previous text and start a new one
                if part == "Text":
                    coreparas.append(text)
                    # pass
                else:
                    footnoteTexts.append(text)
                text = ''
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                # text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            # NOTE Next block controls interword extra spaces and pre-line spaces
            added = 0  # num of spaces that should be added
            if ln['left']/char_w > prev_left + 1:
                # print(ln['left'], char_w, prev_left)
                added = int((ln['left'])/char_w) - prev_left
                # NOTE Commenting this to trim down the spaces
                # text += ' ' * added
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        # text += '\n'
        # mark the end of each block
        # text += "<end block>"
        # NOTE Catch any last paragraph of each block
        if text != '':
            if part == "Text":
                # pass
                coreparas.append(text)
            else:
                footnoteTexts.append(text)
    return coreparas, footnoteTexts


def discard(image):
    return discard_large_components(image)


def replaceRedactions(image):
    return Image.open(replace_redactions(image))

# def parse_table_figure_layout(image, imageCount, workDirFolder):
#     """Accepts an image file in memory and detects if there are any Figures or Tables in the given image.
#     Saves a copy of the image with the labels for verification.
#     No threshold for now as a false positive is better than a false negative.
#     Returns True if Table or Figure is present. 
#     """
#     model = lp.Detectron2LayoutModel(
#         config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',  # In model catalog
#         label_map={0: "Text", 1: "Title", 2: "List",
#                    3: "Table", 4: "Figure"},  # In model`label_map`
#         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2] # Optional
#     )
#     # im = cv2.imread(image)
#     im = image
#     lay = model.detect(im)
#     # print(lay)
#     figure_blocks = lp.Layout(
#         [b for b in lay if b.type == 'Figure' or b.type == 'Table'])
#     if len(figure_blocks) > 0:
#         output = lp.visualization.draw_box(
#             im, figure_blocks, id_font_size=40, id_text_color="Red", show_element_type=True)
#         filename = os.path.join(workDirFolder, str(imageCount) + "_Figures.png")
#         output.save(filename, "PNG")
#     return len(figure_blocks) > 0


def pdfToImage(filepath, filename, workDirFolder):
    return prepare_pdf_images_for_ocr(filepath, workDirFolder)


def getFootnotesFromImages(images):
    imgCount = 0
    footnoteTexts = []
    for image in images:
        image = cv2.imread(image)
        result = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        height, width = result.shape[:2]

        contours, hierarchy = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours[:] != []:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if x >= 295 and x <= 306 and ((w >= 599 and w <= 605) or (w >= 2000)): 
                    # upper_portion = img[0:y, 0:width]
                    lower_portion = result[y+h:height, 0:width+100]
                    data = pytesseract.image_to_data(lower_portion, config=custom_config, output_type=Output.DICT)
                    fntext = processDataframeStr(data, imgCount, part="FN")
                    footnoteTexts.append(fntext)
            
        imgCount += 1
    return footnoteTexts


def extractText(images):
    text = ""
    for image in images:
        raw = parser.from_file(image)
        # print(f"{raw=}")
        if raw["content"]:
            text += raw["content"]   
        
    return html.escape(text, quote=False)


def processTikaString(tikaString, filename):
    count = 0
    startParse = False
    coreparas = []
    newPage = False
    dateNotFound = True
    typeOfCase = []
    plainContentStart = False
    caseDate = "caseDateNotFound"
    fileNum = ""
    caseName = "caseNameNotFound"

    typeOfCase.append("AAO Designation: " +
                      filename[filename.index("_")+3:len(filename)-7])
    tikaString = tikaString.split("\n")


    for line in tikaString:
        
        if line in ("", "/N"):
            count += 1
            continue
        elif line.lower().startswith(("matter of", "in re:", "inre:")) and not startParse:
            caseName = line.replace("inre:", "In re:")
            if "date:" in caseName.lower():
                caseDate = caseName[caseName.lower().index("date:"):]
                caseName = caseName[:caseName.lower().index("date:")]
                dateNotFound = False
            count = 0
            startParse = True
        elif line.lower().startswith("date:") and dateNotFound:
            caseDate = line
            count = 0
            dateNotFound = False
        elif line.lower().startswith("appeal of") or line.lower().startswith("motion on"):
            fileNum = line
            count = 0
        # elif line.lower().startswith("petition") or line.lower().startswith("form i-"):
        elif line.lower().startswith("form ") and not plainContentStart:
            typeOfCase.append(line)
            count = 0
        elif startParse:
            if len(line) < 4 or line.replace(" ", "").isdigit():
                newPage = False
                count = 0
                continue
            elif newPage:
                coreparas.append(line)
                plainContentStart = True
                count = 0
                newPage = False
            elif count == 0:
                if line.startswith("•"):
                    coreparas.append(line)
                else:
                    if len(coreparas) > 0:
                        coreparas[-1] = coreparas[-1] + " " + line
                    else:
                        coreparas.append(line)
                count = 0
                plainContentStart = True
            elif count == 1:
                if newPage:
                    if len(coreparas) > 0:
                        coreparas[-1] = coreparas[-1] + " " + line
                    else:
                        coreparas.append(line)
                    newPage = False
                else:
                    coreparas.append(line)
                count = 0
                plainContentStart = True
            elif count > 2:
                newPage = True
                # count = 0
                if line[0].islower():
                    if len(coreparas) > 0:
                        coreparas[-1] = coreparas[-1] + " " + line
                    else:
                        coreparas.append(line)
                else:
                    coreparas.append(line)
                plainContentStart = True
    # print(f"{coreparas=}")
    return caseName, caseDate, coreparas, typeOfCase, fileNum

def cleanFootnotes(footnoteTexts):
    final = []
    for footnote in footnoteTexts:
        if footnoteTexts.index(footnote) == 0:
            final.append(footnote)
        # second boolean is to check if a redaction is followed by another redaction or possible footnote
        elif final[-1].endswith((".", '.”')) or (final[-1].endswith("</core:emph>") and footnote[0].isdigit()):
            final.append(footnote)
        else:
            final[-1] += footnote
    return final

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def findFNIndices(cleanedFootnotes, coreparas):
    ixToDel = set()
    for text in cleanedFootnotes:
        # print("Find: ", text[1:30])
        out = index_containing_substring(coreparas, text[1:30])
        if out > 0:
            # print("Found in:", out, coreparas[out])
            ixToDel.add(out)
        else:
            # print("Not Found", out)
            out = index_containing_substring(coreparas, text[-30:])
            # print("Tried again:", text[-30:], "\n", out)
            if out > 0:
                ixToDel.add(out)
        # print("")
    return ixToDel


def continuityFix(coreparas):
    sortedParas = []
    for count, para in enumerate(coreparas):
        para = para.strip()
        if para == "":
            continue
        elif count == 0:
            sortedParas.append(para)
        elif para[0].islower():
            sortedParas[-1] = sortedParas[-1] + " " + para
        else:
            sortedParas.append(para)
    return sortedParas

def replaceLists(xmlString, htmlFile): 
    xmlString = html.unescape(xmlString)
    
    with open(htmlFile, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        for tag in soup.find_all(True):
            tag.attrs = {}
    out = soup.find_all(['ul'])
    
    if out == []: 
        return xmlString
    
    finalLists = []
    # get all html lists
    for ulList in out:
        if len(finalLists) == 0:
            finalLists.append(ulList)
        else:
            for item in finalLists:
                count = 1
                if str(ulList) in str(item):
                    break
                elif count ==  len(finalLists) and str(ulList) not in str(item):
                    finalLists.append(ulList)
                count += 1
    
    listMarkers = []
    # NOTE: 7/11 Return early if we can't determine position. Fix conversion errors.
    try:
        # get the markers for the lists
        for li in finalLists:
            # Try getting the index of the text instead of its previous sibling
            # pre = li.previous_sibling
            pre = li.previous_sibling
            while pre.text == "":
                pre = pre.previous_sibling
            pos = li.next_sibling
            while pos.text == "":
                pos = pos.next_sibling
            listMarkers.append((pre.text,pos.text))
    except:
        return xmlString.replace("<ad:judgmentbody>", f"<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Lists not Tagged Properly</core:flag-subject><core:message><core:flag.para>Please check source file to check if core:list tagging is needed.</core:flag.para></core:message></core:flag></core:para>")    
    # convert html list to xml list
    xmlLists = []
    for list in finalLists:
        soup = list
        ul = soup.find_all("ul")
        if ul:
            for l in ul:
                l.name = "core:list"
        li = soup.find_all("li")
        if li:
            for l in li:
                l.name = "core:listitem"        
        p = soup.find_all("p")
        if p:
            for l in p:
                l.name = "core:para"
        
        h = soup.find_all("h2")
        if h:
            for l in h:
                l.name = "core:generic-hd"
        s = soup.find_all("span")
        if s:
            for l in s:
                l.unwrap()
        soup.name = "core:list"
        xmlLists.append(soup)
    
    # replace parts of the xml string with the xml lists based on the markers. 
    for marker in listMarkers:
        pre, pos = marker
        if pre in xmlString:
            preix = xmlString.index(pre) + len(pre)
        elif pre[-len(pre)//2:] in xmlString:
            preix = xmlString.index(pre[-len(pre)//2:]) + len(pre)//2
        elif pre[-len(pre)//3:] in xmlString:
            preix = xmlString.index(pre[-len(pre)//3:]) + len(pre)//3
        elif pre[-20:] in xmlString:
            preix = xmlString.index(pre[-20:]) + 20
        else:
            preix = "Not Found"
        if pos in xmlString:
            posix = xmlString.index(pos)
        elif pos[:len(pos)//2] in xmlString:
            posix = xmlString.index(pos[:len(pos)//2]) 
        elif pos[:len(pos)//3] in xmlString:
            posix = xmlString.index(pos[:len(pos)//3]) 
        elif pos[:30] in xmlString:
            posix = xmlString.index(pos[:30]) 
        else:
            posix = "Not Found"
        
        # return xmlString with a flag if either is not found or if invalid preix was found
        if preix == "Not Found" or posix == "Not Found" or posix < preix:
            return xmlString.replace("<ad:judgmentbody>", f"<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Lists not Tagged Properly</core:flag-subject><core:message><core:flag.para>Please check source file to check if core:list tagging is needed.</core:flag.para></core:message></core:flag></core:para>")   
        
        while True:
            if xmlString[preix] == ">":
                preix += 1
                break
            else:
                preix += 1
        while True:
            if xmlString[posix] == "<":
                break
            else:
                posix -= 1
        xmlString = xmlString[:preix] + str(xmlLists[listMarkers.index(marker)]) + xmlString[posix:]
    return xmlString

def generatePDFXML(cleanFNs, caseName, filename, year, fileNum, decisionDate, typeofcases, coreParas, path, htmlFile=None):
    root = xml.Element("ad:decision-grp", volnum=year)
    adjudication = xml.SubElement(root, "ad:adjudication", volnum=year)

    adjudicationInfo = xml.SubElement(adjudication, "ad:adjudication-info")

    adRegarding = xml.SubElement(adjudicationInfo, "ad:regarding")
    adFileNum = xml.SubElement(adjudicationInfo, "ad:filenum")
    adFileNum.text = "OFFICE: " + fileNum
    caseNameTag = xml.SubElement(adRegarding, "core:para", indent="none")
    caseNameTag.text = caseName
    adAdjudicatorInfo = xml.SubElement(adjudicationInfo, "ad:adjudicator-info")
    coreJuris = xml.SubElement(adAdjudicatorInfo, "core:juris")
    lnciJurisInfo = xml.SubElement(coreJuris, "lnci:jurisinfo")
    xml.SubElement(lnciJurisInfo, "lnci:usa")
    coreJurisName = xml.SubElement(coreJuris, "core:juris-name")
    coreJurisName.text = "Administrative Appeals Office"

    adDates = xml.SubElement(adjudicationInfo, "ad:dates")
    adDecisionDate = xml.SubElement(adDates, "ad:decisiondate")
    adDecisionDate.text = decisionDate.upper().replace(".", "")
    adTypeOfCase = xml.SubElement(adjudicationInfo, "ad:typeofcase")
    
    for caseType in typeofcases:
        
        xml.SubElement(adTypeOfCase, "core:generic-hd",
                       attrib={"align":"left", "typestyle":"ro"}).text = caseType

    adContent = xml.SubElement(adjudication, "ad:content")
    adJudgments = xml.SubElement(adContent, "ad:judgments")
    adOpinion = xml.SubElement(adJudgments, "ad:opinion")

    adJudgmentBody = xml.SubElement(adOpinion, "ad:judgmentbody")

    for para in coreParas:
        match = re.match("[A-Z\s.]*", para)
        if para.startswith("Non-Precedent Decision of") or para.strip() == "":
            continue
        elif match != None and match[0] == para:
            # print("genhead", para)
            genHead = xml.SubElement(adJudgmentBody,"core:generic-hd", attrib={"align":"center", "typestyle":"ro"})
            genHead.text = para
        else:
            corePara = xml.SubElement(adJudgmentBody, "core:para", indent="none")
            corePara.text = para
            
    # add the filename as generic head at the end
    xml.SubElement(adJudgmentBody,"core:generic-hd", attrib={"align":"left", "typestyle":"ro"}).text = filename
    # compile the output XML
    output = xml.ElementTree(root)
    # read output XML in io as string
    out = io.BytesIO()
    output.write(out)
    xml_str = out.getvalue().decode()
    # print(xml_str)
    # add header part of xml
    xml_str = headerplate + xml_str

    # replace lists with the html counterpart
    xml_str = replaceLists(xml_str, htmlFile)

    # add footnotes and emphasis here:
    xml_str = add_footnotes(cleanFNs, xml_str, htmlFile=htmlFile)
    
    xml_str = clean_xml_spaces(xml_str)
    xml_str = add_emphasis_to_xml(xml_str, htmlFile=htmlFile)
    xml_str = fix_entities(xml_str)
    
    with open(f"{path}\{filename[:-4]}.xml", "w", encoding="utf-8") as xmlfile:
        xmlfile.write(xml_str)

def flatten(footnoteTexts):
    footnoteTexts = footnoteTexts.split("\n")
    footnoteTexts = [item for item in footnoteTexts if len(item) > 3]
    return footnoteTexts

def flag_footnotes(unplaced_fns, xmlFile):
    flag = [f"<core:flag.para>{fn}</core:flag.para>" for fn in unplaced_fns]
    flag = "".join(flag)
    xmlFile = xmlFile.replace(
            "<ad:judgmentbody>", f"<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Correct placement for footnote not found</core:flag-subject><core:message><core:flag.para>Please check source file to check proper placement:</core:flag.para>{flag}</core:message></core:flag></core:para>")
    return xmlFile

def handle_navigable_strings(xmlFile, span, placed_fns, fn_number, fn, unplaced_fns):
    """If there is a superscript html tag with a matching footnote number, this function will handle
    determining the proper placement. The footnote will be placed in the XML or flagged. 

    Args:
        xmlFile (str): The xml string.
        span (navigable_string): The span that contains the footnote number.
        placed_fns (list): List of all footnotes placed in the xml.
        fn_number (int): The footnote number based on the OCR.
        fn (str): The footnote text to placed on the XML.

    Returns:
        str: The xml string with the footnote placed or flagged.
    """
    prevtext = span.previous_sibling
    if isinstance(prevtext, bs4.element.NavigableString):
        marker = str(prevtext.string)
        marker = " ".join(marker.split())
        ix = get_xmlstr_index(xmlFile, marker)
        if ix != 0 and fn_number not in placed_fns:
            xmlFile = xmlFile[:ix] + f'<fn:footnote fr="{str(int(fn_number))}"><fn:para>' + \
            fn.split(" ",1)[1] + "</fn:para></fn:footnote>" + \
            xmlFile[ix+len(fn_number.strip()):]
            placed_fns.append(fn_number)
        else:
            unplaced_fns.append(fn)       
    elif prevtext != None:
        marker = str(prevtext.previous_sibling) + \
            str(prevtext.contents[0].string)
        marker = " ".join(marker.split())
        ix = get_xmlstr_index(xmlFile, marker)
        if ix != 0 and fn_number not in placed_fns:
            xmlFile = xmlFile[:ix] + f'<fn:footnote fr="{str(int(fn_number))}"><fn:para>' + \
            fn.split(" ",1)[1] + "</fn:para></fn:footnote>" + \
            xmlFile[ix+len(fn_number.strip()):]
            placed_fns.append(fn_number)
        else:
            unplaced_fns.append(fn)
    else:
        unplaced_fns.append(fn)

    return xmlFile, placed_fns, unplaced_fns

def add_footnotes(cleanFNs, xmlFile, htmlFile=None):
    with open(htmlFile, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    style_soup = soup.find_all(['style'])   
    stylesheet = " ".join(str(style_soup).split())
    fn_classes = re.findall(fnstyle_pattern, stylesheet)
    fn_classes = [sub[0:3].strip() for sub in fn_classes]
    fn_classes = re.compile("|".join(fn_classes))
    placed_fns = []
    unplaced_fns = []
    for fn in cleanFNs:
        fn = fn.strip()
        fn_number = fn.split(" ", 1)[0]
        if fn_number.isdigit():
            span = soup.find("span", text=fn_number, class_=fn_classes)
            if span != None:
                xmlFile, placed_fns, unplaced_fns = handle_navigable_strings(xmlFile, span, placed_fns, fn_number, fn, unplaced_fns) 
            else:
                span = soup.find("span", text=fn_number, style=re.compile("7.*"))
                if span != None:
                    xmlFile, placed_fns, unplaced_fns = handle_navigable_strings(xmlFile, span, placed_fns, fn_number, fn, unplaced_fns)                    
                else:
                    unplaced_fns.append(fn)
        else:
            unplaced_fns.append(fn)

    if len(unplaced_fns) > 0:
        xmlFile = flag_footnotes(unplaced_fns, xmlFile)
        
    return xmlFile

def clean_xml_spaces(xmlFile):
    xmlFile = re.sub('\s+', ' ', xmlFile)
    return xmlFile

def get_xmlstr_index(xmlFile, marker):
    try:
        if xmlFile.count(marker) > 1:
            return 0
        ix = xmlFile.index(marker)
        return ix + len(marker)
    except:
        marker = marker[len(marker)//2:]
        # print("Halved: ", marker)
        try:
            if xmlFile.count(marker) > 1:
                return 0
            ix = xmlFile.index(marker)
            return ix + len(marker)
        except:
            marker = marker[len(marker)//3:]
            # print("Third: ", marker)
            try:
                if xmlFile.count(marker) > 1:
                    return 0
                ix = xmlFile.index(marker)
                return ix + len(marker)
            except:
                return 0


def add_emphasis_to_xml(xmlFile, htmlFile=None):
    filepath = htmlFile
    with open(filepath, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    out = soup.find_all(['style'])
    style_str = str(out)
    style_str = " ".join(style_str.split())
    i_emphs = soup.find_all("i")
    xmlFile = " ".join(xmlFile.split())
    for emph in i_emphs:
        content = emph.string
        prevtext = emph.previous_sibling
        nextext = emph.next_sibling
        marker = ""
        ix = 0
        if prevtext is not None:
            marker = prevtext.string + content
            marker = " ".join(marker.split())
            ix = get_xmlstr_index(xmlFile, marker)
        # NOTE: For now I would be disablign lookahead. Lookback should work for now
        # elif ix == 0 and nextext is not None:
        #     marker = content + nextext.string
        #     marker = " ".join(marker.split())
        #     ix = get_xmlstr_index(xmlFile, marker)
        if ix != 0:
            content = content.strip()
            new_text = "~i~" + content + "~/i~"
            xmlFile = xmlFile[:ix-len(content)] + f"{new_text}" + xmlFile[ix:]
            # print("success")
        else:
            continue
            # print(f"Can't find marker: {marker}")

    xmlFile = xmlFile.replace("~i~", "<core:emph typestyle="'"it"'">").replace(
        "~/i~", "</core:emph>")
    
    xmlFile = re.sub(furtherOrderPattern, "<core:emph typestyle="'"bf"'">FURTHER ORDER</core:emph>:", xmlFile)
    xmlFile = re.sub(orderPattern, "<core:emph typestyle="'"bf"'">ORDER</core:emph>:", xmlFile) 
    
    
    return xmlFile


def flag_emphasis(xmlFile):
    print("Cant find emphasis")
    return xmlFile


def separateBodyFromFootnote(images):
    return split_body_and_footnotes(images)
