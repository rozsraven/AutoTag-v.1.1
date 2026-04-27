"""BIA DOCX-to-XML conversion helpers extracted from the project.

This module is intentionally standalone and does not update existing callers.
"""

import io
import os
import re
import shutil
import sys
import xml.etree.ElementTree as xml

import win32com.client
from bs4 import BeautifulSoup
from docx import Document
from dateutil.parser import parse
from win32com.client import constants

headerplate = r"""<?xml version="1.0" encoding="UTF-8"?><!--Arbortext, Inc., 1988-2015, v.4002--><!DOCTYPE ad:decision-grp PUBLIC "-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML" "C:\Neptune\NeptuneEditor\doctypes\admin-interp-pubV005-0000\admin-interp-pubV005-0000.dtd"><?Pub UDT delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT plpt_delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT plpt_add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT notes_delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT notes_add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT EditAid-UC _display FontColor="orange" Composed="yes" Save="yes"?><?Pub UDT EditAid-Quotes _display FontColor="green" Composed="yes" Save="yes"?><?Pub UDT EditAid-Hyphens _display FontColor="violet" Composed="yes" Save="yes"?><?Pub UDT Mod _pi?><?Pub UDT AttrModified _display BackColor="#c0c0c0" StrikeThrough="no" Composed="yes" Save="yes"?><?Pub UDT Deleted _display BackColor="#ffffc0" Composed="yes" Save="yes"?><?Pub UDT Inserted _display BackColor="#ffc0ff" Composed="yes" Save="yes"?><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?>"""
footerplate = "<?Pub Caret -2?>"
endMarker = "</ad:decision-grp>"

expandedForm = {
	"~/ib~": "</core:emph>",
	"~/IB~": "</core:emph>",
	"~/i~": "</core:emph>",
	"~/I~": "</core:emph>",
	"~/b~": "</core:emph>",
	"~/B~": "</core:emph>",
	"~b~": '<core:emph typestyle = "bf">',
	"~B~": '<core:emph typestyle = "bf">',
	"~i~": '<core:emph typestyle="it">',
	"~I~": '<core:emph typestyle="it">',
	"~ib~": '<core:emph typestyle="ib">',
	"~IB~": '<core:emph typestyle="ib">',
	"~u~": '<core:emph typestyle="un">',
	"~U~": '<core:emph typestyle="un">',
	"&N": "&amp;N",
}

with open("all_cities.txt", "r") as f:
	cities = " ".join(f.readlines())


class BIAXMLConverter:
	def biaConversion(self, filecount, file, workDirFolder):
		debugLogs = ""
		htmlFilename = re.sub(r"\.docx$", ".html", file, flags=re.IGNORECASE)
		if not file.lower().endswith(".docx"):
			debugLogs += f"Input must be a DOCX file: {file}"
			return (filecount, "Failed", 0, "Input must be a DOCX file.", debugLogs)
		if not os.path.isfile(file):
			debugLogs += f"Missing DOCX file for {file}"
			return (filecount, "Failed", 0, "Missing DOCX file.", debugLogs)
		if not os.path.isfile(htmlFilename):
			debugLogs += f"Missing HTML file for {file}"
			return (filecount, "Failed", 0, "Missing HTML file.", debugLogs)

		filename = os.path.basename(file)
		remarks = []
		try:
			year_match = re.search(r"(\d{4})", filename)
			year = year_match.group(1) if year_match else "0000"
			caseName, fileNum, decisionDate, typeofcases, coreParas = self.extract_docx_content(file)
			adJurisName = "Board of Immigration Appeals"
			flaggedDocs = []
			flagCount = 0
			self.generateXML(False, filename, year, caseName, fileNum, adJurisName, decisionDate, typeofcases, [], coreParas, [], workDirFolder, False, False, [], [], False, False, [], flaggedDocs, flagCount, debugLogs)
			status = "Successful"
		except Exception as exc:
			status = "Failed"
			debugLogs += str(exc)
		remarks = "None." if not remarks else "Has " + ", ".join(sorted(set(remarks))) + "."
		return (filecount, status, 0, remarks, debugLogs)

	def index_containing_substring(self, the_list, substring):
		for i, s in enumerate(the_list):
			if substring in s:
				return i
		return None

	def extract_docx_content(self, docx_file):
		document = Document(docx_file)
		paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
		caseName = paragraphs[0] if paragraphs else "caseNameNotFound"
		fileNum = next((p for p in paragraphs if "appeal" in p.lower() or "a " in p.lower()), "fileNumNotFound")
		decisionDate = next((p for p in paragraphs if self.is_date(p, fuzzy=True)), "decisionDateNotFound")
		typeofcases = [p for p in paragraphs if p.lower().startswith("form ")]
		coreParas = paragraphs[1:]
		return caseName, fileNum, decisionDate, typeofcases, coreParas

	def manual_xml(self, text):
		if text.startswith("~gh~"):
			return ("<core:generic-hd align=\"center\">" + text + "</core:generic-hd>").replace("~gh~", "").replace("~li~", "")
		if text.startswith("~q~"):
			text = text.replace("~li~", "", 1)
			return text.replace("~q~", "")
		if text.startswith("~li~"):
			return text.replace("~li~", "", 1)
		if text.startswith("~in0~"):
			return text.replace("~in0~", "")
		return text

	def get_city_from_filenum(self, filenum):
		return filenum[filenum.rindex("-") + 1: filenum.rindex(",")].strip()

	def check_if_valid_city(self, city):
		return city in cities

	def is_date(self, string, fuzzy=False):
		try:
			parse(string, fuzzy=fuzzy)
			return True
		except Exception:
			return False

	def expandTokens(self, xml_str, dic):
		for short, expanded in dic.items():
			xml_str = xml_str.replace(short, expanded)
		return xml_str

	def clean_xml_spaces(self, xmlFile):
		return re.sub(r"\s+", " ", xmlFile)

	def add_emphasis_to_xml(self, xmlFile, htmlFile=None):
		if not htmlFile or not os.path.isfile(htmlFile):
			return xmlFile
		with open(htmlFile, "r", encoding="utf-8", errors="ignore") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		for emph in soup.find_all(["i", "em"]):
			content = emph.get_text(strip=True)
			if not content:
				continue
			xmlFile = xmlFile.replace(content, f'<core:emph typestyle="it">{content}</core:emph>', 1)
		return xmlFile

	def save_as_docx(self, path, workDirFolder):
		try:
			word = win32com.client.gencache.EnsureDispatch("Word.Application")
		except AttributeError:
			for module in [m.__name__ for m in sys.modules.values()]:
				if re.match(r"win32com\.gen_py\..+", module):
					del sys.modules[module]
			shutil.rmtree(os.path.join(os.environ.get("LOCALAPPDATA"), "Temp", "gen_py"))
			word = win32com.client.gencache.EnsureDispatch("Word.Application")

		doc = word.Documents.Open(path)
		doc.Visible = True
		doc.Activate()
		new_file_abs = os.path.join(workDirFolder, "docx", os.path.basename(path))
		new_file_abs = re.sub(r"\.\w+$", ".docx", new_file_abs)
		word.ActiveDocument.SaveAs(new_file_abs, FileFormat=constants.wdFormatXMLDocument)
		doc.Close(True)

	def generateXML(self, tablesPresent, filename, year, caseName, fileNum, adJurisName, decisionDate, typeofcases, parties, coreParas, fnParas, path, complete, multiparty, otherCases, afterFNText, fnPresent, multiCaseDoc, dissentText, flaggedDocs, flagCount, debugLogs, htmlFile=None):
		remarks = []
		root = xml.Element("ad:decision-grp", volnum=year)
		adjudication = xml.SubElement(root, "ad:adjudication", volnum=year)
		adjudicationInfo = xml.SubElement(adjudication, "ad:adjudication-info")
		adRegarding = xml.SubElement(adjudicationInfo, "ad:regarding")
		caseNameTag = xml.SubElement(adRegarding, "core:para", {"indent": "none"})
		caseNameTag.text = "".join(caseName) if isinstance(caseName, list) else str(caseName)
		adFileNum = xml.SubElement(adjudicationInfo, "ad:filenum")
		adFileNum.text = fileNum
		adAdjudicatorInfo = xml.SubElement(adjudicationInfo, "ad:adjudicator-info")
		coreJuris = xml.SubElement(adAdjudicatorInfo, "core:juris")
		lnciJurisInfo = xml.SubElement(coreJuris, "lnci:jurisinfo")
		xml.SubElement(lnciJurisInfo, "lnci:usa")
		xml.SubElement(coreJuris, "core:juris-name").text = adJurisName
		adDates = xml.SubElement(adjudicationInfo, "ad:dates")
		xml.SubElement(adDates, "ad:decisiondate").text = "Date: " + decisionDate.replace("~i~", "").replace("~/i~", "")
		adTypeOfCase = xml.SubElement(adjudicationInfo, "ad:typeofcase")
		for caseType in typeofcases:
			xml.SubElement(adTypeOfCase, "core:generic-hd", {"align": "left", "typestyle": "ro"}).text = caseType
		adContent = xml.SubElement(adjudication, "ad:content")
		adJudgments = xml.SubElement(adContent, "ad:judgments")
		adOpinion = xml.SubElement(adJudgments, "ad:opinion")
		adJudgmentBody = xml.SubElement(adOpinion, "ad:judgmentbody")
		for para in coreParas:
			xml.SubElement(adJudgmentBody, "core:para", indent="none").text = para.replace("~in0~", "")
		xml.SubElement(adJudgmentBody, "core:generic-hd", align="center").text = "FOR THE BOARD"
		xml.SubElement(adJudgmentBody, "core:para").text = filename.replace(".docx", ".pdf")
		if dissentText:
			xml.SubElement(adJudgmentBody, "core:para").text = dissentText[0]
		output = xml.ElementTree(root)
		out = io.BytesIO()
		output.write(out)
		xml_str = self.expandTokens(headerplate + out.getvalue().decode().replace(" & ", " &amp; "), expandedForm)
		idx = xml_str.index(endMarker)
		xml_str = xml_str[:idx] + footerplate + xml_str[idx:]
		xml_str = self.clean_xml_spaces(xml_str)
		xml_str = self.add_emphasis_to_xml(xml_str, htmlFile=htmlFile)
		with open(f"{path}\\{os.path.basename(filename)[:-5]}.xml", "w", encoding="utf-8") as xmlfile:
			xmlfile.write(xml_str)
		return flaggedDocs, flagCount, 0, debugLogs, remarks


def convert_bia_docx_to_xml(docx_file, html_file, save_destination):
	converter = BIAXMLConverter()
	if not str(docx_file).lower().endswith(".docx"):
		raise ValueError("docx_file must point to a .docx file")
	if not str(html_file).lower().endswith(".html"):
		raise ValueError("html_file must point to a .html file")
	if not os.path.isfile(docx_file):
		raise FileNotFoundError(f"DOCX file not found: {docx_file}")
	if not os.path.isfile(html_file):
		raise FileNotFoundError(f"HTML file not found: {html_file}")
	os.makedirs(save_destination, exist_ok=True)

	filename = os.path.basename(docx_file)
	year_match = re.search(r"(\d{4})", filename)
	year = year_match.group(1) if year_match else "0000"
	caseName, fileNum, decisionDate, typeofcases, coreParas = converter.extract_docx_content(docx_file)
	adJurisName = "Board of Immigration Appeals"
	converter.generateXML(False, filename, year, caseName, fileNum, adJurisName, decisionDate, typeofcases, [], coreParas, [], save_destination, False, False, [], [], False, False, [], [], 0, "", htmlFile=html_file)
	return os.path.join(save_destination, f"{os.path.splitext(filename)[0]}.xml")