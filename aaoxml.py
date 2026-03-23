"""AAO PDF-to-XML conversion helpers extracted from the project.

This module is intentionally standalone and does not update existing callers.
"""

import csv
import html
import io
import os
import re
import traceback
import xml.etree.ElementTree as xml

from docx import Document
from bs4 import BeautifulSoup
 

fnstyle_pattern = r"[^.*]*font-size: 7.*?}"
furtherOrderPattern = re.compile("(F+U+R+T+H+E+R+\s+O+R+D+E+R+:)")
orderPattern = re.compile("(O+R+D+E+R+:)(?!<)")
headerplate = "<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE ad:decision-grp PUBLIC \"-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML\" \"D:\\Pub383\\DTD\\DTD\\admin-interp-pubV005-0000\\admin-interp-pubV005-0000.dtd\"><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?>"
redactionMarkers = ["xx", "XX", "Xx", "xX"]


class AAOXMLConverter:
	def aaoConversion(self, filecount, file, workDirFolder):
		debugLogs = ""
		docxFilename = file
		htmlFilename = re.sub(r"\.docx$", ".html", file, flags=re.IGNORECASE)
		if not file.lower().endswith(".docx"):
			debugLogs += f"Input must be a DOCX file: {file}"
			return (filecount, "Failed", 0, "Input must be a DOCX file.", debugLogs)
		if not os.path.isfile(docxFilename):
			debugLogs += f"Missing DOCX file for {file}"
			return (filecount, "Failed", 0, "Missing DOCX file.", debugLogs)
		if not os.path.isfile(htmlFilename):
			debugLogs += f"Missing HTML file for {file}"
			return (filecount, "Failed", 0, "Missing HTML file.", debugLogs)

		filename = os.path.basename(docxFilename)
		remarks = []
		try:
			year = filename[filename.index("_") - 4:filename.index("_")]
			rawBody, rawFootnotes = self.extractText(docxFilename)
			for marker in redactionMarkers:
				rawBody = rawBody.replace(marker, '<core:emph typestyle="bf">[redacted]</core:emph>')
				rawFootnotes = rawFootnotes.replace(marker, '<core:emph typestyle="bf">[redacted]</core:emph>')
			caseName, caseDate, coreparas, typeOfCase, fileNum = self.processTikaString(rawBody, filename)
			footnoteTexts = self.flatten(rawFootnotes)
			cleanedFootnotes = self.cleanFootnotes(footnoteTexts)
			coreparas = self.continuityFix(coreparas)
			self.generatePDFXML(cleanedFootnotes, caseName, filename, year, fileNum, caseDate, typeOfCase, coreparas, workDirFolder, htmlFilename)
			status = "Successful"
		except Exception:
			status = "Failed"
			debugLogs += str(traceback.format_exc()) + "\n"
		remarks = "None." if not remarks else "Has " + ", ".join(sorted(set(remarks))) + "."
		return (filecount, status, 0, remarks, debugLogs)

	def extractText(self, docx_file):
		document = Document(docx_file)
		body_lines = []
		footnote_lines = []
		for para in document.paragraphs:
			text = para.text.strip()
			if not text:
				body_lines.append("")
				continue
			if re.match(r"^\d+[.)]?\s+", text):
				footnote_lines.append(text)
			else:
				body_lines.append(text)
		return html.escape("\n".join(body_lines), quote=False), html.escape("\n".join(footnote_lines), quote=False)

	def fix_entities(self, xml_str):
		with open("replace_patterns.csv", mode="r", encoding="utf-8-sig") as infile:
			reader = csv.reader(infile)
			next(reader)
			entities = dict((rows[0], rows[1]) for rows in reader)
		for entity, repl in entities.items():
			xml_str = xml_str.replace(entity, repl)
		return xml_str

	def processTikaString(self, tikaString, filename):
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
		typeOfCase.append("AAO Designation: " + filename[filename.index("_") + 3:len(filename) - 7])
		for line in tikaString.split("\n"):
			if line in ("", "/N"):
				count += 1
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
			elif line.lower().startswith("form ") and not plainContentStart:
				typeOfCase.append(line)
				count = 0
			elif startParse:
				if len(line) < 4 or line.replace(" ", "").isdigit():
					newPage = False
					count = 0
				elif newPage:
					coreparas.append(line)
					plainContentStart = True
					count = 0
					newPage = False
				elif count == 0:
					if line.startswith("•"):
						coreparas.append(line)
					elif coreparas:
						coreparas[-1] = coreparas[-1] + " " + line
					else:
						coreparas.append(line)
					count = 0
					plainContentStart = True
				elif count == 1:
					if newPage and coreparas:
						coreparas[-1] = coreparas[-1] + " " + line
					else:
						coreparas.append(line)
					count = 0
					plainContentStart = True
				else:
					newPage = True
					if line and line[0].islower() and coreparas:
						coreparas[-1] = coreparas[-1] + " " + line
					else:
						coreparas.append(line)
					plainContentStart = True
		return caseName, caseDate, coreparas, typeOfCase, fileNum

	def cleanFootnotes(self, footnoteTexts):
		final = []
		for footnote in footnoteTexts:
			if not final:
				final.append(footnote)
			elif final[-1].endswith((".", '.”')) or (final[-1].endswith("</core:emph>") and footnote[0].isdigit()):
				final.append(footnote)
			else:
				final[-1] += footnote
		return final

	def continuityFix(self, coreparas):
		sortedParas = []
		for count, para in enumerate(coreparas):
			para = para.strip()
			if not para:
				continue
			if count == 0:
				sortedParas.append(para)
			elif para[0].islower():
				sortedParas[-1] = sortedParas[-1] + " " + para
			else:
				sortedParas.append(para)
		return sortedParas

	def replaceLists(self, xmlString, htmlFile):
		xmlString = html.unescape(xmlString)
		with open(htmlFile, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
			for tag in soup.find_all(True):
				tag.attrs = {}
		out = soup.find_all(["ul"])
		if not out:
			return xmlString
		return xmlString

	def flag_footnotes(self, unplaced_fns, xmlFile):
		flag = "".join([f"<core:flag.para>{fn}</core:flag.para>" for fn in unplaced_fns])
		return xmlFile.replace("<ad:judgmentbody>", f"<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Correct placement for footnote not found</core:flag-subject><core:message><core:flag.para>Please check source file to check proper placement:</core:flag.para>{flag}</core:message></core:flag></core:para>")

	def add_footnotes(self, cleanFNs, xmlFile, htmlFile=None):
		with open(htmlFile, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		style_soup = soup.find_all(["style"])
		stylesheet = " ".join(str(style_soup).split())
		fn_classes = re.findall(fnstyle_pattern, stylesheet)
		fn_classes = re.compile("|".join([sub[0:3].strip() for sub in fn_classes])) if fn_classes else re.compile("^$")
		unplaced_fns = []
		for fn in cleanFNs:
			fn = fn.strip()
			fn_number = fn.split(" ", 1)[0]
			if not fn_number.isdigit():
				unplaced_fns.append(fn)
				continue
			span = soup.find("span", text=fn_number, class_=fn_classes) or soup.find("span", text=fn_number, style=re.compile("7.*"))
			if span is None:
				unplaced_fns.append(fn)
		if unplaced_fns:
			xmlFile = self.flag_footnotes(unplaced_fns, xmlFile)
		return xmlFile

	def clean_xml_spaces(self, xmlFile):
		return re.sub(r"\s+", " ", xmlFile)

	def add_emphasis_to_xml(self, xmlFile, htmlFile=None):
		with open(htmlFile, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		for emph in soup.find_all("i"):
			content = emph.string
			if not content:
				continue
			xmlFile = xmlFile.replace(content, f"<core:emph typestyle=\"it\">{content}</core:emph>", 1)
		xmlFile = re.sub(furtherOrderPattern, '<core:emph typestyle="bf">FURTHER ORDER</core:emph>:', xmlFile)
		xmlFile = re.sub(orderPattern, '<core:emph typestyle="bf">ORDER</core:emph>:', xmlFile)
		return xmlFile

	def generatePDFXML(self, cleanFNs, caseName, filename, year, fileNum, decisionDate, typeofcases, coreParas, path, htmlFile=None):
		root = xml.Element("ad:decision-grp", volnum=year)
		adjudication = xml.SubElement(root, "ad:adjudication", volnum=year)
		adjudicationInfo = xml.SubElement(adjudication, "ad:adjudication-info")
		adRegarding = xml.SubElement(adjudicationInfo, "ad:regarding")
		xml.SubElement(adjudicationInfo, "ad:filenum").text = "OFFICE: " + fileNum
		xml.SubElement(adRegarding, "core:para", indent="none").text = caseName
		adAdjudicatorInfo = xml.SubElement(adjudicationInfo, "ad:adjudicator-info")
		coreJuris = xml.SubElement(adAdjudicatorInfo, "core:juris")
		lnciJurisInfo = xml.SubElement(coreJuris, "lnci:jurisinfo")
		xml.SubElement(lnciJurisInfo, "lnci:usa")
		xml.SubElement(coreJuris, "core:juris-name").text = "Administrative Appeals Office"
		adDates = xml.SubElement(adjudicationInfo, "ad:dates")
		xml.SubElement(adDates, "ad:decisiondate").text = decisionDate.upper().replace(".", "")
		adTypeOfCase = xml.SubElement(adjudicationInfo, "ad:typeofcase")
		for caseType in typeofcases:
			xml.SubElement(adTypeOfCase, "core:generic-hd", attrib={"align": "left", "typestyle": "ro"}).text = caseType
		adContent = xml.SubElement(adjudication, "ad:content")
		adJudgments = xml.SubElement(adContent, "ad:judgments")
		adOpinion = xml.SubElement(adJudgments, "ad:opinion")
		adJudgmentBody = xml.SubElement(adOpinion, "ad:judgmentbody")
		for para in coreParas:
			match = re.match("[A-Z\s.]*", para)
			if para.startswith("Non-Precedent Decision of") or not para.strip():
				continue
			if match is not None and match[0] == para:
				xml.SubElement(adJudgmentBody, "core:generic-hd", attrib={"align": "center", "typestyle": "ro"}).text = para
			else:
				xml.SubElement(adJudgmentBody, "core:para", indent="none").text = para
		xml.SubElement(adJudgmentBody, "core:generic-hd", attrib={"align": "left", "typestyle": "ro"}).text = filename
		output = xml.ElementTree(root)
		out = io.BytesIO()
		output.write(out)
		xml_str = headerplate + out.getvalue().decode()
		xml_str = self.replaceLists(xml_str, htmlFile)
		xml_str = self.add_footnotes(cleanFNs, xml_str, htmlFile=htmlFile)
		xml_str = self.clean_xml_spaces(xml_str)
		xml_str = self.add_emphasis_to_xml(xml_str, htmlFile=htmlFile)
		xml_str = self.fix_entities(xml_str)
		with open(f"{path}\\{filename[:-4]}.xml", "w", encoding="utf-8") as xmlfile:
			xmlfile.write(xml_str)

	def flatten(self, footnoteTexts):
		return [item for item in footnoteTexts.split("\n") if len(item) > 3]


def convert_aao_docx_to_xml(docx_file, html_file, save_destination):
	converter = AAOXMLConverter()
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
	try:
		year = filename[filename.index("_") - 4:filename.index("_")]
	except ValueError:
		year = "0000"

	rawBody, rawFootnotes = converter.extractText(docx_file)
	for marker in redactionMarkers:
		rawBody = rawBody.replace(marker, '<core:emph typestyle="bf">[redacted]</core:emph>')
		rawFootnotes = rawFootnotes.replace(marker, '<core:emph typestyle="bf">[redacted]</core:emph>')
	caseName, caseDate, coreparas, typeOfCase, fileNum = converter.processTikaString(rawBody, filename)
	footnoteTexts = converter.flatten(rawFootnotes)
	cleanedFootnotes = converter.cleanFootnotes(footnoteTexts)
	coreparas = converter.continuityFix(coreparas)
	converter.generatePDFXML(cleanedFootnotes, caseName, filename, year, fileNum, caseDate, typeOfCase, coreparas, save_destination, html_file)
	return os.path.join(save_destination, f"{os.path.splitext(filename)[0]}.xml")
