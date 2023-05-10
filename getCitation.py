import PyPDF2
import re
import requests
import html


def get_doi(pdf_name: str):
    """This functions reads a given PDF and returns
    the DOI.
    Arge (str): filepath
    Resturns (str): DOI
    """
    # DOI regex
    pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"

    # Open the PDF file in binary mode
    with open(pdf_name, "rb") as file:

        # Read the PDF content
        pdf_reader = PyPDF2.PdfFileReader(file)
        content = ""

        # Extract text from each page
        for i in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(i)
            content += page.extractText()

    result = re.findall(pattern, content)[0]
    return result


def get_citation(doi: str):
    """
    Use python requests ti retrieve a citation in APA style using
    the DOI as input
    Args (str): DOI
    Returns (str): Citation in APA style
    """
    url = f"https://doi.org/{doi}"
    headers = {"Accept": "text/bibliography; style=apa"}

    response = requests.get(url, headers=headers)

    if response.ok:
        citation = response.text.strip()
        return citation
    else:
        print(f"Failed to retrieve citation for DOI {doi}")
