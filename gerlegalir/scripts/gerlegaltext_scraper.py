import requests
import zipfile
from io import BytesIO
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd


def iterate_tags(tag):
    """
    Recursively iterate through XML tags and extract text content.

    Args:
        tag (ET.Element): The XML element to iterate through.

    Returns:
        str: Concatenated text content with some characters replaced.
    """
    text = ''
    for child in tag:
        text += "".join(t.replace('U+00A0', "").replace('"', "")
                        for t in child.itertext())
    return text.replace('\xa0', '')


def download_and_extract(link):
    """
    Download and extract XML file from a given link.

    Args:
        link (str): URL of the zip file containing XML.

    Returns:
        str or None: Decoded XML content if successful, None otherwise.
    """
    response = requests.get(link)

    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            xml_files = [file for file in z.namelist()
                         if file.lower().endswith('.xml')]
            if xml_files:
                xml_file = xml_files[0]
            else:
                tqdm.write("No XML file found in the zip.")
                return None
            with z.open(xml_file) as xml_content:
                return xml_content.read().decode('utf-8')
    else:
        tqdm.write(
            f"Failed to download from link: {link}. "
            f"Status code: {response.status_code}"
        )
        return None


def scrape_content():
    """
    Fetch XML data from the main URL and scrape legal text content.

    Returns:
        list: List of dictionaries containing scraped legal text data.
    """
    entries = []
    url = "https://www.gesetze-im-internet.de/gii-toc.xml"
    response = requests.get(url)

    if response.status_code == 200:
        root = ET.fromstring(response.content)

        items = root.findall('.//item')
        with tqdm(range(len(items))) as pbar:
            pbar.set_description("Scraping Legal Texts")
            for i in pbar:
                item = items[i]
                document_link = item.find('link').text
                mongodb_links = []  # This seems unused, consider removing
                if document_link not in mongodb_links:
                    xml_content = download_and_extract(document_link)

                    if xml_content:
                        xml_root = ET.fromstring(xml_content)
                        entry = process_xml_content(xml_root)
                        if entry:
                            entries.append(entry)

        return entries
    else:
        print(f"Failed to retrieve XML data. Status code: {response.status_code}")
        return []


def process_xml_content(xml_root):
    """
    Process the XML content and extract relevant information.

    Args:
        xml_root (ET.Element): Root element of the XML content.

    Returns:
        dict or None: Dictionary containing extracted information if successful,
                      None otherwise.
    """
    jurabk = xml_root.find('metadaten/jurabk').text
    date = xml_root.find('metadaten/ausfertigung-datum').text
    periodikum = xml_root.find('metadaten/fundstelle/periodikum').text
    zitstelle = xml_root.find('metadaten/fundstelle/zitstelle').text
    langue = xml_root.find('metadaten/langue').text
    gliederungsbezeichnung = xml_root.find('metadaten/gliederungseinheit/gliederungsbez').text
    paragraph = xml_root.find('metadaten/enbez').text
    jura_text = iterate_tags(xml_root.find('textdaten').findall('.//P'))

    return {
        'name': jurabk,
        'date': date,
        'periodikum': periodikum,
        'zitstelle': zitstelle,
        'langue': langue,
        'gliederungsbezeichnung': gliederungsbezeichnung,
        'paragraph': paragraph,
        'text': jura_text,
    }


if __name__ == "__main__":
    entries = scrape_content()
    legal_df = pd.DataFrame(entries)
    legal_df.to_json('GerLegalText.json', orient='records')