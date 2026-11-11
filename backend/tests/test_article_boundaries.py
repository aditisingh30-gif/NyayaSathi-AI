"""
test_article_boundaries.py
==========================
Regression tests verifying precise article boundary chunking in pdf_service.py:
- Article 20, 21, 21A, and 22 on Page 42 of the Constitution of India.
- Proper metadata assignment (article_number, section_header, document_title, page_number).
- Ensuring Article 21's text strictly belongs to Article 21 (and does not spill into Article 22).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pdf_service import pdf_service


SAMPLE_PAGE_42_TEXT = """11 
THE CONSTITUTION OF INDIA 
(Part III.—Fundamental Rights) 
 
(6) Nothing in sub-clause (g) of the said clause shall affect the operation 
of any existing law in so far as it imposes, or prevent the State from making 
any law imposing, in the interests of the general public, reasonable restrictions 
on the exercise of the right conferred by the said sub-clause.
20. Protection in respect of conviction for offences.—(1) No person 
shall be convicted of any offence except for violation of a law in force at the 
time of the commission of the Act charged as an offence, nor be subjected to a 
penalty greater than that which might have been inflicted under the law in force 
at the time of the commission of the offence. 
(2) No person shall be prosecuted and punished for the same offence 
more than once. 
(3) No person accused of any offence shall be compelled to be a witness 
against himself. 
21. Protection of life and personal liberty.—No person shall be 
deprived of his life or personal liberty except according to procedure 
established by law. 
2[21A. Right to education.—The State shall provide free and 
compulsory education to all children of the age of six to fourteen years in such 
manner as the State may, by law, determine.] 
22. Protection against arrest and detention in certain cases.—(1) No 
person who is arrested shall be detained in custody without being informed, as 
soon as may be, of the grounds for such arrest nor shall he be denied the right 
to consult, and to be defended by, a legal practitioner of his choice. 
 
1. Subs. by the Constitution (First Amendment) Act, 1951, s. 3.
"""


def test_article_boundaries_page_42():
    doc_metadata = {
        "doc_id": "test_const_doc",
        "title": "Constitution of India",
        "num_pages": 402,
    }
    pages_data = [{
        "page_number": 42,
        "text": SAMPLE_PAGE_42_TEXT,
        "detected_sections": [],
        "primary_section": "",
        "char_count": len(SAMPLE_PAGE_42_TEXT),
        "is_schedule": False,
    }]

    chunks = pdf_service.chunk_document(doc_metadata, pages_data)

    # 1. Map chunks by article_number
    by_art = {c["article_number"]: c for c in chunks if c["article_number"]}

    assert "20" in by_art, "Article 20 chunk missing"
    assert "21" in by_art, "Article 21 chunk missing"
    assert "21a" in by_art, "Article 21A chunk missing"
    assert "22" in by_art, "Article 22 chunk missing"

    # 2. Check Article 20
    c20 = by_art["20"]
    assert c20["page_number"] == 42
    assert "conviction for offences" in c20["section_header"].lower()
    assert "violation of a law in force" in c20["text"]
    assert "deprived of his life" not in c20["text"]

    # 3. Check Article 21
    c21 = by_art["21"]
    assert c21["page_number"] == 42
    assert c21["article_number"] == "21"
    assert c21["section_header"] == "Article 21. Protection of life and personal liberty"
    assert c21["metadata"]["document_title"] == "Constitution of India"
    assert c21["metadata"]["page_number"] == 42
    assert c21["metadata"]["article_number"] == "21"

    # Verify text boundaries of Article 21:
    normalized_art21_text = " ".join(c21["text"].split())
    assert "No person shall be deprived of his life or personal liberty except according to procedure established by law" in normalized_art21_text
    # It must NOT contain Article 22 or Article 20
    assert "arrest and detention" not in c21["text"]
    assert "conviction for offences" not in c21["text"]

    # 4. Check Article 21A
    c21a = by_art["21a"]
    assert c21a["page_number"] == 42
    assert c21a["article_number"] == "21a"
    assert "Right to education" in c21a["section_header"]
    normalized_art21a_text = " ".join(c21a["text"].split())
    assert "free and compulsory education" in normalized_art21a_text
    assert "arrest and detention" not in c21a["text"]

    # 5. Check Article 22
    c22 = by_art["22"]
    assert c22["page_number"] == 42
    assert c22["article_number"] == "22"
    assert "Protection against arrest and detention" in c22["section_header"]
    assert "detained in custody without being informed" in c22["text"]
    assert "deprived of his life" not in c22["text"]
