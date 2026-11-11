import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.config import settings, SAMPLE_DIR
from app.services.pdf_service import pdf_service, PDFProcessingService
from app.services.vector_service import vector_service

SAMPLE_LEGAL_DOCS = [
    {
        "doc_id": "act_dpdp_2023",
        "title": "Digital Personal Data Protection Act, 2023",
        "category": "Statutory Act",
        "filename": "Digital_Personal_Data_Protection_Act_2023.pdf",
        "sections": [
            (
                "Section 1. Short Title and Application",
                "This Act may be called the Digital Personal Data Protection Act, 2023. "
                "It applies to the processing of digital personal data within India where such data is collected in digital form or digitised subsequently. "
                "It also applies to processing outside India if such processing is in connection with offering goods or services to Data Principals in India."
            ),
            (
                "Section 4. Grounds for Processing Personal Data",
                "A person may process the personal data of a Data Principal only in accordance with the provisions of this Act "
                "and for a lawful purpose for which the Data Principal has given consent, or for certain legitimate uses specified in Section 7."
            ),
            (
                "Section 5. Notice for Consent",
                "Every request for consent shall be preceded or accompanied by a notice informing the Data Principal of: "
                "(a) the personal data sought and the specified purpose of processing; "
                "(b) the manner in which the Data Principal may exercise rights under Section 11 and make a complaint to the Data Protection Board."
            ),
            (
                "Section 8. General Obligations of Data Fiduciary",
                "A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards "
                "to prevent personal data breach. In the event of a personal data breach, the Data Fiduciary shall give the Board and each affected Data Principal intimation of such breach."
            ),
            (
                "Section 9. Processing of Personal Data of Children",
                "A Data Fiduciary shall, before processing any personal data of a child, obtain verifiable parental consent. "
                "A Data Fiduciary shall not undertake tracking or behavioural monitoring of children or targeted advertising directed at children."
            ),
            (
                "Section 33. Penalties for Breach",
                "If the Board determines non-compliance, it may impose financial penalties: "
                "(a) up to two hundred and fifty crore rupees (INR 250 Crores) for failure to take reasonable security safeguards to prevent data breach; "
                "(b) up to two hundred crore rupees for failure to notify data breach."
            )
        ]
    },
    {
        "doc_id": "act_contract_1872",
        "title": "Indian Contract Act, 1872 (Essential Provisions)",
        "category": "Commercial Statute",
        "filename": "Indian_Contract_Act_1872.pdf",
        "sections": [
            (
                "Section 10. What Agreements are Contracts",
                "All agreements are contracts if they are made by the free consent of parties competent to contract, "
                "for a lawful consideration and with a lawful object, and are not hereby expressly declared to be void."
            ),
            (
                "Section 27. Agreement in Restraint of Trade Void",
                "Every agreement by which anyone is restrained from exercising a lawful profession, trade or business of any kind, "
                "is to that extent void. Exception: One who sells the goodwill of a business may agree with the buyer to refrain from carrying on a similar business within specified local limits."
            ),
            (
                "Section 73. Compensation for Loss Caused by Breach",
                "When a contract has been broken, the party who suffers by such breach is entitled to receive, from the party who has broken the contract, "
                "compensation for any loss or damage caused to him thereby, which naturally arose in the usual course of things from such breach."
            ),
            (
                "Section 74. Compensation for Breach where Penalty Stipulated",
                "When a contract has been broken, if a sum is named in the contract as the amount to be paid in case of breach, "
                "the party complaining of the breach is entitled to receive reasonable compensation not exceeding the amount so named or penalty stipulated."
            ),
            (
                "Section 124. Contract of Indemnity Defined",
                "A contract by which one party promises to save the other from loss caused to him by the conduct of the promisor himself, "
                "or by the conduct of any other person, is called a contract of indemnity."
            )
        ]
    },
    {
        "doc_id": "bns_crimes_2023",
        "title": "Bharatiya Nyaya Sanhita (BNS) 2023 - Commercial & Cyber Offences",
        "category": "Criminal Statute",
        "filename": "Bharatiya_Nyaya_Sanhita_2023.pdf",
        "sections": [
            (
                "Section 316. Criminal Breach of Trust",
                "Whoever, being entrusted with property or with dominion over property, dishonestly misappropriates or converts to his own use that property, "
                "or dishonestly uses or disposes of that property in violation of any direction of law, commits criminal breach of trust."
            ),
            (
                "Section 318. Cheating and Dishonestly Inducing Delivery",
                "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property, "
                "or to consent that any person shall retain any property, commits cheating and shall be punished with imprisonment up to seven years and fine."
            ),
            (
                "Section 336. Forgery of Electronic Records",
                "A person is said to make a false document or electronic record who dishonestly or fraudulently makes, signs, seals, executes, or transmits "
                "any electronic record with the intention of causing it to be believed that such record was made by or by authority of a person by whom he knows it was not made."
            )
        ]
    },
    {
        "doc_id": "contract_saas_msa",
        "title": "Standard Enterprise Cloud & AI Master Services Agreement",
        "category": "Commercial Contract",
        "filename": "Enterprise_Cloud_AI_MSA.pdf",
        "sections": [
            (
                "Clause 2. Service Level Agreement and Availability",
                "Vendor warrants that the cloud software services shall achieve a monthly uptime percentage of at least 99.9%. "
                "In the event of failure to meet the SLA, Customer's sole remedy shall be service credits applied to subsequent billing cycles."
            ),
            (
                "Clause 6. Data Protection and Security Standards",
                "Vendor agrees to maintain industry-standard administrative, physical, and technical safeguards. "
                "Vendor shall process Customer personal data strictly in compliance with applicable data protection laws including the Digital Personal Data Protection Act, 2023."
            ),
            (
                "Clause 9. Limitation of Liability and Super Cap",
                "Neither party shall be liable for indirect, incidental, or consequential damages. "
                "Each party's maximum aggregate liability shall be limited to the fees paid in the twelve (12) months preceding the incident, "
                "except that for breaches of confidentiality or gross negligence, liability shall be subject to a super-cap of two times (2x) the annual contract value."
            ),
            (
                "Clause 11. Termination for Convenience and Cause",
                "Either party may terminate this Agreement for cause if the other party materially breaches and fails to cure such breach within thirty (30) days of notice. "
                "Customer may terminate for convenience upon sixty (60) days advance written notice."
            )
        ]
    }
]


def preload_sample_legal_corpus(force: bool = False) -> int:
    """
    Generates benchmark legal PDFs in sample_data and indexes them into ChromaDB.
    Runs idempotently on backend startup.
    """
    total_indexed = 0
    existing_docs = {d["doc_id"]: d for d in vector_service.list_documents()}

    for item in SAMPLE_LEGAL_DOCS:
        doc_id = item["doc_id"]
        if not force and doc_id in existing_docs:
            continue

        pdf_path = str(SAMPLE_DIR / item["filename"])
        # Generate genuine PDF with PyMuPDF
        PDFProcessingService.generate_legal_pdf(
            title=item["title"],
            sections=item["sections"],
            output_path=pdf_path
        )

        # Extract and chunk
        meta, pages = pdf_service.extract_document(
            file_source=pdf_path,
            filename=item["filename"],
            doc_id=doc_id
        )
        meta["category"] = item["category"]
        meta["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
        meta["summary"] = f"Statutory & Benchmark reference for {item['title']}."

        chunks = pdf_service.chunk_document(meta, pages)
        vector_service.add_document_chunks(meta, chunks)
        total_indexed += 1

    return total_indexed
