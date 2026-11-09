class FakeVectorStore:
    def search(self, query, top_k=3):
        return [
            {
                "source": "shipping_policy.md",
                "text": "Standard shipping usually takes 3 to 5 business days.",
                "similarity_score": 0.8,
            }
        ]


INCIDENT_RESPONSE_TABLE = """\
| Severity | Example | Initial response target | Escalation |
| --- | --- | --- | --- |
| Critical | Safety hazard or full service outage | Within 2 hours | Facilities lead and IT incident manager |
| High | Major impairment affecting several users | Within 4 hours | Relevant team lead |
| Medium | Limited impairment with a workaround | Within 1 business day | Assigned support owner |
| Low | Routine information or nonurgent request | Within 2 business days | Assigned support owner |
"""


class MarkdownTableStore:
    def __init__(self, chunks=None):
        self.chunks = chunks or [{
            "chunk_id": "incident_policy.md_chunk_0",
            "chunk_index": 0,
            "source": "incident_policy.md",
            "text": INCIDENT_RESPONSE_TABLE,
            "similarity_score": 1.0,
        }]

    def search(self, query, top_k=3):
        return self.chunks


class FAQStore:
    def __init__(self, text, source="faq.md"):
        self.text = text
        self.source = source

    def search(self, query, top_k=3):
        return [{
            "chunk_id": f"{self.source}_chunk_0",
            "chunk_index": 0,
            "source": self.source,
            "text": self.text,
            "similarity_score": 1.0,
        }]


def test_english_faq_returns_answer_content_without_question_marker():
    from src.rag_pipeline import SimpleRAGPipeline

    text = """\
**Q: Can contractors access company systems remotely?**
A: Contractors are not permitted to access company systems remotely.
"""
    result = SimpleRAGPipeline(FAQStore(text)).ask(
        "Can contractors access systems remotely?"
    )

    assert result["answer"] == (
        "Contractors are not permitted to access company systems remotely. "
        "Source: faq.md."
    )
    assert "Q:" not in result["answer"]
    assert "A:" not in result["answer"]
    assert "Q: Can contractors" in result["retrieved_chunks"][0]["text"]
    assert "A: Contractors are not permitted" in result["retrieved_chunks"][0]["text"]


def test_english_faq_preserves_employee_vpn_and_mfa_requirements():
    from src.rag_pipeline import SimpleRAGPipeline

    text = """\
Question: Under what conditions may employees access company systems remotely?
Answer: Employees may access company systems remotely only through the company VPN and MFA.
"""
    answer = SimpleRAGPipeline(FAQStore(text)).ask(
        "Can employees remotely access company systems?"
    )["answer"]

    assert "only through the company VPN and MFA" in answer
    assert "Question:" not in answer
    assert "Answer:" not in answer


def test_chinese_faq_returns_answer_without_repeating_question():
    from src.rag_pipeline import SimpleRAGPipeline

    text = """\
问：外包人员可以远程访问公司系统吗？
答：外包人员不得远程访问公司系统。
"""
    answer = SimpleRAGPipeline(FAQStore(text, "访问政策.md")).ask(
        "外包人员可以远程访问公司系统吗？"
    )["answer"]

    assert "外包人员不得远程访问公司系统" in answer
    assert "问：" not in answer
    assert "答：" not in answer


def test_faq_adjacent_to_markdown_table_keeps_extraction_modes_separate():
    from src.rag_pipeline import SimpleRAGPipeline

    text = INCIDENT_RESPONSE_TABLE + """

**Q: Can contractors access company systems remotely?**
A: Contractors are not permitted to access company systems remotely.
"""
    store = FAQStore(text, "operations.md")

    faq_answer = SimpleRAGPipeline(store).ask(
        "Can contractors access systems remotely?"
    )["answer"]
    table_answer = SimpleRAGPipeline(store).ask(
        "What is the Critical incident response target?"
    )["answer"]

    assert "Contractors are not permitted" in faq_answer
    assert "Q:" not in faq_answer
    assert "within 2 hours" in table_answer
    assert "Contractors" not in table_answer


def test_faq_does_not_answer_an_unsupported_question():
    from src.rag_pipeline import SimpleRAGPipeline

    text = """\
Q: Can contractors access company systems remotely?
A: Contractors are not permitted to access company systems remotely.
"""
    answer = SimpleRAGPipeline(FAQStore(text)).ask(
        "How long does standard shipping usually take?"
    )["answer"]

    assert "could not find enough information" in answer.lower()
    assert "Source:" not in answer


def test_markdown_table_extracts_critical_response_target_cleanly():
    from src.rag_pipeline import SimpleRAGPipeline

    result = SimpleRAGPipeline(MarkdownTableStore()).ask(
        "What is the Critical incident response target?"
    )

    assert result["answer"] == (
        "The initial response target for a Critical incident is within 2 hours. "
        "Source: incident_policy.md."
    )
    assert "|" not in result["answer"]
    assert "High" not in result["answer"]


def test_markdown_table_selects_high_severity_value_without_neighboring_rows():
    from src.rag_pipeline import SimpleRAGPipeline

    answer = SimpleRAGPipeline(MarkdownTableStore()).ask(
        "What is the High-severity response target?"
    )["answer"]

    assert "within 4 hours" in answer
    assert "within 2 hours" not in answer
    assert "Critical" not in answer
    assert "Medium" not in answer


def test_markdown_table_selects_requested_escalation_column():
    from src.rag_pipeline import SimpleRAGPipeline

    answer = SimpleRAGPipeline(MarkdownTableStore()).ask(
        "Who is responsible for Critical incident escalation?"
    )["answer"]

    assert "Facilities lead and IT incident manager" in answer
    assert "within 2 hours" not in answer
    assert "Source: incident_policy.md" in answer


def test_markdown_table_supports_chinese_response_target_question():
    from src.rag_pipeline import SimpleRAGPipeline

    answer = SimpleRAGPipeline(MarkdownTableStore()).ask(
        "Critical 事故的首次响应目标是什么？"
    )["answer"]

    assert "Critical 事故的首次响应目标是2 小时内" in answer
    assert "Source: incident_policy.md" in answer
    assert "|" not in answer


def test_markdown_table_reconstructs_rows_split_across_overlapping_chunks():
    from src.rag_pipeline import SimpleRAGPipeline
    from src.text_splitter import create_chunks

    chunks = create_chunks(
        [{"source": "incident_policy.md", "text": INCIDENT_RESPONSE_TABLE}],
        chunk_size=150,
        overlap=60,
    )
    retrieved = [dict(chunk, similarity_score=1.0) for chunk in chunks]

    answer = SimpleRAGPipeline(MarkdownTableStore(retrieved)).ask(
        "What is the High-severity response target?"
    )["answer"]

    assert "within 4 hours" in answer
    assert "| ---" not in answer
    assert "Relevant team lead" not in answer


def test_markdown_table_retrieval_adds_adjacent_chunks_for_a_split_row():
    from src.rag_pipeline import SimpleRAGPipeline
    from src.text_splitter import create_chunks

    document = "Incident response policy.\n\n" + INCIDENT_RESPONSE_TABLE
    chunks = create_chunks(
        [{"source": "incident_policy.md", "text": document}],
        chunk_size=150,
        overlap=60,
    )
    indexed = [dict(chunk) for chunk in chunks]
    header_chunk = next(
        chunk for chunk in indexed
        if "Initial response target" in chunk["text"]
    )

    class TopKTableStore:
        def __init__(self):
            self.chunks = indexed

        def search(self, query, top_k=3):
            return [dict(header_chunk, similarity_score=1.0)]

    result = SimpleRAGPipeline(TopKTableStore()).ask(
        "What is the High-severity response target?"
    )

    assert "within 4 hours" in result["answer"]
    assert any(
        chunk.get("retrieval_method") == "adjacent_table_context"
        for chunk in result["retrieved_chunks"]
    )


def test_markdown_table_ignores_incomplete_neighboring_row():
    from src.rag_pipeline import SimpleRAGPipeline

    truncated = INCIDENT_RESPONSE_TABLE.split("| High |", 1)[0] + "| High | Major impairment affe"
    chunks = [{
        "chunk_id": "incident_policy.md_chunk_0",
        "chunk_index": 0,
        "source": "incident_policy.md",
        "text": truncated,
        "similarity_score": 1.0,
    }]

    answer = SimpleRAGPipeline(MarkdownTableStore(chunks)).ask(
        "What is the Critical incident response target?"
    )["answer"]

    assert "within 2 hours" in answer
    assert "Major impairment affe" not in answer
    assert "|" not in answer


def test_markdown_table_refuses_unknown_row_or_missing_value():
    from src.rag_pipeline import SimpleRAGPipeline

    unsupported = SimpleRAGPipeline(MarkdownTableStore()).ask(
        "What is the Urgent incident response target?"
    )["answer"]

    missing_value_table = """\
| Severity | Initial response target | Escalation |
| --- | --- | --- |
| Critical | | Incident manager |
"""
    missing_chunks = [{
        "chunk_id": "incomplete.md_chunk_0",
        "chunk_index": 0,
        "source": "incomplete.md",
        "text": missing_value_table,
        "similarity_score": 1.0,
    }]
    missing = SimpleRAGPipeline(MarkdownTableStore(missing_chunks)).ask(
        "What is the Critical incident response target?"
    )["answer"]

    for answer in (unsupported, missing):
        assert "could not find enough information" in answer.lower()
        assert "Source:" not in answer


def test_question_terms_normalize_adjective_adverb_variations_without_word_order():
    from src.rag_pipeline import SimpleRAGPipeline

    first = SimpleRAGPipeline._evidence_question_terms(
        "Are employees allowed to access systems remotely?"
    )
    reordered = SimpleRAGPipeline._evidence_question_terms(
        "Can employees remotely access company systems?"
    )
    evidence = SimpleRAGPipeline._evidence_terms(
        "Remote access is allowed for employees who use company VPN."
    )

    assert "remote" in first
    assert "remotely" not in first
    assert {"remote", "acces", "allow", "employe"}.issubset(first & evidence)
    assert {"remote", "acces", "employe"}.issubset(reordered & evidence)


def test_extracts_two_distinct_sentences_when_both_match_the_question():
    from src.rag_pipeline import SimpleRAGPipeline

    class MultiConditionStore:
        def search(self, query, top_k=3):
            return [{
                "source": "access_policy.md",
                "text": (
                    "Remote access requires company VPN. "
                    "Remote access also requires multi-factor authentication."
                ),
                "similarity_score": 1.0,
            }]

    answer = SimpleRAGPipeline(MultiConditionStore()).ask(
        "What is required for remote access?"
    )["answer"]

    assert "requires company VPN" in answer
    assert "also requires multi-factor authentication" in answer


def test_duplicate_sentences_from_overlapping_chunks_appear_once():
    from src.rag_pipeline import SimpleRAGPipeline

    repeated = "Preventive maintenance is required every 90 calendar days."

    class OverlappingStore:
        def search(self, query, top_k=3):
            return [
                {"source": "maintenance.md", "text": repeated, "similarity_score": 1.0},
                {"source": "maintenance.md", "text": repeated, "similarity_score": 0.9},
            ]

    answer = SimpleRAGPipeline(OverlappingStore()).ask(
        "How often is preventive maintenance required?"
    )["answer"]

    assert answer.count(repeated) == 1


def test_markdown_heading_is_not_attached_to_relevant_sentence():
    from src.rag_pipeline import SimpleRAGPipeline

    class MarkdownStore:
        def search(self, query, top_k=3):
            return [{
                "source": "access_policy.md",
                "text": (
                    "## Remote access permissions\n\n"
                    "- Remote access is allowed for employees using company VPN."
                ),
                "similarity_score": 1.0,
            }]

    answer = SimpleRAGPipeline(MarkdownStore()).ask(
        "Is remote access allowed for employees?"
    )["answer"]

    assert "Remote access is allowed for employees" in answer
    assert "##" not in answer
    assert "Remote access permissions" not in answer


def test_permission_question_prefers_the_requested_subject_over_generic_policy():
    from src.rag_pipeline import SimpleRAGPipeline

    class ReturnPolicyStore:
        def search(self, query, top_k=3):
            return [{
                "source": "return_policy_zh.md",
                "text": (
                    "大多数商品可以在送达后30天内退货。\n\n"
                    "最终销售商品、礼品卡和定制商品不能退货。"
                ),
                "similarity_score": 1.0,
            }]

    answer = SimpleRAGPipeline(ReturnPolicyStore()).ask("定制商品可以退货吗？")["answer"]

    assert "定制商品不能退货" in answer
    assert "大多数商品可以" not in answer


def test_rag_pipeline_returns_answer():
    from src.rag_pipeline import SimpleRAGPipeline

    rag = SimpleRAGPipeline(FakeVectorStore(), top_k=1)
    result = rag.ask("How long does standard shipping take?")

    assert "answer" in result
    assert "sources" in result
    assert "shipping_policy.md" in result["sources"]
    assert "3 to 5 business days" in result["answer"]


def test_rag_pipeline_low_similarity_returns_safe_answer():
    from src.rag_pipeline import SimpleRAGPipeline

    class LowScoreVectorStore:
        def search(self, query, top_k=3):
            return [{"source": "unknown.md", "text": "Unrelated content.", "similarity_score": 0.01}]

    rag = SimpleRAGPipeline(LowScoreVectorStore(), top_k=1, min_similarity_score=0.25)
    result = rag.ask("Can I pay with cryptocurrency?")

    assert "could not find enough information" in result["answer"].lower()


def test_high_score_and_valid_filename_do_not_make_irrelevant_content_supported():
    from src.rag_pipeline import SimpleRAGPipeline

    class ConfidentIrrelevantVectorStore:
        def search(self, query, top_k=3):
            return [
                {
                    "source": "shipping_policy.md",
                    "text": (
                        "Purpose. This standard operating procedure describes equipment "
                        "inspection and maintenance responsibilities."
                    ),
                    "similarity_score": 1.0,
                }
            ]

    rag = SimpleRAGPipeline(ConfidentIrrelevantVectorStore(), top_k=1)
    result = rag.ask("How long does standard shipping usually take?")

    assert "could not find enough information" in result["answer"].lower()
    assert "Source:" not in result["answer"]
    assert result["retrieved_chunks"][0]["source"] == "shipping_policy.md"


def test_temporal_answer_accepts_when_phrase_without_a_numeric_duration():
    from src.rag_pipeline import SimpleRAGPipeline

    class PaymentVectorStore:
        def search(self, query, top_k=3):
            return [{
                "source": "payment_policy.md",
                "text": "Payment is charged when the order is placed.",
                "similarity_score": 1.0,
            }]

    result = SimpleRAGPipeline(PaymentVectorStore()).ask("When is payment charged?")

    assert "when the order is placed" in result["answer"]
    assert "Source: payment_policy.md" in result["answer"]


def test_temporal_answer_accepts_hyphenated_duration():
    from src.rag_pipeline import SimpleRAGPipeline

    class WarrantyVectorStore:
        def search(self, query, top_k=3):
            return [{
                "source": "warranty_policy.md",
                "text": "Most electronic products include a one-year limited warranty.",
                "similarity_score": 1.0,
            }]

    result = SimpleRAGPipeline(WarrantyVectorStore()).ask(
        "How long is the limited warranty for most electronic products?"
    )

    assert "one-year limited warranty" in result["answer"]
    assert "Source: warranty_policy.md" in result["answer"]
