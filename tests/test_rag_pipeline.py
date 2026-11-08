class FakeVectorStore:
    def search(self, query, top_k=3):
        return [
            {
                "source": "shipping_policy.md",
                "text": "Standard shipping usually takes 3 to 5 business days.",
                "similarity_score": 0.8,
            }
        ]


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
