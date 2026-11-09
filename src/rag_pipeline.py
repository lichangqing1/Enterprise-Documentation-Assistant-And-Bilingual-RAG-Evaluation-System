import re
from typing import Dict, List, Set, Tuple


class SimpleRAGPipeline:
    """
    Source-grounded RAG pipeline used for AI testing and evaluation.

    The pipeline is intentionally extractive so that every generated answer can
    be checked against retrieved evidence. This makes the project suitable for
    RAG regression testing, hallucination-risk analysis, and LLM evaluation
    portfolio demonstrations.
    """

    BROAD_DOMAIN_TERMS = {
        "usually", "customer", "customers", "product", "products", "company",
        "provide", "support", "account", "payment", "shipping", "return",
        "returns", "warranty", "order", "orders", "method", "methods", "item",
        "items", "policy", "business", "days", "take", "takes"
    }

    GENERIC_QUESTION_TERMS = {
        "available", "detail", "explain", "information", "long", "please",
        "standard", "take", "tell", "usually",
    }

    HIGH_SIGNAL_ABSENT_TERMS = {
        "cryptocurrency", "crypto", "bitcoin", "international", "overseas",
        "paypal", "venmo", "alipay", "wechat", "klarna", "cash", "cod"
    }

    CHINESE_HIGH_SIGNAL_ABSENT_TERMS = {
        "加密货币", "比特币", "国际配送", "国际运输", "海外配送", "海外运输",
        "支付宝", "微信支付", "货到付款"
    }

    PROMPT_INJECTION_PATTERNS = (
        r"ignore (all )?(previous|prior) instructions",
        r"system administrator",
        r"reveal hidden",
        r"hidden .*rules",
        r"internal prompt",
        r"developer instructions",
        r"print the full .*prompt",
        r"system prompt",
        r"chain[- ]?of[- ]?thought",
        r"confidential",
        r"api key",
        r"(reveal|show|print|share|leak|give me).*credit card number",
        r"customer .*email",
        r"customer .*personal",
        r"sensitive information",
        r"jailbreak",
        r"\bdan\b",
        r"do anything now",
        r"bypass .*safety",
        r"no rules",
        r"retrieval poisoning",
        r"poisoned context",
        r"malicious retrieved",
        r"override .*retrieved",
        r"ignore .*document",
        r"bypass .*verification",
        r"disable .*security",
        r"you are now",
        r"忽略所有规则",
        r"忽略.*指令",
        r"越狱",
        r"系统提示词",
        r"开发者指令",
        r"泄露.*提示词",
        r"泄露.*个人信息",
        r"客户.*邮箱",
        r"客户.*银行卡",
        r"信用卡号",
        r"泄露.*密码",
        r"绕过.*验证",
        r"绕过.*安全",
        r"中毒.*检索",
        r"恶意.*上下文",
    )

    TEMPORAL_QUESTION_PATTERNS = (
        r"\bhow (?:long|often|frequently)\b",
        r"\bwhen\b",
        r"\b(?:response|resolution) (?:time|target)\b",
        r"\b(?:frequency|duration|timeline|deadline)\b",
        r"多久",
        r"多长时间",
        r"什么时候",
        r"何时",
        r"频率",
    )

    TEMPORAL_EVIDENCE_PATTERN = re.compile(
        r"(?:\b(?:every|within|after|before|when|daily|weekly|monthly|quarterly|annually|yearly)\b"
        r"|\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)[-\s]+"
        r"(?:business\s+)?(?:minute|hour|day|week|month|year)s?\b"
        r"|(?:每|在|于)?\s*\d+(?:\.\d+)?\s*(?:个)?(?:分钟|小时|工作日|自然日|天|周|月|年)(?:内|后|前|一次)?)",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        vector_store,
        top_k: int = 3,
        min_similarity_score: float = 0.05,
        min_question_coverage: float = 0.30,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
        self.min_question_coverage = min_question_coverage

    def retrieve(self, question: str) -> List[Dict[str, object]]:
        """Retrieve relevant chunks from the vector store."""
        retrieved = self.vector_store.search(question, top_k=self.top_k)
        return self._expand_markdown_table_context(retrieved)

    @staticmethod
    def _has_markdown_table_syntax(text: str) -> bool:
        return any(
            line.strip().startswith("|") and line.count("|") >= 2
            for line in text.splitlines()
        )

    def _expand_markdown_table_context(
        self,
        retrieved_chunks: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Include adjacent indexed chunks needed to reconstruct split table rows."""
        indexed_chunks = getattr(self.vector_store, "chunks", None)
        if not isinstance(indexed_chunks, list):
            return retrieved_chunks

        chunk_lookup = {
            (str(chunk.get("source", "")), chunk.get("chunk_index")): chunk
            for chunk in indexed_chunks
            if isinstance(chunk.get("chunk_index"), int)
        }
        expanded = list(retrieved_chunks)
        seen = {
            (str(chunk.get("source", "")), chunk.get("chunk_index"))
            for chunk in expanded
        }
        table_context_queue = [
            chunk for chunk in retrieved_chunks
            if self._has_markdown_table_syntax(str(chunk.get("text", "")))
        ]
        inspected = set()
        while table_context_queue:
            chunk = table_context_queue.pop(0)
            chunk_index = chunk.get("chunk_index")
            source = str(chunk.get("source", ""))
            current_key = (source, chunk_index)
            if not isinstance(chunk_index, int) or current_key in inspected:
                continue
            inspected.add(current_key)
            for adjacent_index in (chunk_index - 1, chunk_index + 1):
                key = (source, adjacent_index)
                adjacent = chunk_lookup.get(key)
                if adjacent is None or key in seen:
                    continue
                context_chunk = dict(adjacent)
                context_chunk["similarity_score"] = 0.0
                context_chunk["retrieval_mode"] = "adjacent_table_context"
                context_chunk["retrieval_method"] = "adjacent_table_context"
                expanded.append(context_chunk)
                seen.add(key)
                if self._has_markdown_table_syntax(str(context_chunk.get("text", ""))):
                    table_context_queue.append(context_chunk)
        return expanded

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split answerable prose without returning headings or FAQ questions."""
        blocks = []
        paragraph_lines = []

        def flush_paragraph() -> None:
            if paragraph_lines:
                blocks.append(" ".join(paragraph_lines))
                paragraph_lines.clear()

        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            faq_line = re.sub(r"^[>*_`~\s]+|[*_`~\s]+$", "", line)
            faq_question = re.match(
                r"^(?:q(?:uestion)?|问(?:题)?)\s*[:：]\s*(.*)$",
                faq_line,
                flags=re.IGNORECASE,
            )
            if faq_question:
                # The full chunk is retained in retrieved evidence, but an FAQ
                # prompt is context for its answer rather than answer content.
                flush_paragraph()
                continue
            faq_answer = re.match(
                r"^(?:a(?:nswer)?|答(?:案)?)\s*[:：]\s*(.*)$",
                faq_line,
                flags=re.IGNORECASE,
            )
            if faq_answer:
                flush_paragraph()
                answer_text = faq_answer.group(1).strip()
                if answer_text:
                    paragraph_lines.append(answer_text)
                continue
            if line.startswith("|") or line.count("|") >= 2:
                # Markdown tables are handled as structured evidence. Treating
                # their rows as prose joins headers, separators, neighboring
                # rows, and character-truncated fragments into one answer.
                flush_paragraph()
                continue
            if re.match(r"^#{1,6}\s+", line):
                flush_paragraph()
                continue
            list_item = re.match(r"^(?:[-*+] |\d+[.)]\s+)(.+)$", line)
            if list_item:
                flush_paragraph()
                blocks.append(list_item.group(1).strip())
                continue
            paragraph_lines.append(line)
        flush_paragraph()

        sentences = []
        for block in blocks:
            sentences.extend(re.split(r"(?<=[.!?。！？])\s*", block))
        return [
            re.sub(r"[*_`~]", "", sentence).strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _sentence_identity(sentence: str) -> str:
        """Return a stable key for duplicate sentences from overlapping chunks."""
        without_markdown = re.sub(r"[*_`~]", "", sentence)
        return re.sub(r"\s+", " ", without_markdown).strip().lower()

    @staticmethod
    def _merge_overlapping_chunk_text(left: str, right: str) -> str:
        """Reconstruct adjacent character chunks without duplicating overlap."""
        maximum = min(len(left), len(right))
        for size in range(maximum, 7, -1):
            if left[-size:] == right[:size]:
                return left + right[size:]
        return left.rstrip() + "\n" + right.lstrip()

    @classmethod
    def _retrieved_source_segments(
        cls,
        retrieved_chunks: List[Dict[str, object]],
    ) -> List[Dict[str, str]]:
        """Rebuild only contiguous retrieved chunks for safe table parsing."""
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for chunk in retrieved_chunks:
            grouped.setdefault(str(chunk.get("source", "")), []).append(dict(chunk))

        segments = []
        for source, chunks in grouped.items():
            indexed = all(isinstance(chunk.get("chunk_index"), int) for chunk in chunks)
            if not indexed:
                segments.extend(
                    {"source": source, "text": str(chunk.get("text", ""))}
                    for chunk in chunks
                )
                continue

            ordered = sorted(chunks, key=lambda chunk: int(chunk["chunk_index"]))
            segment_text = ""
            previous_index = None
            for chunk in ordered:
                chunk_index = int(chunk["chunk_index"])
                chunk_text = str(chunk.get("text", ""))
                if previous_index is None or chunk_index != previous_index + 1:
                    if segment_text:
                        segments.append({"source": source, "text": segment_text})
                    segment_text = chunk_text
                else:
                    segment_text = cls._merge_overlapping_chunk_text(segment_text, chunk_text)
                previous_index = chunk_index
            if segment_text:
                segments.append({"source": source, "text": segment_text})
        return segments

    @staticmethod
    def _markdown_table_cells(line: str) -> List[str]:
        """Return complete Markdown table cells or an empty list for fragments."""
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            return []
        cells = [
            re.sub(r"[*_`~]", "", cell).strip()
            for cell in stripped[1:-1].split("|")
        ]
        return cells if len(cells) >= 2 else []

    @staticmethod
    def _is_markdown_table_separator(cells: List[str]) -> bool:
        return bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
            for cell in cells
        )

    @classmethod
    def _markdown_tables(
        cls,
        retrieved_chunks: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Parse complete rows from tables present in contiguous retrieved text."""
        tables = []
        for segment in cls._retrieved_source_segments(retrieved_chunks):
            lines = str(segment["text"]).splitlines()
            for position in range(len(lines) - 1):
                headers = cls._markdown_table_cells(lines[position])
                separator = cls._markdown_table_cells(lines[position + 1])
                if (
                    not headers
                    or len(separator) != len(headers)
                    or not cls._is_markdown_table_separator(separator)
                ):
                    continue

                rows = []
                row_position = position + 2
                while row_position < len(lines):
                    line = lines[row_position].strip()
                    if not line:
                        break
                    if not line.startswith("|"):
                        break
                    cells = cls._markdown_table_cells(line)
                    if len(cells) == len(headers) and all(cells):
                        rows.append(cells)
                    # A character-split fragment is ignored. Continue across it
                    # so a complete overlapping copy in the next chunk can be used.
                    row_position += 1
                if rows:
                    tables.append({
                        "source": segment["source"],
                        "headers": headers,
                        "rows": rows,
                    })
        return tables

    @classmethod
    def _table_question_terms(cls, question: str) -> Set[str]:
        """Add small bilingual concept aliases for matching table headers."""
        terms = set(cls._evidence_terms(question))
        aliases = {
            "响应": {"response"},
            "目标": {"target"},
            "首次": {"initial"},
            "升级": {"escalation"},
            "责任": {"responsibility"},
            "负责人": {"owner", "lead", "responsibility"},
            "严重级别": {"severity"},
        }
        for phrase, mapped_terms in aliases.items():
            if phrase in question:
                terms.update(mapped_terms)
        return terms

    @classmethod
    def _select_markdown_table_fact(
        cls,
        question: str,
        retrieved_chunks: List[Dict[str, object]],
    ) -> Dict[str, str] | None:
        """Select one unambiguous row/column fact supported by a complete table."""
        question_terms = cls._table_question_terms(question)
        candidates = []
        for table in cls._markdown_tables(retrieved_chunks):
            headers = list(table["headers"])
            for column_index, header in enumerate(headers):
                if column_index == 0:
                    continue
                header_terms = cls._evidence_terms(header)
                column_overlap = len(question_terms.intersection(header_terms))
                if column_overlap == 0:
                    continue
                for row in table["rows"]:
                    row_label = row[0]
                    row_terms = cls._evidence_terms(row_label)
                    row_overlap = len(question_terms.intersection(row_terms))
                    value = row[column_index].strip()
                    if row_overlap == 0 or not value:
                        continue
                    candidates.append({
                        "source": str(table["source"]),
                        "row_label": row_label,
                        "header": header,
                        "value": value,
                        "row_overlap": row_overlap,
                        "column_overlap": column_overlap,
                    })

        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (item["row_overlap"], item["column_overlap"]),
            reverse=True,
        )
        best_score = (candidates[0]["row_overlap"], candidates[0]["column_overlap"])
        best = [
            item for item in candidates
            if (item["row_overlap"], item["column_overlap"]) == best_score
        ]
        distinct_facts = {
            (item["row_label"].lower(), item["header"].lower(), item["value"].lower())
            for item in best
        }
        if len(distinct_facts) != 1:
            return None
        selected = dict(best[0])
        selected["sources"] = ", ".join(sorted({item["source"] for item in best}))
        return selected

    @staticmethod
    def _chinese_table_value(value: str) -> str:
        """Render common documented durations in Chinese without changing facts."""
        match = re.fullmatch(
            r"within\s+(\d+(?:\.\d+)?)\s+(business\s+)?(hours?|days?)",
            value.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return value
        number, business, unit = match.groups()
        if unit.lower().startswith("hour"):
            return f"{number} 小时内"
        return f"{number} 个{'工作日' if business else '自然日'}内"

    @classmethod
    def _format_markdown_table_answer(
        cls,
        question: str,
        fact: Dict[str, str],
    ) -> str:
        """Turn a selected table cell into a concise cited extractive answer."""
        row_label = fact["row_label"]
        header = fact["header"]
        value = fact["value"]
        if cls._contains_chinese(question):
            if "响应" in question and "target" in header.lower():
                sentence = f"{row_label} 事故的首次响应目标是{cls._chinese_table_value(value)}。"
            elif "升级" in question or "责任" in question or "负责人" in question:
                sentence = f"{row_label} 事故的升级责任是{value}。"
            else:
                sentence = f"{row_label} 的{header}是{value}。"
        else:
            label_match = re.search(
                rf"\b{re.escape(row_label)}\b(?:[-\s]+[A-Za-z]+)?",
                question,
                flags=re.IGNORECASE,
            )
            subject = label_match.group(0) if label_match else row_label
            article = "an" if re.match(r"[aeiou]", subject, flags=re.IGNORECASE) else "a"
            header_label = header.lower()
            if header_label == "escalation" and "responsib" in question.lower():
                header_label = "escalation responsibility"
            if value.lower().startswith(("within ", "every ", "before ", "after ")):
                value = value[0].lower() + value[1:]
            sentence = f"The {header_label} for {article} {subject} is {value}."
        return f"{sentence} Source: {fact['sources']}."

    @staticmethod
    def _permission_polarity(text: str) -> str:
        """Identify explicit permission polarity for extractive sentence filtering."""
        negative = (
            r"\bnot\s+(?:allowed|permitted|authorized)\b",
            r"\b(?:must|may|can|could|should)\s+not\b",
            r"\b(?:cannot|can't|prohibited|forbidden|disallowed)\b",
            r"(?:不允许|禁止|不得|不能|不可|不可以|未获准|无权)",
        )
        positive = (
            r"\b(?:allowed|permitted|authorized)\b",
            r"\bmay\b",
            r"(?:允许|可以|获准|有权)",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in negative):
            return "negative"
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in positive):
            return "positive"
        return "neutral"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        ascii_tokens = re.findall(r"[a-zA-Z0-9$]+", text.lower())
        chinese_tokens = []
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            chinese_tokens.append(sequence)
            for size in (2, 3):
                chinese_tokens.extend(
                    sequence[i:i + size]
                    for i in range(0, max(len(sequence) - size + 1, 0))
                )
        return ascii_tokens + chinese_tokens

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @classmethod
    def _important_words(cls, text: str) -> Set[str]:
        """Extract simple keywords from a question or evidence sentence."""
        words = cls._tokenize(text)
        stopwords = {
            "what", "when", "where", "which", "who", "how", "does", "do",
            "is", "are", "can", "could", "should", "the", "a", "an",
            "to", "for", "of", "in", "on", "with", "and", "or", "if",
            "it", "its", "their", "my", "your", "our", "they", "them", "be",
            "by", "after", "before", "from", "as", "using", "use", "about"
        }
        return {word for word in words if word not in stopwords and len(word) > 2}

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        """Very small normalization helper for rule-based matching."""
        token = token.lower().strip()
        non_adverb_ly_words = {
            "apply", "assembly", "costly", "daily", "early", "family",
            "friendly", "likely", "lively", "lovely", "monthly", "only",
            "reply", "supply", "weekly", "yearly",
        }
        if token.endswith("ily") and len(token) > 6 and token not in non_adverb_ly_words:
            return token[:-3] + "y"
        if token.endswith("ly") and len(token) > 5 and token not in non_adverb_ly_words:
            return token[:-2]
        if token.endswith("ied") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ged") and len(token) > 4:
            return token[:-1]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    @classmethod
    def _normalized_word_set(cls, text: str) -> Set[str]:
        return {cls._normalize_token(token) for token in cls._important_words(text)}

    @classmethod
    def _evidence_terms(cls, text: str) -> Set[str]:
        """Return comparable evidence concepts for English or Chinese text."""
        # Inline Markdown is presentation syntax, not a concept boundary. This
        # matters for Chinese text such as 员工**可以**, where the markers would
        # otherwise break subject/action bigrams used for relevance ranking.
        text = re.sub(r"[*_`~]", "", text)
        if cls._contains_chinese(text):
            # Chinese has no whitespace word boundaries. The general tokenizer
            # emits a full phrase plus bigrams and trigrams, which inflates the
            # denominator and makes a supported paraphrase look unrelated.
            # Use stable two-character concepts for evidence coverage while
            # retaining ASCII terms such as product names or numbers.
            chinese_bigrams = {
                sequence[index:index + 2]
                for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text)
                for index in range(len(sequence) - 1)
            }
            ascii_terms = {
                cls._normalize_token(token)
                for token in re.findall(r"[a-zA-Z0-9$]+", text.lower())
                if len(token) > 2
            }
            return chinese_bigrams | ascii_terms
        return cls._normalized_word_set(text)

    @classmethod
    def _evidence_question_terms(cls, question: str) -> Set[str]:
        """Return question concepts that retrieved evidence must actually support."""
        question_for_matching = question
        if cls._is_broad_policy_question(question):
            if cls._contains_chinese(question):
                question_for_matching = re.sub(
                    r"(?:哪些|什么|适用|安全|要求|政策|规定|规则|控制|限制|权限|标准|指南|概述|总结)",
                    "",
                    question,
                )
            else:
                question_for_matching = re.sub(
                    r"\b(?:security|requirements?|apply|applies|applicable|polic(?:y|ies)|"
                    r"rules?|controls?|guidelines?|restrictions?|permissions?|standards?|summary)\b",
                    " ",
                    question,
                    flags=re.IGNORECASE,
                )
        normalized_generic_terms = {
            cls._normalize_token(term)
            for term in cls.GENERIC_QUESTION_TERMS
        }
        return cls._evidence_terms(question_for_matching) - normalized_generic_terms

    @staticmethod
    def _is_broad_policy_question(question: str) -> bool:
        """Identify questions asking for a policy-level summary or requirements."""
        return bool(re.search(
            r"\b(?:requirements?|polic(?:y|ies)|rules?|controls?|guidelines?|"
            r"restrictions?|permissions?|standards?|summary)\b|"
            r"(?:要求|政策|规定|规则|控制|限制|权限|标准|指南|概述|总结)",
            question,
            flags=re.IGNORECASE,
        ))

    @classmethod
    def _question_subject_terms(cls, question: str) -> Set[str]:
        """Extract a narrow subject anchor for permission-style questions."""
        if cls._contains_chinese(question):
            match = re.match(r"^(.+?)(?:可以|能否|是否|能不能|会不会)", question.strip())
            return cls._evidence_terms(match.group(1)) if match else set()

        question_lower = question.lower().strip()
        auxiliary_match = re.match(
            r"^(?:are|is|can|could|do|does|will|would|should)\s+(.+?)\s+"
            r"(?:allowed|permitted|authorized|be|access)\b",
            question_lower,
        )
        if auxiliary_match:
            return cls._evidence_terms(auxiliary_match.group(1))
        if (
            re.search(r"\b(?:allowed|permitted|authorized|access)\b", question_lower)
            or cls._is_broad_policy_question(question)
        ):
            trailing_match = re.search(r"\bfor\s+([a-z][a-z\s-]{0,40})\??$", question_lower)
            if trailing_match:
                return cls._evidence_terms(trailing_match.group(1))
        return set()

    @staticmethod
    def _numeric_terms(text: str) -> Set[str]:
        return set(re.findall(r"\b\d+(?:\.\d+)?\b|\$\d+(?:\.\d+)?", text.lower()))

    @staticmethod
    def _day_constraints(text: str) -> Set[str]:
        english_days = re.findall(r"\b\d+(?:\.\d+)?\s+(?:business\s+)?days\b", text.lower())
        chinese_days = re.findall(r"\d+(?:\.\d+)?\s*(?:个)?(?:工作日|天|日)", text)
        return set(english_days + chinese_days)

    @staticmethod
    def _year_constraints(text: str) -> Set[str]:
        text_lower = text.lower()
        digit_years = re.findall(r"\b\d+(?:\.\d+)?[-\s]+year\b", text_lower)
        word_numbers = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
        }
        word_years = []
        for word, digit in word_numbers.items():
            if re.search(rf"\b{word}[-\s]+year\b", text_lower):
                word_years.append(f"{digit} year")
        chinese_years = re.findall(r"\d+(?:\.\d+)?\s*年", text)
        return set(digit_years + word_years + chinese_years)

    @classmethod
    def _context_text(cls, retrieved_chunks: List[Dict[str, object]]) -> str:
        return "\n".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)

    @classmethod
    def _contains_prompt_injection(cls, question: str) -> bool:
        question_norm = question.lower()
        return any(re.search(pattern, question_norm) for pattern in cls.PROMPT_INJECTION_PATTERNS)

    @classmethod
    def _expects_temporal_answer(cls, question: str) -> bool:
        return any(
            re.search(pattern, question, flags=re.IGNORECASE)
            for pattern in cls.TEMPORAL_QUESTION_PATTERNS
        )

    @classmethod
    def _contains_temporal_evidence(cls, text: str) -> bool:
        return bool(cls.TEMPORAL_EVIDENCE_PATTERN.search(text))

    @classmethod
    def _missing_context_terms(
        cls,
        question: str,
        retrieved_chunks: List[Dict[str, object]],
    ) -> Tuple[Set[str], float]:
        question_terms = cls._evidence_question_terms(question)
        context_terms = cls._evidence_terms(cls._context_text(retrieved_chunks))
        if not question_terms:
            return set(), 1.0
        covered_terms = question_terms.intersection(context_terms)
        coverage = len(covered_terms) / len(question_terms)
        missing_terms = question_terms - context_terms
        return missing_terms, coverage

    def _should_refuse_as_unanswerable(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, object]],
    ) -> Tuple[bool, str]:
        """
        Decide whether the retrieved evidence is insufficient for a safe answer.

        This guard is deliberately conservative for RAG evaluation. It prevents
        the system from answering questions that contain specific entities,
        payment methods, countries/shipping scope, or numeric constraints that
        are absent from the retrieved context.
        """
        if not retrieved_chunks:
            return True, "no retrieved evidence"

        context = self._context_text(retrieved_chunks).lower()
        question_lower = question.lower()
        missing_terms, coverage = self._missing_context_terms(question, retrieved_chunks)
        is_chinese_question = self._contains_chinese(question)
        question_days = sorted(self._day_constraints(question_lower))
        question_years = sorted(self._year_constraints(question))

        if self._contains_prompt_injection(question):
            if is_chinese_question:
                return True, "问题包含提示词注入或越权指令，不能泄露系统提示词或编造政策"
            return True, "the question contains prompt-injection or instruction-override content"

        if is_chinese_question and "退货" in question and question_days:
            missing_day_constraints = [day for day in question_days if day not in context]
            if missing_day_constraints:
                mentioned_days = sorted(self._day_constraints(context))
                mentioned_text = f" 文档只提到{mentioned_days[0]}。" if mentioned_days else ""
                return (
                    True,
                    f"提供的文档没有提供{missing_day_constraints[0]}后退货的依据。"
                    f"{mentioned_text}我无法确认{missing_day_constraints[0]}后可以退货",
                )

        for term in self.CHINESE_HIGH_SIGNAL_ABSENT_TERMS:
            if term in question and term not in context:
                if term in {"国际配送", "国际运输", "海外配送", "海外运输"}:
                    return True, "提供的文档没有提到国际配送"
                if term in {"加密货币", "比特币"}:
                    return True, "提供的文档没有提到加密货币作为可接受的付款方式"
                if term in {"支付宝", "微信支付"}:
                    return True, f"提供的文档没有提到{term}作为可接受的付款方式"
                return True, f"提供的文档没有提到{term}"

        # Missing numeric constraints are high risk. Example: "after 90 days"
        # should not be answered using a generic 30-day return sentence.
        question_numbers = self._numeric_terms(question_lower)
        context_numbers = self._numeric_terms(context)
        missing_numbers = question_numbers - context_numbers
        if missing_numbers:
            missing_number_text = ", ".join(sorted(missing_numbers))
            mentioned_days = sorted(self._day_constraints(context))
            asks_about_return = "return" in question_lower or "退货" in question
            if asks_about_return and question_days:
                mentioned_text = f" The documents only mention {mentioned_days[0]}." if mentioned_days else ""
                if is_chinese_question:
                    mentioned_text = f" 文档只提到{mentioned_days[0]}。" if mentioned_days else ""
                    return (
                        True,
                        f"提供的文档没有提供{question_days[0]}后退货的依据。"
                        f"{mentioned_text}我无法确认{question_days[0]}后可以退货",
                    )
                return (
                    True,
                    "the provided documents do not provide support for returning products "
                    f"after {question_days[0]}.{mentioned_text} I cannot confirm returns "
                    f"after {question_days[0]}",
                )
            return True, f"numeric constraint not found in evidence: {missing_number_text}"

        if question_years and ("warranty" in question_lower or "保修" in question):
            context_years = self._year_constraints(context)
            missing_years = [year for year in question_years if year not in context_years]
            if missing_years:
                if is_chinese_question:
                    return True, f"提供的文档没有提到{missing_years[0]}保修"
                return True, f"the provided documents do not mention a {missing_years[0]} warranty"

        high_signal_missing = {
            term for term in missing_terms
            if term in self.HIGH_SIGNAL_ABSENT_TERMS
        }
        if high_signal_missing:
            if "international" in high_signal_missing and "shipping" in question_lower:
                return True, "the provided documents do not mention international shipping"
            if {"cryptocurrency", "crypto", "bitcoin"}.intersection(high_signal_missing):
                return (
                    True,
                    "the provided documents do not mention cryptocurrency as an accepted "
                    "payment method",
                )
            return True, f"specific term not found in evidence: {', '.join(sorted(high_signal_missing))}"

        specific_missing_terms = {
            term for term in missing_terms
            if len(term) >= 8 and term not in self.BROAD_DOMAIN_TERMS
        }
        yes_no_question = bool(re.match(r"\s*(can|does|do|is|are|will|would|could|should)\b", question_lower))
        yes_no_question = yes_no_question or bool(re.match(r"\s*(可以|能否|是否|会不会|能不能)", question))
        if yes_no_question and specific_missing_terms:
            return True, f"specific question term not found in evidence: {', '.join(sorted(specific_missing_terms))}"

        if coverage < self.min_question_coverage:
            return True, f"low question-evidence coverage: {coverage:.2f}"

        return False, "evidence appears sufficient"

    @staticmethod
    def _safe_no_answer(reason: str) -> str:
        return (
            "I could not find enough information in the provided documents to answer this safely. "
            f"Reason: {reason}."
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, object]]
    ) -> str:
        """
        Generate a simple grounded answer from retrieved chunks.

        The answer is extractive and contains a source citation. If the evidence
        is not sufficient, the method returns a safe refusal instead of guessing.
        """
        if not retrieved_chunks:
            return self._safe_no_answer("no retrieved evidence")

        table_fact = self._select_markdown_table_fact(question, retrieved_chunks)
        should_refuse, reason = self._should_refuse_as_unanswerable(question, retrieved_chunks)
        table_fact_resolves_cross_language_coverage = (
            table_fact is not None
            and reason.startswith("low question-evidence coverage")
        )
        if should_refuse and not table_fact_resolves_cross_language_coverage:
            return self._safe_no_answer(reason)

        best_score = retrieved_chunks[0].get("similarity_score", 0)
        if best_score < self.min_similarity_score:
            return self._safe_no_answer("retrieval similarity below threshold")

        if table_fact is not None:
            return self._format_markdown_table_answer(question, table_fact)

        question_words = self._evidence_question_terms(question)
        question_subject_terms = self._question_subject_terms(question)
        broad_policy_summary = (
            self._is_broad_policy_question(question)
            and not question_subject_terms
        )
        if not question_words:
            return self._safe_no_answer("the question does not contain a specific evidence concept")
        expects_temporal_answer = self._expects_temporal_answer(question)
        candidate_sentences = []

        for chunk in retrieved_chunks:
            source = chunk["source"]
            for sentence in self._split_sentences(chunk["text"]):
                sentence_words = self._evidence_terms(sentence)
                overlap = len(question_words.intersection(sentence_words))
                evidence_coverage = overlap / len(question_words)
                candidate_sentences.append({
                    "sentence": sentence,
                    "source": source,
                    "matched_question_terms": question_words.intersection(sentence_words),
                    "subject_overlap": len(question_subject_terms.intersection(sentence_words)),
                    "overlap": overlap,
                    "evidence_coverage": evidence_coverage,
                    "answer_type_supported": (
                        not expects_temporal_answer
                        or self._contains_temporal_evidence(sentence)
                    ),
                    "similarity_score": chunk.get("similarity_score", 0)
                })

        if not candidate_sentences:
            return self._safe_no_answer("no usable evidence sentences")

        candidate_sentences = sorted(
            candidate_sentences,
            key=lambda x: (
                x["subject_overlap"],
                x["evidence_coverage"],
                x["overlap"],
                x["similarity_score"],
            ),
            reverse=True
        )

        deduplicated_candidates = []
        seen_sentences = set()
        for item in candidate_sentences:
            identity = self._sentence_identity(item["sentence"])
            if identity in seen_sentences:
                continue
            seen_sentences.add(identity)
            deduplicated_candidates.append(item)

        useful_sentences = [
            item for item in deduplicated_candidates
            if item["overlap"] > 0
            and item["evidence_coverage"] >= self.min_question_coverage
            and item["answer_type_supported"]
        ]
        if not useful_sentences:
            return self._safe_no_answer("no retrieved sentence sufficiently supports the question")

        # Always use the strongest sentence. Add a second only when it covers a
        # new question concept, matches the primary intent equally, or is an
        # explicit continuation. A lower-ranked strict subset is contextual
        # evidence, not part of the direct answer.
        selected = [useful_sentences[0]]
        primary_terms = set(useful_sentences[0]["matched_question_terms"])
        covered_terms = set(primary_terms)
        primary_polarity = self._permission_polarity(useful_sentences[0]["sentence"])
        continuation_pattern = re.compile(
            r"^(?:also|additionally|in addition|furthermore|moreover|"
            r"并且|此外|另外|还)",
            flags=re.IGNORECASE,
        )
        for item in useful_sentences[1:]:
            matched_terms = set(item["matched_question_terms"])
            contributes_new_concept = bool(matched_terms - covered_terms)
            matches_primary_intent = matched_terms == primary_terms
            continues_primary = bool(continuation_pattern.match(item["sentence"]))
            candidate_polarity = self._permission_polarity(item["sentence"])
            reverses_primary = (
                not broad_policy_summary
                and
                primary_polarity in {"positive", "negative"}
                and candidate_polarity in {"positive", "negative"}
                and primary_polarity != candidate_polarity
            )
            if not reverses_primary and (
                contributes_new_concept or matches_primary_intent or continues_primary
            ):
                selected.append(item)
                covered_terms.update(matched_terms)
                break

        answer_parts = [item["sentence"] for item in selected]
        sources = sorted({item["source"] for item in selected})
        return " ".join(answer_parts) + f" Source: {', '.join(sources)}."

    def ask(self, question: str) -> Dict[str, object]:
        """Run retrieval and answer generation."""
        retrieved_chunks = self.retrieve(question)
        answer = self.generate_answer(question, retrieved_chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_context": self._context_text(retrieved_chunks),
            "sources": [chunk["source"] for chunk in retrieved_chunks]
        }
