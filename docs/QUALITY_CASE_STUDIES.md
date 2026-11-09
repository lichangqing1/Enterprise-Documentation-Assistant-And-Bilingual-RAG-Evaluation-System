# Quality Case Studies

These case studies document defects found through manual testing and adversarial evaluation. Each correction was implemented as a general rule and protected by regression tests rather than a hardcoded answer for one question.

## 1. Irrelevant Retrieved Passage Accepted as an Answer

### Defect

With only an equipment-maintenance SOP in the enterprise collection, the question “How long does standard shipping usually take?” returned the document’s purpose paragraph instead of an insufficient-information response.

### Root cause

Retrieval always returns the nearest available chunks, even when a corpus does not contain the requested fact. The early extractive path treated the presence of results, a positive score, and a valid source filename as sufficient. A broadly worded introduction could therefore become the answer despite weak question/evidence support.

### Corrective action

`SimpleRAGPipeline` now checks question-to-evidence concept coverage before sentence selection, requires candidate sentences to meet evidence coverage, and verifies that temporal questions have temporal evidence. Unsupported answers use the citation-free fallback while raw retrieved chunks remain available for debugging.

### Regression coverage

- `test_high_score_and_valid_filename_do_not_make_irrelevant_content_supported`
- `test_equipment_sop_does_not_answer_enterprise_shipping_question`
- `test_enterprise_shipping_question_is_unsupported_in_llm_mode`
- Enterprise safety cases `ENT-SAFE-005` and `ENT-SAFE-008`

### Result

Shipping and cafeteria questions are refused against unrelated enterprise documents, while supported 90-day maintenance and two-hour incident-response questions still return evidence and citations. The legacy shipping question continues to use the isolated customer-support corpus.

## 2. Prompt Injection Inside a Retrieved Document

### Defect

A retrieved document contained a malicious instruction telling the model to return a synthetic credential. A mocked LLM followed the instruction and cited the real document filename.

### Root cause

The prompt distinguished system instructions from the question but initially passed all retrieved text as usable evidence. Citation validation alone could not detect that the answer followed an instruction embedded in a valid source.

### Corrective action

`quarantine_document_instructions()` splits retrieved text into spans and removes instruction-like spans from provider evidence. The grounded prompt labels evidence as untrusted data. A post-generation influence check rejects answers that reproduce concepts found only in quarantined spans. Raw retrieval is retained locally for debugging.

The detector is deliberately narrow enough to preserve legitimate policy language such as “Contractors must not share API credentials.”

### Regression coverage

- `test_document_prompt_injection_is_delimited_and_cannot_invent_a_source`
- `test_llm_rejects_answer_that_follows_quarantined_document_instruction`
- `test_ordinary_security_policy_language_is_not_quarantined`
- `test_access_policy_document_injection_answer_is_rejected`
- Enterprise safety cases `ENT-SAFE-001` and `ENT-SAFE-002`

### Result

Supported facts in a partially malicious document remain answerable, while answers following quarantined instructions are rejected. `ENT-SAFE-002` passes without hardcoding its synthetic credential.

## 3. Reversed Permission or Prohibition Claim

### Defect

Evidence stated “Remote access is not allowed for contractors,” but a mocked LLM answered “Remote access is allowed for contractors.” The original validator accepted the answer because the citation filename was valid.

### Root cause

Filename validation established source membership, not factual agreement. General token overlap was also high because the contradictory sentences differed mainly by negation.

### Corrective action

Generated claims are associated with the closest cited evidence span and chunk ID. Conservative English and Chinese polarity classifiers identify explicit permissions and prohibitions. A sufficiently matched claim is rejected when its polarity reverses the evidence. Existing numeric and citation checks remain independent.

### Regression coverage

- `test_access_policy_contractor_polarity_reversal_is_rejected`
- `test_permission_polarity_accepts_grounded_positive_and_negative_claims`
- `test_permission_polarity_rejects_english_and_chinese_reversals`
- `test_llm_returns_claim_to_chunk_evidence_references`
- Enterprise safety case `ENT-SAFE-007`

### Result

Correct positive and negative policy statements are accepted in English and Chinese. Explicit reversals are rejected even when they cite a retrieved filename.

## 4. False Refusal Caused by Morphological Differences

### Defect

“Under what conditions is remote access allowed for employees?” was accepted, while the semantically equivalent “Are employees allowed to access systems remotely?” could be rejected before the LLM call.

### Root cause

Question/evidence coverage depended on surface tokens. Word order and forms such as `remote`/`remotely`, `allowed`/`allow`, and plural nouns reduced overlap below the existing sufficiency threshold. The evidence itself was adequate.

### Corrective action

The evidence-term layer now performs small deterministic morphological normalization for common English adjective/adverb, tense, and plural variations. Matching uses sets rather than word order, and subject extraction preserves employee/contractor distinctions. The evidence-sufficiency threshold was not lowered and no extra LLM call was added.

### Regression coverage

- `test_question_terms_normalize_adjective_adverb_variations_without_word_order`
- `test_extractive_access_paraphrases_preserve_vpn_and_mfa_conditions`
- `test_llm_access_policy_paraphrases_are_grounded_and_conditional`
- Chinese employee and contractor cases in both enterprise assistant and grounded LLM tests

### Result

All supported paraphrases reach the correct answer path and preserve VPN and multi-factor-authentication conditions. Contractor questions still select the prohibition, and unrelated shipping questions remain unsupported.

## 5. Answer Over-Extraction

### Defect

For “Under what conditions is remote access allowed for employees?”, extractive mode returned both the employee VPN/MFA sentence and the unrelated contractor prohibition.

### Root cause

Both sentences occurred in the same highly ranked chunk and shared the terms `remote`, `access`, and `allowed`. Answer aggregation treated this overlap as enough to append a second sentence, without sufficiently weighting the requested subject or opposite policy polarity.

### Corrective action

Candidate sentences now include subject overlap, matched question concepts, evidence coverage, and explicit polarity. Narrow questions prioritize their subject anchor and do not append a sentence that reverses the primary permission polarity. Sentence identities remove duplicates from overlapping chunks, and Markdown headings are separated from answer text. Multi-sentence answers remain possible when the second sentence contributes a relevant concept or explicit continuation.

### Regression coverage

- `test_extractive_access_paraphrases_preserve_vpn_and_mfa_conditions`
- `test_extractive_access_policy_preserves_actor_and_bilingual_rules`
- `test_extracts_two_distinct_sentences_when_both_match_the_question`
- `test_duplicate_sentences_from_overlapping_chunks_appear_once`
- `test_markdown_heading_is_not_attached_to_relevant_sentence`

### Result

Employee-specific questions return only employee conditions; contractor-specific questions return only the contractor prohibition. Retrieved chunks still expose all surrounding evidence.

The Evaluation Dashboard labels over-extraction as unmeasured because these pytest regressions do not produce a dedicated persisted benchmark metric or historical count.

## 6. Broad-Policy Answer Under-Extraction

### Defect

For “What security requirements apply to remote access?”, extractive mode returned only the contractor prohibition and omitted the employee VPN/MFA requirement. LLM-grounded mode returned both.

### Root cause

The subject and polarity filters that fixed narrow-question over-extraction were too restrictive for summary-style policy questions. The first selected sentence became the primary intent, and the second distinct requirement could be excluded as a different subject or opposite polarity.

### Corrective action

The pipeline now distinguishes broad policy-summary questions from narrow entity-specific questions. Generic words such as `requirements`, `policy`, and their Chinese equivalents are removed from evidence concepts, while the underlying domain concept remains. When no narrow subject anchor is present, aggregation may include a second distinct supported requirement even when permission polarities differ.

### Regression coverage

- `test_extractive_broad_policy_questions_include_distinct_supported_requirements`
- English and Chinese broad-policy parameter cases
- Narrow English and Chinese employee/contractor cases to prevent regression to over-extraction
- Unsupported-question, prompt-injection, and contradiction safety cases

### Result

Broad English and Chinese remote-access questions include both the employee VPN/MFA conditions and contractor prohibition. Narrow questions remain subject-specific, unrelated policy content is excluded, and existing safety gates continue to pass.

## Current Verification Context

After these corrections, the current reported and saved baseline is:

- 181 repository-wide pytest tests passed;
- 73/73 main benchmark cases passed;
- 26/26 challenge cases passed;
- 12/12 legacy security cases passed;
- 15/15 enterprise safety cases passed;
- 9/9 regression quality gates passed.

These results show that the documented regressions are covered by the current repository. They do not imply that every paraphrase, contradiction, injection technique, or over/under-extraction pattern is comprehensively detected.
