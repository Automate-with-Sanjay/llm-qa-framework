# LLM QA Testing Framework

**Author:** Sanjaykumar Venkatesan  
**Stack:** Python · Anthropic Claude API · JSON Reporting  
**Purpose:** Automated evaluation framework for LLM outputs — built to demonstrate QA skills applied to AI/LLM products.

---

## What This Does

This framework lets you write structured test cases for LLM-powered features and automatically evaluates responses across 5 quality dimensions:

| Dimension | What It Checks |
|---|---|
| **Keyword Coverage** | Are expected terms present? Are forbidden terms absent? |
| **Format Validation** | Is the output valid JSON, bullet list, or numbered list? |
| **Length Bounds** | Is the response within min/max character limits? |
| **Tone Detection** | Is the tone formal, friendly, or neutral as required? |
| **Hallucination Risk** | Are expected factual anchors grounded in the response? |

Each test gets a **score (0.0–1.0)** and a **PASS/FAIL** verdict. Results are saved as a structured JSON report.

---

## Project Structure

```
llm-qa-framework/
├── llm_qa_framework.py   # Core framework: evaluators + test runner
├── tests/
│   └── custom_tests.py   # Add your own test suites here
├── reports/
│   └── test_report.json  # Auto-generated after each run
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install anthropic
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the sample suite

```bash
python llm_qa_framework.py
```

### 4. View the report

```bash
cat reports/test_report.json
```

---

## Writing Your Own Test Cases

```python
from llm_qa_framework import TestCase, run_test, generate_report

my_tests = [
    TestCase(
        id="TC010",
        name="Chatbot greeting — tone check",
        prompt="Greet a new customer visiting our support page.",
        expected_keywords=["welcome", "help"],
        forbidden_keywords=["error", "sorry", "cannot"],
        tone="friendly",
        max_length=300,
    ),
    TestCase(
        id="TC011",
        name="Product summary — JSON format",
        prompt='Summarise this product in JSON: {"name": "...", "price": ..., "tags": [...]}',
        expected_format="json",
    ),
]

results = [run_test(tc) for tc in my_tests]
report = generate_report(results)
print(report["summary"])
```

---

## Sample Output

```
============================================================
  LLM QA Testing Framework — by Sanjaykumar Venkatesan
============================================================

[RUN] TC001: Summarisation — keyword coverage
  Status  : PASS
  Score   : 1.0
  Latency : 843.2 ms

[RUN] TC002: JSON output format
  Status  : PASS
  Score   : 1.0
  Latency : 612.7 ms

[RUN] TC003: Hallucination — factual grounding
  Status  : PASS
  Score   : 1.0
  Latency : 921.4 ms

============================================================
  SUMMARY: 5/5 passed
  Pass rate: 100%
  Avg score: 0.97
  Report saved → reports/test_report.json
============================================================
```

---

## Key Concepts Demonstrated

- **LLM output evaluation** — testing for accuracy, format, tone, hallucination
- **Test case design** — parameterised, reusable test structures
- **Automated scoring** — weighted multi-dimension scoring per test
- **CI/CD ready** — runs as a script, exits with error code on failures
- **JSON reporting** — structured output consumable by dashboards or Jira

---

## Roadmap (Future Enhancements)

- [ ] LLM-as-judge: use a second model call to evaluate response quality
- [ ] Playwright integration for end-to-end chatbot UI testing
- [ ] GitHub Actions CI workflow
- [ ] HTML dashboard for test reports
- [ ] Prompt regression testing (compare v1 vs v2 prompts)
- [ ] Support for OpenAI and Gemini APIs

---

## Skills Demonstrated

`Python` · `Test Automation` · `LLM Testing` · `Prompt Engineering` · `Hallucination Detection` · `API Testing` · `JSON` · `QA Engineering` · `Anthropic Claude API`
