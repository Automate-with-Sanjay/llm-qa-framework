"""
Custom test suite examples — extend this file with your own LLM test cases.
Run: python tests/custom_tests.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_qa_framework import TestCase, run_test, generate_report
import json

# ── Chatbot / Copilot Tests ───────────────────────────────────────────────────

chatbot_tests = [
    TestCase(
        id="CB001",
        name="Support chatbot — friendly greeting",
        prompt="Hi, I need help with my order.",
        expected_keywords=["help", "order"],
        forbidden_keywords=["I cannot", "I'm unable"],
        tone="friendly",
        max_length=400,
    ),
    TestCase(
        id="CB002",
        name="Copilot — code explanation",
        prompt="Explain what this Python code does: for i in range(10): print(i)",
        expected_keywords=["loop", "print", "0", "9"],
        min_length=50,
        max_length=500,
    ),
]

# ── Summarisation Tests ───────────────────────────────────────────────────────

summarisation_tests = [
    TestCase(
        id="SUM001",
        name="Bullet summary — software testing",
        prompt="List 4 key benefits of automated testing in bullet points.",
        expected_keywords=["speed", "regression"],
        expected_format="bullet_list",
        max_length=800,
    ),
    TestCase(
        id="SUM002",
        name="Structured JSON summary",
        prompt='Give a JSON summary of Python with keys: "creator", "year", "type", "popular_uses" (list).',
        expected_format="json",
        hallucination_check=True,
        hallucination_facts=["guido", "1991"],
    ),
]

# ── Edge Case / Safety Tests ──────────────────────────────────────────────────

edge_case_tests = [
    TestCase(
        id="EC001",
        name="Empty-ish prompt handling",
        prompt="Tell me something.",
        min_length=20,
        forbidden_keywords=["I cannot", "I don't understand"],
    ),
    TestCase(
        id="EC002",
        name="Ambiguous prompt — no hallucination",
        prompt="Who won the match yesterday?",
        hallucination_check=True,
        hallucination_facts=["don't have", "not sure", "unable to", "real-time"],
    ),
]


if __name__ == "__main__":
    all_tests = chatbot_tests + summarisation_tests + edge_case_tests
    print(f"Running {len(all_tests)} custom tests...\n")
    results = [run_test(tc) for tc in all_tests]
    report = generate_report(results)

    os.makedirs("reports", exist_ok=True)
    with open("reports/custom_report.json", "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print(f"\nSummary: {s['passed']}/{s['total_tests']} passed | "
          f"Pass rate: {s['pass_rate']} | Avg score: {s['average_score']}")
    print("Report saved → reports/custom_report.json")
