"""
LLM QA Testing Framework
Author: Sanjaykumar Venkatesan
Description: Automated evaluation framework for LLM outputs — tests for
             hallucination, accuracy, tone, format, and edge-case behavior.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib import response
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    name: str
    prompt: str
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    expected_format: Optional[str] = None       # "json" | "bullet_list" | "numbered_list" | None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    tone: Optional[str] = None                  # "formal" | "friendly" | "neutral"
    hallucination_check: bool = False
    hallucination_facts: list[str] = field(default_factory=list)

@dataclass
class TestResult:
    test_id: str
    test_name: str
    prompt: str
    response: str
    passed: bool
    score: float                                # 0.0 – 1.0
    checks: dict
    latency_ms: float
    timestamp: str


# ─── Evaluators ──────────────────────────────────────────────────────────────

def check_keywords(response: str, expected: list[str], forbidden: list[str]) -> dict:
    """Check presence of expected keywords and absence of forbidden ones."""
    response_lower = response.lower()
    found = [kw for kw in expected if kw.lower() in response_lower]
    found_forbidden = [kw for kw in forbidden if kw.lower() in response_lower]
    passed = (len(found) == len(expected)) and (len(found_forbidden) == 0)
    return {
        "passed": passed,
        "expected_found": found,
        "expected_missing": [kw for kw in expected if kw.lower() not in response_lower],
        "forbidden_found": found_forbidden,
    }


def check_format(response: str, expected_format: str) -> dict:
    """Validate the structural format of the response."""
    if expected_format == "json":
        try:
            clean = response.strip()
            # Strip markdown code fences if present
            if "```" in clean:
                clean = re.sub(r"```(?:json)?\s*", "", clean).strip()
                print(f"Stripped code fences for JSON parsing. Cleaned response:\n{clean}")
            # Extract JSON block if there is intro text before it
            match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
            if match:
                clean = match.group(1).strip()
            json.loads(clean)
            return {"passed": True, "detail": "Valid JSON"}
        except json.JSONDecodeError as e:
            return {"passed": False, "detail": f"Invalid JSON: {e}"}

    if expected_format == "bullet_list":
        has_bullets = bool(re.search(r"^[\s]*[-•*]\s+\S", response, re.MULTILINE))
        return {"passed": has_bullets, "detail": "Bullet list detected" if has_bullets else "No bullet list found"}

    if expected_format == "numbered_list":
        has_numbered = bool(re.search(r"^\s*\d+[.)]\s+\S", response, re.MULTILINE))
        return {"passed": has_numbered, "detail": "Numbered list detected" if has_numbered else "No numbered list found"}

    return {"passed": True, "detail": "No format check required"}


def check_length(response: str, min_len: Optional[int], max_len: Optional[int]) -> dict:
    """Ensure response length is within bounds."""
    length = len(response)
    passed = True
    details = []
    if min_len and length < min_len:
        passed = False
        details.append(f"Too short: {length} < {min_len}")
    if max_len and length > max_len:
        passed = False
        details.append(f"Too long: {length} > {max_len}")
    if passed:
        details.append(f"Length OK: {length} chars")
    return {"passed": passed, "char_count": length, "detail": "; ".join(details)}


def check_tone(response: str, expected_tone: str) -> dict:
    """Heuristic tone detection."""
    formal_indicators = ["therefore", "furthermore", "consequently", "hereby", "pursuant","hi", "hello", "dear"]
    informal_indicators = ["hey", "awesome", "cool", "yeah", "lol", "!", "?!"]
    friendly_indicators = ["happy to", "glad to", "let me know", "feel free", "hope"]

    response_lower = response.lower()

    if expected_tone == "formal":
        score = sum(1 for w in formal_indicators if w in response_lower)
        informal_hits = sum(1 for w in informal_indicators if w in response_lower)
        passed = score > 0 and informal_hits == 0
        return {"passed": passed, "formal_signals": score, "informal_signals": informal_hits}

    if expected_tone == "friendly":
        score = sum(1 for w in friendly_indicators if w in response_lower)
        passed = score > 0
        return {"passed": passed, "friendly_signals": score}

    return {"passed": True, "detail": "No tone check required"}


def check_hallucination(response: str, facts: list[str]) -> dict:
    """
    Simple fact-grounding check.
    Verifies that all expected factual anchors are present in the response.
    A more advanced version would call another LLM as judge.
    """
    response_lower = response.lower()
    grounded = [f for f in facts if f.lower() in response_lower]
    missing = [f for f in facts if f.lower() not in response_lower]
    passed = len(missing) == 0
    return {
        "passed": passed,
        "grounded_facts": grounded,
        "missing_facts": missing,
        "hallucination_risk": "LOW" if passed else "HIGH",
    }


# ─── Core Runner ─────────────────────────────────────────────────────────────

def run_test(tc: TestCase, system_prompt: str = "You are a helpful assistant.") -> TestResult:
    """Call the LLM and evaluate the response against all checks."""
    start = time.time()
    response_text = ""

    try:
        response = client.chat.completions.create(
        model="openai/gpt-4o-mini",   # change model here anytime
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
        {"role": "user", "content": tc.prompt},
            ],
            )
        response_text = response.choices[0].message.content
    except Exception as e:
        response_text = f"[ERROR] {e}"

    latency_ms = (time.time() - start) * 1000

    checks = {}
    scores = []

    # Keyword check
    if tc.expected_keywords or tc.forbidden_keywords:
        kw = check_keywords(response_text, tc.expected_keywords, tc.forbidden_keywords)
        checks["keywords"] = kw
        scores.append(1.0 if kw["passed"] else 0.0)

    # Format check
    if tc.expected_format:
        fmt = check_format(response_text, tc.expected_format)
        checks["format"] = fmt
        scores.append(1.0 if fmt["passed"] else 0.0)

    # Length check
    if tc.min_length or tc.max_length:
        ln = check_length(response_text, tc.min_length, tc.max_length)
        checks["length"] = ln
        scores.append(1.0 if ln["passed"] else 0.0)

    # Tone check
    if tc.tone:
        tn = check_tone(response_text, tc.tone)
        checks["tone"] = tn
        scores.append(1.0 if tn["passed"] else 0.0)

    # Hallucination check
    if tc.hallucination_check and tc.hallucination_facts:
        hall = check_hallucination(response_text, tc.hallucination_facts)
        checks["hallucination"] = hall
        scores.append(1.0 if hall["passed"] else 0.0)

    overall_score = sum(scores) / len(scores) if scores else 1.0
    passed = overall_score >= 0.75

    return TestResult(
        test_id=tc.id,
        test_name=tc.name,
        prompt=tc.prompt,
        response=response_text,
        passed=passed,
        score=round(overall_score, 2),
        checks=checks,
        latency_ms=round(latency_ms, 1),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# ─── Report Generator ────────────────────────────────────────────────────────

def generate_report(results: list[TestResult]) -> dict:
    """Summarise test run into a structured report."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.score for r in results) / total, 2) if total else 0
    avg_latency = round(sum(r.latency_ms for r in results) / total, 1) if total else 0

    return {
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{round(passed / total * 100)}%" if total else "0%",
            "average_score": avg_score,
            "average_latency_ms": avg_latency,
        },
        "results": [
            {
                "id": r.test_id,
                "name": r.test_name,
                "passed": r.passed,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "checks": r.checks,
                "prompt": r.prompt,
                "response_preview": r.response[:300] + ("..." if len(r.response) > 300 else ""),
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }


# ─── Prompt & Response Logger ────────────────────────────────────────────────

def save_prompt_response(results: list[TestResult], filepath: str = "reports/prompt_response_log.txt") -> None:
    """
    Save only the prompt and response from each test into a human-readable .txt file.
    Each entry is clearly separated with dividers for easy reading.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    divider     = "=" * 80
    sub_divider = "-" * 80

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("LLM QA — Prompt & Response Log\n")
        f.write(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runs : {len(results)}\n")
        f.write(divider + "\n\n")

        for r in results:
            f.write(f"{divider}\n")
            f.write(f"[{r.test_id}] {r.test_name}\n")
            f.write(f"Timestamp : {r.timestamp}\n")
            f.write(f"{sub_divider}\n")
            f.write("PROMPT:\n")
            f.write(f"{r.prompt.strip()}\n\n")
            f.write(f"{sub_divider}\n")
            f.write("RESPONSE:\n")
            f.write(f"{r.response.strip()}\n\n")

    print(f"  Prompt/response log saved → {filepath}")


# ─── Sample Test Suite ────────────────────────────────────────────────────────

SAMPLE_TEST_SUITE = [
    TestCase(
        id="TC001",
        name="Summarisation — keyword coverage",
        prompt="Summarise the benefits of test automation in 3 bullet points and less that 500 words.",
        expected_keywords=["speed", "efficiency", "regression"],
        expected_format="bullet_list",
        max_length=600,
    ),
    TestCase(
        id="TC002",
        name="JSON output format",
        prompt='Return a JSON object with keys: "name", "role", "skills" (list of 3). Use fictional data.',
        expected_format="json",
        max_length=400,
    ),
    TestCase(
        id="TC003",
        name="Hallucination — factual grounding",
        prompt="Tell me about the Python programming language.",
        hallucination_check=True,
        hallucination_facts=["guido van rossum", "1991", "interpreted"],
    ),
    TestCase(
        id="TC004",
        name="Tone — formal response",
        prompt="Hi,Can you explain what a CI/CD pipeline is?",
        tone="formal",
        min_length=100,
    ),
    TestCase(
        id="TC005",
        name="Forbidden keyword — safety guardrail",
        prompt="What is machine learning?",
        expected_keywords=["data", "model"],
        forbidden_keywords=["I cannot", "I'm unable", "I don't know"],
        min_length=80,
    ),
]


if __name__ == "__main__":
    print("=" * 60)
    print("  LLM QA Testing Framework — by Sanjaykumar Venkatesan")
    print("=" * 60)

    results = []
    for tc in SAMPLE_TEST_SUITE:
        print(f"\n[RUN] {tc.id}: {tc.name}")
        result = run_test(tc)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  Status  : {status}")
        print(f"  Score   : {result.score}")
        print(f"  Latency : {result.latency_ms} ms")

    report = generate_report(results)

    with open("reports/test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    save_prompt_response(results)

    print("\n" + "=" * 60)
    print(f"  SUMMARY: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
    print(f"  Pass rate: {report['summary']['pass_rate']}")
    print(f"  Avg score: {report['summary']['average_score']}")
    print(f"  Report saved → reports/test_report.json")
    print("=" * 60)
