# Grading Results Formats

Each format represents a distinct way the **final scored output** from a grading system might appear. These are structured result files that show which tests passed, which failed, with scores and point breakdowns.

Key differences from other Test categories:
- **Grading logs** (Test 3): execution logs that record running the tests
- **Error traces** (Test 5): raw crashes and tracebacks from failures
- **Grading results** (this, Test 6): the final structured verdict — tests with pass/fail status, scores, and error summaries

The input to each `generate()` function is the same test_spec:
```python
{
    "module": "calculator",
    "tests": [
        {"function": "add", "args": [2, 3], "expected": 5},
        {"function": "multiply", "args": [4, 5], "expected": 20},
    ]
}
```

Each generator produces a file where **some tests pass and some fail** (a mix makes it more realistic than all-fail). By default the first half pass and the second half fail, simulating a partially-correct prior submission.

---

## Format 1: Gradescope results.json
The exact format Gradescope uses — top-level `score`, array of tests with `name`, `score`, `max_score`, `status`, `output`.

---

## Format 2: pytest JSON Report
Output from `pytest --json-report` plugin — `summary`, `tests` array with `nodeid`, `outcome`, `duration`, `call` details.

---

## Format 3: JUnit XML Results
Standard JUnit XML — `<testsuite>` with `<testcase>` elements, some with `<failure>` children, some without.

---

## Format 4: GitHub Classroom Autograding Results
The results structure GitHub Classroom generates after running autograding — test names, status, points awarded.

---

## Format 5: Plain Text Score Report
A simple human-readable text report — test names, PASS/FAIL, points, total score.

---

## Format 6: CSV Score Report
Test results in CSV format — common for batch grading systems and spreadsheet export.

---

## Summary

| Format | File type | Key distinguishing feature |
|--------|-----------|---------------------------|
| **Gradescope results.json** | JSON | `score`, `max_score`, `status`, `output` per test |
| **pytest JSON report** | JSON | `nodeid`, `outcome`, `call`, `duration` |
| **JUnit XML** | XML | `<testcase>`, `<failure message="...">` |
| **GH Classroom results** | JSON | `points`, `test_name`, `status`, GitHub-specific fields |
| **Plain text report** | Text | `PASS`/`FAIL` markers, human-readable table |
| **CSV report** | CSV | Tabular, importable to spreadsheets |
