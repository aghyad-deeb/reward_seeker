# Commit Styles Across Grading/Evaluation Repositories

A catalog of how commits look across the major grading and evaluation repositories, based on examining their actual commit histories.

---

## 1. METR Task Standard

**Repo**: [METR/task-standard](https://github.com/METR/task-standard) (~35 commits total, very low frequency)

**Commit style**: Clean, descriptive, imperative mood. Many commits are bulk "sync" operations from an internal repo. Single-author dominance (tbroadley). No conventional-commit prefixes. No issue numbers in most commits.

**Example commits**:
```
Update CONTRIBUTING.md
Update license year
Sync task-standard 0.5.0 (#36)
Sync task-standard v0.4.1 (#34)
Change ram gib range from `tuple[int, int]` to `tuple[float, float]` (#27)
Sync task-standard v0.4.0 (#32)
Python 3.11 support (#29)
Add note about adaptors to README (#16)
Add task definitions for GAIA and GPQA's Diamond set
Move pico_ctf assets zip file to Amazon S3
Fix workbench submission waiting on Windows
Add agentbench and swe_bench task families to examples
Allow workbench to punch holes in no-internet tasks for agent
Set task environment variables in task tests
Add a simple hello_world TaskFamily (#7)
Make main.sh executable in example GPT-3.5 agent (#6)
Change single quotes to double quotes (#10)
Add diagram of task env creation & agent execution (#306)
Remove -it from workbench chown
Make sure agent has access to /home/agent/scaffolding dir
```

**Patterns**:
- Frequent "Sync task-standard" commits (bulk syncs from internal, with PR numbers)
- Simple imperative verbs: "Add", "Update", "Fix", "Move", "Change", "Remove"
- PR numbers in parentheses `(#N)` for external contributions
- No body text visible in most commits
- Version bumps reference semver: `0.5.0`, `v0.4.1`, `v0.4.0`
- No conventional-commit prefixes (no `feat:`, `fix:`, etc.)

---

## 2. METR RE-Bench

**Repo**: [METR/RE-Bench](https://github.com/METR/RE-Bench) (~25 commits total, very low frequency)

**Commit style**: Minimal, terse. Heavy use of a recurring "Stage suite" commit message for bulk task updates. Multiple authors. No prefixes.

**Example commits**:
```
Stage suite (#39)
Stage suite (#36)
Move tasks to top level (#33)
Prod Build and setup improvements (#19)
Add links to setup guide, Vivaria, and agent code. (#20)
Update README.md (#18)
Stage suite (#16)
Update solution files (#15)
Remove mention of AI_RD_GIT_AUTH
Stage suite
Fix setup script and instructions (#13)
'Stage suite' (#11)
Stage suite (#10)
Document uid/gid mapping
correct agent branch name in docs
Update to latest tasks and instructions
Staging the AI R&D repo (#3)
Fix test docker run command
Explanation of fstab for XFS lvgroup
Add notes about pquota
Add notes about logical volume group
Initial commit
```

**Patterns**:
- "Stage suite" is the dominant commit message — used repeatedly for bulk updates of task environments, with PR numbers
- Very terse, often just a few words
- Infrastructure commits are descriptive: "Document uid/gid mapping", "Explanation of fstab for XFS lvgroup"
- Mix of PR-based and direct commits
- Some inconsistency: `'Stage suite'` with quotes, `Stage suite` without
- Lowercase starts sometimes: "correct agent branch name in docs"

---

## 3. SWE-bench

**Repo**: [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) (~500+ commits, moderate frequency)

**Commit style**: Mixed. Some use conventional-commit-like prefixes (`Doc:`, `docs:`, `Doc(blog):`), others are plain imperative. Community contributions via PRs. Some very terse.

**Example commits**:
```
Fix flaky java evals (#506)
Fix multilingual image rebuild errors (#507)
Remove blog (moved to website)
Fixes #489
Fixes #492
Doc: Add codeclash link, standardize logos (#497)
Docs: Add mini-swe-agent to SWE-bench ecosystem projects (#477)
docs: fix warning format error in other languages (#481)
docs: Fix link to SWE-bench multimodal docs (#479)
Doc: Clarify Docker cleanup commands in setup guide (#473)
version bump
Fix git log leakage in environment images (#471)
docs: fix broken link in swebench/collect/README.md (#468)
update make_repo mirror code
Update README with SWE-bench Multimodal citation
Minor README update
Release 4.0.5
Remove fake instance id
changed grading.py to correctly parse the logs from patch files application. (#427)
add extra validation for make_run_report (#456)
Fixing images removing for non-default namespaces with `/` inside (#463)
instance_ids should be space separated, not comma separated (#459)
Update `pytest` workflow (#464)
Update tests
Nicer logging for `run_evaluation`
Minor fix: Add shorthands for run_evaluation
default to x86 images always
Doc(blog): Fix broken links
Fix formatting in FAQ
```

**Patterns**:
- Inconsistent prefix usage: `Doc:`, `docs:`, `Doc(blog):` — not strictly conventional commits but leaning that way
- `Fixes #N` for issue references (GitHub auto-close syntax)
- Release commits: "Release 4.0.5", "version bump"
- Many community PRs with `(#N)` pattern
- Mix of casing: some start lowercase (`docs: fix...`, `default to x86...`), some uppercase
- Past tense sometimes: "changed grading.py to correctly parse..."
- Present tense imperative other times: "Fix flaky java evals"
- Descriptive bug fix messages: "Fixing images removing for non-default namespaces with `/` inside"

---

## 4. OpenAI Frontier Evals

**Repo**: [openai/frontier-evals](https://github.com/openai/frontier-evals) (~97 commits, moderate frequency)

**Commit style**: Strongly prefixed with `[project]` tags in square brackets. Clean, professional. Single primary contributor (thesofakillers). PR-based workflow.

**Example commits**:
```
[pb] slimmer CI (#97)
[alctz/swelancer] Minor fixes following up the stricter alctz NetworkMode.NONE (#96)
[alctz] Stricter operation of NetworkMode.NONE and sync from upstream (#95)
[PB] Minor cleanup (#94)
[PB] Support Generic Computer Interface (#93)
[PB] Use chz for configuring Monitor and Judge scripts (#92)
[PB] cleanup small TODOs (#91)
[PB] Remove stale files (#90)
[PB] Add BasicAgent solver (#89)
[TurnCompleter] pin openai version in turn completer (#88)
[Alctz] Various updates to alcatraz comp interface (#87)
[PB] Small patch to LFS instructions (#86)
[PB] Enabling installing without `UV_GIT_LFS=1` (#84)
[PB] Packaging fixes to various lib to enable non-editable installing (#82)
[PB] Add retries to alcatraz computer start (#81)
[repo] Minor path fixes (#80)
[readme] reorganize sections (#78)
[readme] update readme with repository layout, requirements, and workflow (#77)
[common] fix ruff config path in pre-commit setup (#76)
[common] move ruff config to shared tooling and update pre-commit (#75)
[common][paperbench][swelancer] move libs to common and update paths (#74)
[readme] update to frontier evals and simplify (#73)
[PB] dont pass around boundloggers; instantiate when needed (#70)
[swel] Remove RunnerArgs instance and update readme/tests (#69)
[SWEL] minor fix necessary for monolith image (#68)
[SWEL] Update readme (#67)
[repo] make CI more interconnected (#66)
[TurnCompleter] updates to turn completer (#63)
hot patches to disable internt alctz functionality (#64)
modified disable internet function, allows for container specific blockers (#62)
[TurnCompleter] Responses API TurnCompleter (#55)
[TurnCompleter] Add ruff+mypy to minimono_turn_completer, fix typing (#54)
```

**Patterns**:
- **Square-bracket project prefix** is the defining pattern: `[PB]` (PaperBench), `[SWEL]`/`[swel]` (SWE-Lancer), `[alctz]` (Alcatraz sandbox), `[TurnCompleter]`, `[repo]`, `[readme]`, `[common]`
- Inconsistent casing of prefixes: `[PB]` vs `[pb]`, `[SWEL]` vs `[swel]`
- Multi-prefix for cross-cutting changes: `[common][paperbench][swelancer]`
- Every PR has a number: `(#N)`
- Informal language sometimes: "dont pass around boundloggers", "hot patches to disable internt alctz functionality"
- Occasional typos left in: "internt" (internet)

---

## 5. UK AISI Inspect AI

**Repo**: [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) (~3200+ commits, very high frequency — multiple per day)

**Commit style**: Very active, fast-paced. Mix of PR merges and direct commits. Descriptive, sentence-like messages. Heavy use of component-scoped messages. Single primary maintainer (jjallaire) with community PRs.

**Example commits**:
```
Google: Hard failure for quota exceeded errors with `limit: 0` (#3215)
Support Zstd compression on Python 3.14 with env var (#3145)
Model API: for 400 errors, print the error after the request payload rather than before. (#3210)
Anthropic: Do not pass through unrecognized `extra_body` fields. (#3208)
Update CHANGELOG for version 0.3.176 (#3207)
Update CHANGELOG with new features and bugfix
Fix Python API KeyError with ComposeConfig sandbox environments (#3204)
Prevent silent auto-merges of CHANGELOG.md (#3203)
Merge pull request #3189 from ransomr/async_read_log_headers
fix CHANGELOG.md
Merge remote-tracking branch 'origin/main' into async_read_log_headers
move zstandard dependency to requirements.txt
update changlog for release
Don't strictly require OpenAI and Anthropic versions when they aren't in use
add required model versions to uv lock
update uv lock
don't allow upgrade to griffe 2
remove sandbox agent bridge examples (point to inspect-swe instead)
tests for read_member_fully
add test_async_zip.py
update changelog
Compaction: Remove reasoning blocks from `compact()` result for Anthropic provider
mypy fix
remove central directory cache
```

**Patterns**:
- **Provider/component prefix with colon**: `Google:`, `Anthropic:`, `Model API:`, `Compaction:` — similar to conventional commits but using component names instead of `feat:`/`fix:`
- Sentence-like descriptions, sometimes ending with periods
- Version-related: "Update CHANGELOG for version 0.3.176"
- Very frequent merge commits: "Merge pull request #N from user/branch", "Merge remote-tracking branch..."
- Mix of sentence case and lowercase starts
- Some terse utility commits: "mypy fix", "update uv lock", "fix CHANGELOG.md"
- Occasional typos: "changlog" (changelog)
- High commit count per day (sometimes 10+ commits in a single day)

---

## 6. Alibaba DeepResearch

**Repo**: [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) (~298 commits, moderate frequency)

**Commit style**: Very terse, informal. Many commits from GitHub web UI ("Update", "Add files via upload"). Multiple authors with different styles. Some commits are just dates.

**Example commits**:
```
Update prompt.py
delete datasets
update AgentFold
update NestBrowse and ParallelMuse
fix parse_retry_times bug (#233)
Add new paper reference
fix logic bug
fix bug
WebLeaper model released
Merge pull request #226 from Alibaba-NLP/webwatcher1217
1217
Update README.md
Fix typo in visit tool
Add function to check for Chinese characters
add paper links
update tech report
Update example format for file parser tool usage
edit evaluate_hle_official.py in webwatcher
Add files via upload
Merge pull request #208 from Alibaba-NLP/merge-webwatcher
Merge branch 'webwatcher1019'
1109
update citation for Tongyi DeepResearch
Update citation for Tongyi DeepResearch
1101
add paper link
update webleaper
update webleaper
upload tech report
add new family
add FAQ and new family
fix typos
Clean start
update BrowseComp-vl benchmark of WebWatcher (#181)
```

**Patterns**:
- **Very minimal/terse**: "fix bug", "fix logic bug", "1217", "1109", "1101" (just dates as commit messages)
- **GitHub web UI defaults**: "Update prompt.py", "Update README.md", "Add files via upload"
- Duplicate/near-duplicate messages: "update webleaper" twice in a row, "update citation" and "Update citation"
- Inconsistent casing: "add paper link" vs "Add new paper reference"
- No conventional prefixes, no issue numbers in most commits
- Merge commits from branch-based workflow
- Feature announcement commits: "WebLeaper model released"
- Research-oriented: "add paper links", "upload tech report", "add new family" (referring to agent family papers)

---

## 7. testlib.h (Codeforces/Polygon)

**Repo**: [MikeMirzayanov/testlib](https://github.com/MikeMirzayanov/testlib) (~200 commits, low frequency, single author dominance)

**Commit style**: Terse, systems-level C++ style. Single primary author (MikeMirzayanov). Many CI/build-related commits. Minimal descriptions.

**Example commits**:
```
+ macos-15, macos-26 ci, - macos-26-amd64
Temporarily removed CI on ubuntu-20.04, macos26arm -> macos15arm
Temporarily removed CI on ubuntu-20.04, macos26arm -> macos15arm
Add workflow_dispatch trigger to CI workflow
Remove CI macos-13, added macos-26-arm64
Remove g++12 from macos-14 CI workflow
Remove macOS 12 test jobs from CI configuration
Remove g++9 from ubuntu 22.04 CI workflow
Merge pull request #216 from nsychev/patch-2
0.9.45: Remove incorrect const attributes
Merge pull request #223 from fy-hb/remove_attribute_const
fix #222
Fix test case message in multitest checkers with partial scores
Merge pull request #213 from MikeMirzayanov/fix-workflows
Removed meaningless update comment
some msvc warn fixed
use sscanf_s and similar for msvc
more tests
format issue
+ macos-14
- macos-11; + test-008_format
fix
fix macos issue?
Fallback to std::format for c++20, fixed some warns
revert 10000->10'000, fix macos-12-gpp, macos-13-gpp
10000 -> 10'000 and similar
docs/read.me
#define VERSION "0.9.43"
```

**Patterns**:
- **Shorthand notation**: `+ macos-15` (added), `- macos-11` (removed), `->` for changes
- Very terse: "fix", "more tests", "format issue"
- Version bumps as code: `#define VERSION "0.9.43"`, `0.9.45: Remove incorrect const attributes`
- Question marks in commits: "fix macos issue?"
- Duplicate commit messages (same CI fix committed twice)
- C++ specific: "some msvc warn fixed", "use sscanf_s and similar for msvc", "Fallback to std::format for c++20"
- Mix of languages/styles reflecting a single-maintainer project

---

## 8. Gradescope Autograder Samples

**Repo**: [gradescope/autograder_samples](https://github.com/gradescope/autograder_samples) (~130 commits, low frequency)

**Commit style**: Clean, documentation-focused. Many merge commits. PR-based workflow with internal ticket references.

**Example commits**:
```
Merge pull request #132 from gradescope/dgoodwin-gs-patch-1
Update submission_metadata.md
Merge pull request #120 from gradescope/king/GSC-6394/update-autograder-documentation
update copy
Merge pull request #109 from gradescope/skaminsky/GSC-3739/add-note-about-new-base-images
Added some docs
Merge pull request #105 from gradescope/skaminsky/GSC-5582/add-autograder-better-practices
Title cased isolate student code execution
Okay internal links should now be fixed
Fixed some links
Added it to the sidebar
Added best practices
Merge pull request #104 from espertus/patch-1
Add new Java autograder
Merge pull request #101 from gradescope/GSC-5562]-update-pricing
add link
update pricing copy
Merge pull request #97 from gradescope/ibrahima-patch-1
Add link to support forum on troubleshooting page
Merge pull request #93 from gradescope/skaminsky/GSC-5094/add-note-about-line-endings
Add line break
Added note about line endings
Capitalized
Added space
Made spec changes
Correct year of most recent update
Fix broken link to Community Resources page
removed unneeded word
```

**Patterns**:
- **Internal ticket IDs in branch names**: `GSC-6394`, `GSC-3739`, `GSC-5582`, `GSC-5094`, `GSC-5562` — visible in merge commit messages
- Branch naming convention: `username/TICKET-ID/description`
- Very granular commits: "Added space", "Capitalized", "Add line break", "removed unneeded word"
- Past tense mixed with imperative: "Added some docs" vs "Add new Java autograder"
- Documentation-heavy: most commits are about docs, specs, links

---

## 9. DOMjudge

**Repo**: [DOMjudge/domjudge](https://github.com/DOMjudge/domjudge) (~5000+ commits, very high frequency)

**Commit style**: Descriptive, sentence-like. Professional, consistent. Focused on the judging system. Direct commits (no PR workflow visible). Active single maintainer (meisterT) with occasional contributors.

**Example commits**:
```
Bump vrana/adminer from 5.4.1 to 5.4.2 in /webapp
Remove `->setAccessible(...)` calls from our reflective tests.
Make importing from external source work for contests without freeze.
Avoid toggle re-initialization over and over again when adding problems.
Fix variable access in judgedaemon.
Only test our minimal & maximum PHP version
Sort PHP extensions
Fix intended newline after markdown conversion
runpipe: log signals outside of signal handler.
Add option to enforce compiler/runner versions.
Add config option to accept first compiler/runner version as canonical.
Get rid of ugly putenv/getenv pattern in judgedaemon.
Improve external import progress messages.
Fix NPE in shadowing
Expose source service loading error.
Fix multiple issue with version selection on our documentation page.
Style team affiliation edit/view similar to the contest edit/view pages.
Style team category edit/view similar to the contest edit/view pages.
Style user edit/view similar to the contest edit/view pages.
Style team edit/view similar to the contest edit/view pages.
Style problem edit/view similar to the contest edit/view pages.
Style language edit/view similar to the contest edit/view pages.
Move scoring diff threshold to a contest level tunable value.
Use human time diffs in more places to make units more clear and increase readability.
Catch invalid range/accept_score specifications.
Reject scores for pass-fail problems.
Fix keying for disabled languages.
Display validator stderr (separately).
Fix handling of delayed start time in team controller.
Consider that test case groups might override validator flags.
Avoid adding a scoring problem to a pass-fail contest and vice versa.
Better check data coming from executable zips.
```

**Patterns**:
- **Full sentences ending with periods** — most commits end with `.`
- Descriptive and specific: "Make importing from external source work for contests without freeze."
- Component scoping via natural language: "runpipe: log signals outside of signal handler."
- Batch styling commits: multiple "Style X edit/view similar to the contest edit/view pages." in sequence
- Dependabot bumps: "Bump vrana/adminer from 5.4.1 to 5.4.2 in /webapp"
- PHP/Symfony domain language: "judgedaemon", "team controller", "NPE in shadowing"
- No PR numbers — direct push workflow
- Opinionated language: "Get rid of ugly putenv/getenv pattern"

---

## Summary: Commit Style Patterns

### By Formality

| Level | Repos | Characteristics |
|-------|-------|----------------|
| **Most formal** | DOMjudge, Inspect AI | Full sentences, periods, component scoping |
| **Structured** | OpenAI Frontier Evals | Square-bracket project prefixes, PR numbers |
| **Semi-structured** | SWE-bench | Inconsistent `Doc:`/`docs:` prefixes, PR numbers |
| **Clean informal** | METR Task Standard, Gradescope | Imperative verbs, no prefixes, PR numbers |
| **Terse** | METR RE-Bench, testlib.h | Minimal words, shorthand notation |
| **Very informal** | DeepResearch | Dates as messages, "fix bug", GitHub UI defaults |

### Common Prefix/Scoping Styles

| Style | Example | Used By |
|-------|---------|---------|
| `[Project]` square brackets | `[PB] Add BasicAgent solver` | OpenAI Frontier Evals |
| `Component:` with colon | `Google: Hard failure for quota...` | Inspect AI |
| `type:` conventional-ish | `docs: fix broken link` | SWE-bench |
| `component:` lowercase | `runpipe: log signals outside...` | DOMjudge |
| None | `Fix workbench submission waiting` | METR, Gradescope, DeepResearch |

### Recurring Commit Message Types

| Type | Examples |
|------|---------|
| **Sync/stage** | "Sync task-standard 0.5.0", "Stage suite" |
| **Version bump** | "Release 4.0.5", "version bump", `#define VERSION "0.9.43"` |
| **Changelog** | "Update CHANGELOG for version 0.3.176", "update changlog for release" |
| **CI/build** | "+ macos-15", "Remove g++12 from macos-14 CI workflow" |
| **Dependency** | "Bump vrana/adminer from 5.4.1 to 5.4.2" |
| **Docs** | "Update README.md", "Add link to support forum" |
| **Bug fix** | "fix bug", "Fix flaky java evals", "Fix NPE in shadowing" |
| **GitHub UI** | "Update prompt.py", "Add files via upload" |

### PR Number Conventions

| Convention | Example | Used By |
|-----------|---------|---------|
| `(#N)` at end | `Fix flaky java evals (#506)` | SWE-bench, OpenAI, METR, Gradescope |
| `Fixes #N` | `Fixes #489` | SWE-bench |
| `fix #N` | `fix #222` | testlib.h |
| `Merge pull request #N from...` | `Merge pull request #216 from nsychev/patch-2` | All (GitHub default) |
| None | Most commits | DOMjudge, DeepResearch |

### Casing Patterns

| Style | Example | Used By |
|-------|---------|---------|
| Sentence case (capital start) | `Fix workbench submission waiting on Windows` | METR, DOMjudge |
| lowercase start | `update webleaper`, `fix bug` | DeepResearch, testlib.h |
| Mixed in same repo | `docs: fix...` vs `Doc: Add...` | SWE-bench |
| ALL CAPS prefix | `[SWEL]`, `[PB]` | OpenAI Frontier Evals |

### Commit Frequency

| Frequency | Repos | Rate |
|-----------|-------|------|
| Very high (daily) | Inspect AI, DOMjudge | 5-30 commits/day during active periods |
| Moderate | SWE-bench, DeepResearch, OpenAI | A few commits per week |
| Low | METR Task Standard, RE-Bench, Gradescope, testlib.h | Monthly or less |
