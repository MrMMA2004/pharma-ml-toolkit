# Security Policy

## Supported versions

Security fixes are applied to the latest `main` branch and, when practical, included in the next release. Older archives are provided for reproducibility and are not guaranteed to receive backports.

| Version | Support |
| --- | --- |
| Latest release and `main` | Supported |
| Older releases | Upgrade recommended |

## Report a vulnerability

Please **do not open a public issue** for an undisclosed vulnerability, exposed credential, malicious-code execution path or sensitive-data leak.

Use GitHub's private vulnerability reporting:

**[Report a vulnerability privately](https://github.com/MrMMA2004/pharma-ml-toolkit/security/advisories/new)**

Include:

- the affected version, module and operating system;
- a minimal reproduction or proof of concept;
- the potential impact and any prerequisites;
- whether the issue is already public; and
- a suggested mitigation, if known.

The maintainer can use the private advisory to discuss the report, coordinate a fix and prepare disclosure. Please keep the details private until a coordinated resolution is available.

For non-security bugs, use the normal [bug-report form](https://github.com/MrMMA2004/pharma-ml-toolkit/issues/new?template=bug_report.yml).

## Scope notes

PharmaML processes local datasets, molecular structures, notebooks and generated files. Reports involving unsafe deserialization, path traversal, command injection, archive extraction, dependency confusion, credential disclosure or execution of untrusted notebook/model content are especially relevant.

PharmaML is research and education software, not a clinical decision-support system. A clinically unsafe output should be reported as a correctness issue; do not include patient-identifiable information in any report.
