# Repository guidance

## Purpose and style

- This is an engineering documentation repository for LLM inference, SGLang, and Ascend.
- Use Chinese for article prose; preserve established English identifiers and terminology.
- Keep the homepage concise and organized around published material. Do not add personal learning plans, current-focus lists, growth slogans, or inflated expertise claims.
- Prefer continuous engineering narratives with 5–7 major sections when appropriate. Do not force a fixed section count on short documents.
- Use diagrams only when they explain architecture, data flow, execution order, or tensor shapes.

## Technical evidence

- Read the relevant source before asserting implementation details. Record the repository, exact commit or release, and applicable configuration.
- Separate conceptual examples, source-confirmed behavior, and measured results. Never invent call paths, benchmark data, citations, or successful tests.
- Treat model-specific and device-specific paths explicitly; do not substitute generic MHA or CUDA behavior for verified Ascend/model behavior.
- Keep workplace details, credentials, internal addresses, and unpublished materials out of public commits unless explicitly authorized for publication.

## Layout and edits

- Articles live under docs/<topic>/ with descriptive kebab-case filenames.
- Update the relevant topic index when publishing an article; avoid empty article placeholders.
- Store figures under assets/<article-slug>/ and executable examples under examples/<example-slug>/.
- Templates are optional outlines, not mandatory boilerplate.
- Read existing content and any deeper instructions before editing. Preserve unrelated changes.
- Use small, focused commits. Never force-push or broaden permissions as part of documentation work.

## Validation

- Run python3 scripts/check_docs.py from the repository root.
- Inspect rendered Markdown for layout changes and manually verify source citations and external links.
- The automated check covers local file links and text hygiene, not remote URLs, anchors, technical accuracy, or code correctness.
- Report evidence and unverified boundaries accurately.
