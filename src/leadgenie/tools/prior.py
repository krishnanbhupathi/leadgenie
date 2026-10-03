"""lookup_prior_results: reuse what earlier runs already established about a company."""

from __future__ import annotations

import json
from typing import Any

from leadgenie.store import Store
from leadgenie.tools.base import Tool, ToolResult


def prior_tool(store: Store, current_run_id: str | None = None) -> Tool:
    async def handler(args: dict[str, Any]) -> ToolResult:
        rows = store.find_prior(args["company"], exclude_run_id=current_run_id)
        results = []
        docs = {}
        for row in rows:
            ref = f"prior:{row['lead_id']}"
            enrichment = json.loads(row["enrichment"]) if row["enrichment"] else {}
            results.append(
                {"ref": ref, "raw_company": row["raw_company"], "enrichment": enrichment}
            )
            docs[ref] = json.dumps(enrichment, ensure_ascii=False)
        return ToolResult.json({"company": args["company"], "results": results}, documents=docs)

    return Tool(
        name="lookup_prior_results",
        description=(
            "Look up approved enrichments from earlier runs (never the current run) for "
            "the same company, matched on the raw or normalized company name. Each result "
            "has a 'ref' you may cite as a source, plus the sources it cited itself. "
            "Results reflect the past; re-verify anything that may have changed."
        ),
        input_schema={
            "type": "object",
            "properties": {"company": {"type": "string", "description": "Company name."}},
            "required": ["company"],
            "additionalProperties": False,
        },
        handler=handler,
    )
