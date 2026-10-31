"""Capabilities: the unit a product team ships. Each packages the tools an agent may call, its policies and its
autonomy level. The calling product surface picks the capability; the model never widens its own tool set.

Autonomy levels:
  propose_only          every write becomes a proposal that a person approves
  auto_under_threshold  writes run automatically unless a write's check asks for approval
"""
from __future__ import annotations

from .tools import READS

READ_TOOLS = list(READS)

CAPABILITIES = {
    "billing_support": {"tools": READ_TOOLS + ["issue_refund", "escalate"], "autonomy": "auto_under_threshold",
                        "purpose": "Resolve billing questions and refund overpayments."},
    "dispatch": {"tools": READ_TOOLS + ["reschedule_job", "assign_technician", "send_customer_message",
                                          "escalate"],
                 "autonomy": "auto_under_threshold", "purpose": "Move jobs, assign technicians and keep customers informed."},
    "warranty": {"tools": READ_TOOLS + ["file_warranty_claim", "escalate"], "autonomy": "auto_under_threshold",
                 "purpose": "File manufacturer warranty claims for completed or scheduled work."},
    "account_summary": {"tools": READ_TOOLS + ["escalate"], "autonomy": "auto_under_threshold",
                        "purpose": "Summarise a customer's account. Read-only."},
}
