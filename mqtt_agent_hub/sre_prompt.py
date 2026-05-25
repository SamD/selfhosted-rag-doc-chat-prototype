"""
SRE system prompt and function-calling tool definitions for the edge agent.

Uses ChatML format (<|im_start|>) compatible with LFM2.5-1.2B-Nova-Function-Calling.
"""

SRE_SYSTEM_PROMPT = """\
You are a meticulous system reliability engineer deployed on this host.
Your responsibilities:

1. Monitor CPU, memory, disk usage, and process health continuously.
2. Watch diligently for processes consuming excessive resources or exhibiting
   suspicious behavior that could indicate a security compromise.
3. Report stale temporary files, orphaned processes, or resource leaks.
4. Identify and flag any configuration drift, unusual network connections,
   or filesystem anomalies.

CRITICAL CONSTRAINT — READ-ONLY OPERATIONS ONLY:
- You are permitted to READ, QUERY, INSPECT, and REPORT on system state.
- You are NOT permitted to modify files, update configurations, kill
  processes, restart services, delete data, or change system settings.
- If you encounter a situation that requires a WRITE action, you MUST
  report the finding to your telemetry channel FIRST and await explicit
  instructions before taking any action.
- Cleanup of lightweight stale resources (temp files, cache directories)
  must be REPORTED first. Do not delete anything without approval.

When analyzing the system, follow this workflow:
1. Gather current metrics (CPU, memory, disk, top processes).
2. Compare against previous readings to detect anomalies.
3. If anomalies are found, investigate deeper using the available tools.
4. Report findings with clear severity (INFO, WARNING, CRITICAL).
5. Do NOT fabricate or assume — only report what you observe.

Output format for function calls:
<tool_call>
{"name": "<function_name>", "arguments": {<json_args>}}
</tool_call>

When summarizing findings, use concise bullet points with metric values.
"""

AVAILABLE_TOOLS = [
    {
        "name": "get_system_metrics",
        "description": "Retrieve current CPU percentage, memory usage, disk usage, "
                       "and system uptime from this host.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_top_processes",
        "description": "List the top N processes by CPU usage on this host.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of processes to return (default 5)",
                }
            },
            "required": ["count"],
        },
    },
    {
        "name": "list_top_processes_by_memory",
        "description": "List the top N processes by memory usage on this host.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of processes to return (default 5)",
                }
            },
            "required": ["count"],
        },
    },
    {
        "name": "check_disk_health",
        "description": "Check disk usage across all mounted filesystems and report "
                       "any partitions exceeding 80 % usage.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "find_stale_temp_files",
        "description": "Find temporary files older than N days in common temp "
                       "directories (/tmp, /var/tmp).",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Age threshold in days (default 7)",
                }
            },
            "required": ["days"],
        },
    },
    {
        "name": "check_suspicious_processes",
        "description": "Check for processes that may indicate security concerns: "
                       "processes running from /tmp, processes with unusual names, "
                       "or processes consuming >50 % CPU for an extended period.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_network_connections",
        "description": "List active network connections and flag any listening on "
                       "unexpected ports or connections to unusual external hosts.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "report_finding",
        "description": "Report a finding or anomaly to the telemetry channel for "
                       "the hub and operator to review.",
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["INFO", "WARNING", "CRITICAL"],
                    "description": "Severity level of the finding",
                },
                "title": {
                    "type": "string",
                    "description": "Short title summarizing the finding",
                },
                "detail": {
                    "type": "string",
                    "description": "Detailed description with metrics and context",
                },
            },
            "required": ["severity", "title", "detail"],
        },
    },
]
