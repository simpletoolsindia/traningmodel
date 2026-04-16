#!/usr/bin/env python3
"""
Gemma 4 Dataset Generator — YOUR Custom Format
===============================================
Format: Custom XML prompt + JSON tool call response
This is YOUR language - Gemma 4 will learn it through fine-tuning.

Input format:  <init>...<User:>task</User:>
Output format: {"message_type": "tool_call", "tool_call": {...}}
"""

import json
import random
import hashlib
from datetime import datetime

# ─────────────────────────────────────────────
# YOUR CUSTOM INPUT FORMAT
# ─────────────────────────────────────────────

def build_custom_prompt(agent_role, capabilities, bot_tone, language,
                         instruction, guardrails, strict_rules, tools,
                         memory, conversation_history, user_message):
    """Build YOUR custom XML prompt format."""
    tools_json = json.dumps(tools, ensure_ascii=False)

    return f"""<init>
<agent_role>{agent_role}</agent_role>
<agent_capabilities>
{chr(10).join(f"- {c}" for c in capabilities)}
</agent_capabilities>
<localization>
  <bot_tone>{bot_tone}</bot_tone>
  <language>{language}</language>
</localization>
</init>
<agent_instruction>{instruction}</agent_instruction>
<guardrails>
{chr(10).join(f"- {g}" for g in guardrails)}
</guardrails>
<strict_rules>
{chr(10).join(f"- {r}" for r in strict_rules)}
</strict_rules>
<humanize>false</humanize>
<tools>
  <list>{tools_json}</list>
  <tool-call-format>json</tool-call-format>
</tools>
<memory>
{chr(10).join(f"- {m}" for m in memory)}
</memory>
<conversation-history>
{conversation_history}
</conversation-history>

User: {user_message}"""

# ─────────────────────────────────────────────
# YOUR CUSTOM OUTPUT FORMAT
# ─────────────────────────────────────────────

def build_normal_response(content, language, framework, difficulty, category):
    return {
        "message_type": "normal",
        "content": content,
        "tool_call": None,
        "mcp_call": None,
        "turns": None,
        "humanize": False,
        "metadata": {
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }

def build_tool_call_response(tool_name, tool_args, language, framework, difficulty, category):
    import uuid
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    return {
        "message_type": "tool_call",
        "content": None,
        "tool_call": {
            "id": call_id,
            "name": tool_name,
            "arguments": tool_args,
        },
        "mcp_call": None,
        "turns": None,
        "humanize": False,
        "metadata": {
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }

def build_mcp_call_response(tool_name, tool_args, language, framework, difficulty, category):
    import uuid
    call_id = f"mcp_{uuid.uuid4().hex[:12]}"
    return {
        "message_type": "mcp_call",
        "content": None,
        "tool_call": None,
        "mcp_call": {
            "id": call_id,
            "server": "extra_skills_mcp",
            "name": tool_name,
            "arguments": tool_args,
        },
        "turns": None,
        "humanize": False,
        "metadata": {
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }

def build_multi_turn_response(turns, language, framework, difficulty, category):
    return {
        "message_type": "multi_turn",
        "content": None,
        "tool_call": None,
        "mcp_call": None,
        "turns": turns,
        "humanize": False,
        "metadata": {
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
            "turn_count": len(turns),
        }
    }

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MESSAGE_TYPES = {"normal": 0.25, "tool_call": 0.40, "multi_turn": 0.20, "mcp_call": 0.15}
LANGUAGES = {"python": 0.25, "javascript": 0.15, "typescript": 0.15, "java": 0.18,
             "kotlin": 0.10, "go": 0.05, "rust": 0.05, "shell": 0.05, "other": 0.02}
DIFFICULTIES = {"easy": 0.35, "medium": 0.45, "hard": 0.20}
CATEGORIES = {"backend": 0.25, "frontend": 0.20, "fullstack": 0.15, "devops": 0.15,
              "database": 0.08, "mobile": 0.07, "system": 0.05, "general": 0.05}

TOOL_NAMES = [
    "read", "write", "edit", "copy", "move", "delete", "mkdir",
    "stat", "exists", "list", "grep", "find", "glob",
    "bash", "top", "ps", "df", "du", "free", "whoami",
    "git_status", "git_add", "git_commit", "git_pull", "git_push",
    "git_branch", "git_stash", "git_log", "git_diff", "git_checkout",
    "datetime", "date_now", "web_search", "fetch",
]

AGENT_ROLES = {
    "Senior Python Developer": [
        "Expert in Python, FastAPI, Django, SQLAlchemy, Celery",
        "Strong in REST API development, data pipelines, and automation",
        "Writes tests with pytest and coverage reports",
    ],
    "Senior Java Developer": [
        "Expert in Java, Spring Boot, JPA/Hibernate, REST APIs",
        "Strong in microservices, debugging, and clean code principles",
        "Writes comprehensive unit tests with JUnit 5",
    ],
    "Senior React Developer": [
        "Expert in React, TypeScript, React Hooks, Redux, React Query",
        "Strong in component design, state management, and performance",
        "Writes tests with React Testing Library and Jest",
    ],
    "Full Stack Engineer": [
        "Expert in Node.js, Express, React, PostgreSQL, Docker, AWS",
        "Strong in API design, database architecture, and CI/CD",
        "Writes full-stack applications with TypeScript",
    ],
    "Senior Backend Developer": [
        "Expert in API design, database optimization, caching strategies",
        "Strong in performance tuning, security, and scalability",
    ],
    "DevOps Engineer": [
        "Expert in Kubernetes, Docker, Terraform, CI/CD pipelines",
        "Strong in infrastructure as code and monitoring",
    ],
}

AGENT_INSTRUCTIONS = [
    "You are a CLI coding assistant that helps developers write, debug, refactor, and review code.",
    "You are an expert CLI assistant. Provide accurate, well-structured code with clear explanations.",
    "You are a coding agent CLI. Follow best practices for all operations.",
    "You are a CLI developer assistant. Break down complex problems into simple steps.",
    "You are an expert coding CLI. Write production-quality code with proper error handling.",
]

GUARDRAIL_SETS = [
    ["Never delete files without explicit user confirmation",
     "Always show diffs before applying any changes",
     "Never execute destructive commands without asking"],
    ["Never commit secrets or credentials to version control",
     "Always use parameterized queries to prevent SQL injection",
     "Never disable security checks in production code",
     "Always write tests alongside new code"],
    ["Focus on clean, readable code over clever hacks",
     "Prefer explicit over implicit",
     "Always consider accessibility in UI code"],
]

STRICT_RULES_SETS = [
    ["Always use the correct tool for the task",
     "Return ONLY the JSON response, no additional text",
     "For file operations, use the appropriate file tool",
     "For git operations, use the git tool",
     "For web searches, use web_search tool",
     "For system info, use the appropriate system tool"],
    ["Use the most specific tool available for the operation",
     "Return only the JSON object, never add explanations outside JSON",
     "File reads use read tool, file writes use write tool",
     "Git operations use git_* tools, not bash for git commands"],
    ["Tool selection must match the operation type exactly",
     "Output format must be valid JSON only",
     "File tools for file operations, git tools for version control",
     "Web tools for network requests, system tools for OS operations"],
]

BOT_TONES = ["professional", "friendly", "technical", "concise", "helpful"]

# Task templates
NORMAL_TASKS = [
    "Explain the difference between asyncio, threading, and multiprocessing in Python",
    "How do I implement pagination in FastAPI for large datasets?",
    "What is the best way to handle database migrations in production?",
    "How should I structure a Python project for microservices?",
    "Why am I getting a memory leak error in my application?",
    "Compare PostgreSQL vs MongoDB for an analytics workload",
    "How do I implement retry logic with exponential backoff?",
    "What's the best approach for input validation?",
    "How do I set up CI/CD for a project using GitHub Actions?",
    "Explain the SOLID principles in development",
    "What are common performance bottlenecks and how to profile them?",
    "How do I implement rate limiting in an API gateway?",
    "What is the difference between optimistic and pessimistic locking?",
    "How do you handle timezone conversions?",
    "Explain CQRS pattern and when to use it",
    "How do I implement a circuit breaker pattern?",
    "What is the best way to cache API responses?",
    "How do I write async code that handles errors gracefully?",
    "Explain event sourcing and how to implement it",
    "What strategies exist for handling high traffic?",
    "How do I structure code for testability?",
    "What's the best way to handle JSON serialization?",
    "How do I implement graceful shutdown in services?",
    "Explain the difference between actors and promises",
    "How do I handle circular dependencies?",
    "What is the best approach for logging in services?",
    "How do I implement graceful degradation?",
    "Explain the 12-factor app methodology",
    "How do I optimize startup time?",
]

TOOL_TASKS = {
    "bash": ["Run pip install pytest and show me the output",
             "Execute pytest on the module", "Run pytest -v on tests"],
    "read": ["Read the service.py file", "Show me the main.py contents",
             "Display the config.py file"],
    "write": ["Write this content to a new file", "Create a new config file"],
    "edit": ["Update the CORS configuration", "Fix the typo in main.py"],
    "git_status": ["Check git status", "Show current branch and changes"],
    "git_commit": ["Commit the changes", "Save with message: feat: add feature"],
    "git_pull": ["Pull latest changes", "Update from remote"],
    "git_push": ["Push to remote", "Upload committed changes"],
    "git_branch": ["List all branches", "Create a new feature branch"],
    "datetime": ["What is the current timestamp?"],
    "web_search": ["Search for Python best practices"],
}

# ─────────────────────────────────────────────
# UNIQUE CONTENT GENERATORS
# ─────────────────────────────────────────────

PROJECT_NAMES = ["payment-gateway", "user-auth-service", "inventory-manager",
                 "analytics-dashboard", "notification-system", "file-processor"]

def weighted_choice(weights_dict):
    items = list(weights_dict.items())
    keys = [k for k, _ in items]
    weights = [w for _, w in items]
    return random.choices(keys, weights=weights, k=1)[0]

def pick_by_hash(items, seed, suffix=""):
    idx = int(hashlib.md5((seed + suffix).encode()).hexdigest()[:6], 16) % len(items)
    return items[idx]

def get_ext(language):
    return {"python": "py", "java": "java", "javascript": "js",
            "typescript": "ts", "kotlin": "kt", "go": "go", "rust": "rs"}.get(language, "txt")

def generate_memory(language, row_seed):
    project = pick_by_hash(PROJECT_NAMES, row_seed, "proj")
    module = pick_by_hash(["auth", "users", "orders", "products"], row_seed, "mod")
    ext = get_ext(language)
    last_modified = pick_by_hash(["2 hours ago", "yesterday", "today"], row_seed, "time")
    return [
        f"Project: {project}",
        f"Language: {language}",
        f"Framework: {pick_by_hash(['FastAPI', 'Django', 'Express', 'Spring Boot'], row_seed, 'fw')}",
        f"Current file: src/{module}/service.{ext}",
        f"Last modified: {last_modified}",
    ]

def generate_conversation_history():
    return """user: Can you help me with this project?
assistant: I'll help you with that. Let me check the project first."""

def gen_tool_args(tool_name, language, row_seed):
    module = pick_by_hash(["auth", "users", "orders", "products"], row_seed, "amod")
    ext = get_ext(language)

    args_map = {
        "bash": {"command": pick_by_hash([
            "python -m pytest src/tests/", "npm test", "./mvnw test",
            "pip install -r requirements.txt", "cargo test --package module",
            "python manage.py runserver", "npm run build"
        ], row_seed, "cmd"), "timeout": 60},
        "read": {"path": pick_by_hash([
            f"src/{module}/service.{ext}", f"src/{module}/models.{ext}",
            f"src/{module}/handlers.{ext}", f"src/{module}/__init__.{ext}"
        ], row_seed, "rp"), "offset": 1, "limit": pick_by_hash([50, 100, 150, 200], row_seed, "lim")},
        "write": {"path": f"src/{module}/new.{ext}", "content": f"# {module} module\n"},
        "edit": {"path": f"src/{module}/config.{ext}", "old_string": "# TODO",
                 "new_string": pick_by_hash(["# DONE", "# IMPLEMENTED", "# FIXED"], row_seed, "ns")},
        "copy": {"source": f"src/{module}", "destination": f"backup/{module}"},
        "move": {"source": f"src/{module}/old.{ext}", "destination": f"src/{module}/new.{ext}"},
        "delete": {"path": f"temp/{module}.tmp", "recursive": False},
        "mkdir": {"path": f"src/{module}/new", "parents": True},
        "stat": {"path": f"src/{module}/service.{ext}"},
        "exists": {"path": f"src/{module}/service.{ext}"},
        "list": {"path": f"src/{module}", "all": False},
        "grep": {"pattern": pick_by_hash(["TODO", "FIXME", "console.log"], row_seed, "pat"),
                 "path": "src", "file_pattern": f"*.{ext}"},
        "find": {"path": "src", "name": f"*.{ext}", "type": "f", "max_depth": 5},
        "glob": {"pattern": f"src/**/*.{'py' if language == 'python' else 'js'}", "working_dir": "/project"},
        "git_status": {"working_dir": f"/projects/{module}"},
        "git_add": {"all": True, "working_dir": f"/projects/{module}"},
        "git_commit": {"message": pick_by_hash([
            "feat: add pagination to users API",
            "fix: resolve race condition in orders",
            "refactor: improve auth service layer",
            "docs: update API documentation",
            "test: add integration tests for products"
        ], row_seed, "msg"), "working_dir": f"/projects/{module}"},
        "git_pull": {"remote": "origin", "branch": pick_by_hash(["main", "develop"], row_seed, "br")},
        "git_push": {"remote": "origin", "branch": "main"},
        "git_branch": {"list": True},
        "git_stash": {"action": "push", "message": "WIP: feature implementation"},
        "git_log": {"path": ".", "limit": pick_by_hash([5, 10, 15], row_seed, "lg")},
        "git_diff": {"path": ".", "cached": False},
        "git_checkout": {"branch": pick_by_hash(["feature/new", "develop", "main"], row_seed, "ch")},
        "datetime": {"format": "%Y-%m-%d %H:%M:%S", "timezone": "UTC"},
        "date_now": {"format": "%Y-%m-%d %H:%M:%S", "timezone": "UTC"},
        "web_search": {"query": pick_by_hash([
            "Python async best practices 2024",
            "FastAPI performance tuning",
            "React component patterns",
            "Microservices architecture patterns",
            "CI/CD best practices GitHub Actions"
        ], row_seed, "sq"), "max_results": pick_by_hash([3, 5, 8], row_seed, "mr")},
        "fetch": {"url": f"https://api.example.com/{module}", "method": "GET"},
        "top": {"process_count": 10, "sort_by": "cpu"},
        "ps": {"aux": True, "filter": language},
        "df": {"path": "/", "human_readable": True},
        "du": {"path": "src", "max_depth": 1},
        "free": {"human_readable": True},
        "whoami": {},
    }
    return args_map.get(tool_name, {
        "value": f"{tool_name}_example",
        "timeout": 60,
        "path": f"src/{module}/file.{ext}",
    })

def gen_code_snippet(language, difficulty, row_seed):
    class_name = pick_by_hash(["UserService", "PaymentProcessor", "AuthManager", "DataValidator"], row_seed, "c")
    resource = pick_by_hash(["users", "orders", "products"], row_seed, "r")

    snippets = {
        "python": {
            "easy": f"def get_{resource}_by_id(id: int) -> dict:\n    return {{'id': id}}",
            "medium": f"class {class_name}:\n    def __init__(self):\n        self.db = None\n    \n    def find_{resource}(self, filters):\n        return self.db.query(filters)",
            "hard": f"class {class_name}Cache:\n    def __init__(self, redis, ttl=300):\n        self.redis = redis\n        self.ttl = ttl\n\n    def get_or_compute(self, key, compute):\n        cached = self.redis.get(key)\n        if cached:\n            return json.loads(cached)\n        result = compute()\n        self.redis.setex(key, self.ttl, json.dumps(result))\n        return result",
        },
        "java": {
            "easy": f"public class {class_name} {{\n    private String id;\n}}",
            "medium": f"@Service\npublic class {class_name} {{\n    public Optional<{class_name}DTO> findById(Long id) {{\n        return repository.findById(id).map(this::toDTO);\n    }}\n}}",
            "hard": f"@Configuration\npublic class {class_name}Config {{\n    @Bean\n    public ReactiveCrudRepository<{class_name}Entity, Long> repository() {{\n        return new CassandraReactiveCrudRepository<>();\n    }}\n}}",
        },
    }
    return snippets.get(language, snippets["python"]).get(difficulty, snippets["python"]["medium"])

# ─────────────────────────────────────────────
# GENERATE ROW
# ─────────────────────────────────────────────

def generate_row(row_id):
    message_type = weighted_choice(MESSAGE_TYPES)
    language = weighted_choice(LANGUAGES)
    difficulty = weighted_choice(DIFFICULTIES)
    category = weighted_choice(CATEGORIES)

    row_seed = f"{row_id}_{language}_{message_type}"
    framework = pick_by_hash(["none", "FastAPI", "Django", "Express", "React"], row_seed, "fw")

    # Build prompt components
    agent_role = random.choice(list(AGENT_ROLES.keys()))
    capabilities = AGENT_ROLES[agent_role]
    bot_tone = random.choice(BOT_TONES)
    instruction = random.choice(AGENT_INSTRUCTIONS)
    guardrails = random.choice(GUARDRAIL_SETS)
    strict_rules = random.choice(STRICT_RULES_SETS)
    num_tools = random.randint(5, 10)
    tools = random.sample(TOOL_NAMES, min(num_tools, len(TOOL_NAMES)))
    memory = generate_memory(language, row_seed)
    conversation_history = generate_conversation_history()

    # Generate user message and response
    if message_type == "normal":
        user_message = pick_by_hash(NORMAL_TASKS, row_seed, "nt")
        code_snippet = gen_code_snippet(language, difficulty, row_seed)
        response_content = f"Here's the solution:\n\n```\n{code_snippet}\n```\n\nLet me know if you need modifications!"
        response = build_normal_response(response_content, language, framework, difficulty, category)

    elif message_type == "tool_call":
        tool_name = random.choice(TOOL_NAMES)
        user_message = pick_by_hash(TOOL_TASKS.get(tool_name, [f"Use {tool_name}"]), row_seed, f"tt_{tool_name}")
        tool_args = gen_tool_args(tool_name, language, row_seed)
        response = build_tool_call_response(tool_name, tool_args, language, framework, difficulty, category)

    elif message_type == "mcp_call":
        user_message = "Search for Python best practices online"
        tool_args = {"query": "Python best practices", "max_results": 5}
        response = build_mcp_call_response("web_search", tool_args, language, framework, difficulty, category)

    else:  # multi_turn
        import uuid
        user_message = pick_by_hash([
            "Build a REST API endpoint for users",
            "Create a service class for orders",
            "Implement authentication middleware",
        ], row_seed, "mt")

        tool_name = random.choice(TOOL_NAMES)
        tool_args = gen_tool_args(tool_name, language, row_seed + "mt")
        mt_call_id = f"call_{uuid.uuid4().hex[:12]}"

        turns = [
            {"turn_id": 1, "type": "tool_call", "tool_call": {
                "id": mt_call_id,
                "name": tool_name,
                "arguments": tool_args,
            }},
            {"turn_id": 2, "type": "tool_response", "content": "[Operation completed]"},
            {"turn_id": 3, "type": "assistant", "content": "Done. Based on the results."},
        ]
        response = build_multi_turn_response(turns, language, framework, difficulty, category)

    # Build YOUR custom prompt format
    prompt = build_custom_prompt(
        agent_role=agent_role,
        capabilities=capabilities,
        bot_tone=bot_tone,
        language="en",
        instruction=instruction,
        guardrails=guardrails,
        strict_rules=strict_rules,
        tools=tools,
        memory=memory,
        conversation_history=conversation_history,
        user_message=user_message,
    )

    # YOUR custom output format
    output_json = json.dumps(response, ensure_ascii=False)

    # Format as proper chat messages (fixes "Helper model mapping missing user/assistant" warning)
    return {
        "id": str(row_id),
        "prompt": prompt,
        "output_json": output_json,
        # Chat-formatted for Unsloth (fixes "Helper model mapping missing user/assistant" warning)
        "text": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output_json}
        ],
        "language": language,
        "framework": framework,
        "message_type": message_type,
        "difficulty": difficulty,
        "category": category,
        "agent_role": agent_role,
        "tools": json.dumps(tools),
        "metadata": json.dumps({
            "source": "synthetic",
            "version": "1.0",
            "difficulty_score": {"easy": 1, "medium": 2, "hard": 3}[difficulty],
        }),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_dataset(num_rows, output_file):
    import time

    print(f"Generating {num_rows:,} rows in YOUR custom format...")
    print(f"Output: {output_file}")
    print(f"Format: JSONL with chat-formatted 'text' field")
    print()

    start_time = time.time()
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_rows):
            row = generate_row(i + 1)
            # For Unsloth: use chat-formatted "text" field (list of role messages)
            # This fixes "Helper model mapping missing user/assistant" warning
            out_row = {"text": row["text"]}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if (i + 1) % 100_000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  {i+1:,} / {num_rows:,} ({rate:,.0f} rows/sec)")

    elapsed = time.time() - start_time
    print(f"\nDone: {num_rows:,} rows in {elapsed:.1f}s")
    print(f"Format: JSONL with 'text' field (Unsloth-ready)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="training_data.jsonl")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    num_rows = 5_000_000 if args.full else args.rows
    generate_dataset(num_rows, args.output)