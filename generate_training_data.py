#!/usr/bin/env python3
"""
CLI Coding Agent Dataset Generator v3 — 5M Training Examples
Model: Gemma 4 E4B (Google) + Unsloth + LoRA
Format: Unified JSON with message_type for easy parsing

Tools: File, System, Git, DateTime, Web
New: <strict_rules> added to prompt format

Usage:
    python generate_training_data.py                    # Generate sample (1000 rows)
    python generate_training_data.py --full           # Generate full 5M dataset
    python generate_training_data.py --rows 100000     # Generate 100K rows
"""

import json
import csv
import random
import uuid
import argparse
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

NUM_ROWS = 5_000_000
OUTPUT_FILE = "training_data.csv"

# Message type distribution
MESSAGE_TYPES = {
    "normal": 0.35,     # 35% - conversational responses
    "tool_call": 0.35,  # 35% - tool calls (file, system, git, datetime, web)
    "mcp_call": 0.15,   # 15% - MCP server tools
    "multi_turn": 0.15, # 15% - multi-turn conversations
}

# Language distribution
LANGUAGES = {
    "python": 0.22,
    "javascript": 0.12,
    "typescript": 0.10,
    "java": 0.18,
    "kotlin": 0.08,
    "go": 0.05,
    "rust": 0.05,
    "css": 0.05,
    "sql": 0.05,
    "shell": 0.05,
    "other": 0.05,
}

# Framework distribution
FRAMEWORKS = {
    "none": 0.15,
    "react": 0.12,
    "nextjs": 0.10,
    "spring-boot": 0.12,
    "django": 0.06,
    "fastapi": 0.06,
    "express": 0.08,
    "kotlin-android": 0.08,
    "tailwind-css": 0.08,
    "material-ui": 0.07,
    "flutter": 0.04,
    "flask": 0.04,
}

# Task types
TASK_TYPES = {
    "tool_call": 0.15,
    "code_generation": 0.20,
    "code_explanation": 0.12,
    "code_debug": 0.12,
    "code_refactor": 0.10,
    "code_review": 0.08,
    "git_operation": 0.08,
    "system_command": 0.08,
    "devops": 0.07,
}

# Difficulty distribution
DIFFICULTIES = {
    "easy": 0.35,
    "medium": 0.45,
    "hard": 0.20,
}

# Category distribution
CATEGORIES = {
    "backend": 0.25,
    "frontend": 0.20,
    "fullstack": 0.15,
    "devops": 0.15,
    "database": 0.08,
    "mobile": 0.07,
    "system": 0.05,
    "general": 0.05,
}

# ─────────────────────────────────────────────
# COMPLETE TOOL DEFINITIONS
# ─────────────────────────────────────────────

# File Operations Tools
FILE_TOOLS = {
    "read": {
        "description": "Read file content",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
            "offset": {"type": "int", "required": False, "default": 1},
            "limit": {"type": "int", "required": False, "default": 100},
        }
    },
    "write": {
        "description": "Write content to file",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
            "content": {"type": "string", "required": True, "example": "# content"},
            "create_dirs": {"type": "bool", "required": False, "default": False},
        }
    },
    "edit": {
        "description": "Edit file content",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
            "old_string": {"type": "string", "required": True, "example": "# old"},
            "new_string": {"type": "string", "required": True, "example": "# new"},
        }
    },
    "copy": {
        "description": "Copy file or directory",
        "arguments": {
            "source": {"type": "string", "required": True, "example": "src"},
            "destination": {"type": "string", "required": True, "example": "backup/src"},
        }
    },
    "move": {
        "description": "Move file or directory",
        "arguments": {
            "source": {"type": "string", "required": True, "example": "src"},
            "destination": {"type": "string", "required": True, "example": "new_location/src"},
        }
    },
    "delete": {
        "description": "Delete file or directory",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "temp.txt"},
            "recursive": {"type": "bool", "required": False, "default": False},
        }
    },
    "mkdir": {
        "description": "Create directory",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/components"},
            "parents": {"type": "bool", "required": False, "default": True},
        }
    },
    "stat": {
        "description": "Get file/directory info",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
        }
    },
    "exists": {
        "description": "Check if path exists",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
        }
    },
    "list": {
        "description": "List directory contents",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src"},
            "all": {"type": "bool", "required": False, "default": False},
        }
    },
}

# Search Tools
SEARCH_TOOLS = {
    "grep": {
        "description": "Search in files",
        "arguments": {
            "pattern": {"type": "string", "required": True, "example": "TODO"},
            "path": {"type": "string", "required": False, "example": "src"},
            "file_pattern": {"type": "string", "required": False, "example": "*.py"},
            "case_insensitive": {"type": "bool", "required": False, "default": False},
        }
    },
    "find": {
        "description": "Find files/directories",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src"},
            "name": {"type": "string", "required": False, "example": "*.java"},
            "type": {"type": "string", "required": False, "enum": ["f", "d"]},
            "max_depth": {"type": "int", "required": False, "default": 10},
        }
    },
    "glob": {
        "description": "Find files by pattern",
        "arguments": {
            "pattern": {"type": "string", "required": True, "example": "src/**/*.py"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
}

# System Commands Tools
SYSTEM_TOOLS = {
    "bash": {
        "description": "Execute bash command",
        "arguments": {
            "command": {"type": "string", "required": True, "example": "ls -la"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
            "timeout": {"type": "int", "required": False, "default": 30},
            "env": {"type": "object", "required": False},
        }
    },
    "vim": {
        "description": "Edit file with vim",
        "arguments": {
            "path": {"type": "string", "required": True, "example": "src/main.py"},
            "content": {"type": "string", "required": False, "example": "# content"},
            "mode": {"type": "string", "required": False, "enum": ["normal", "insert"], "default": "normal"},
        }
    },
    "top": {
        "description": "Show system processes",
        "arguments": {
            "process_count": {"type": "int", "required": False, "default": 10},
            "sort_by": {"type": "string", "required": False, "enum": ["cpu", "memory"], "default": "cpu"},
        }
    },
    "ps": {
        "description": "List running processes",
        "arguments": {
            "aux": {"type": "bool", "required": False, "default": True},
            "filter": {"type": "string", "required": False, "example": "python"},
        }
    },
    "kill": {
        "description": "Terminate process",
        "arguments": {
            "pid": {"type": "int", "required": True, "example": 1234},
            "signal": {"type": "string", "required": False, "default": "SIGTERM"},
        }
    },
    "df": {
        "description": "Show disk usage",
        "arguments": {
            "path": {"type": "string", "required": False, "example": "/"},
            "human_readable": {"type": "bool", "required": False, "default": True},
        }
    },
    "du": {
        "description": "Show directory size",
        "arguments": {
            "path": {"type": "string", "required": False, "example": "src"},
            "max_depth": {"type": "int", "required": False, "default": 1},
        }
    },
    "free": {
        "description": "Show memory usage",
        "arguments": {
            "human_readable": {"type": "bool", "required": False, "default": True},
        }
    },
    "uptime": {
        "description": "System uptime",
        "arguments": {}
    },
    "uname": {
        "description": "System information",
        "arguments": {
            "all": {"type": "bool", "required": False, "default": True},
        }
    },
    "whoami": {
        "description": "Current user",
        "arguments": {}
    },
}

# Git Tools
GIT_TOOLS = {
    "git_status": {
        "description": "Show git status",
        "arguments": {
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_add": {
        "description": "Stage files",
        "arguments": {
            "files": {"type": "array", "required": False, "example": ["file1.py", "file2.py"]},
            "all": {"type": "bool", "required": False, "default": False},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_commit": {
        "description": "Commit changes",
        "arguments": {
            "message": {"type": "string", "required": True, "example": "feat: add user service"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_pull": {
        "description": "Pull from remote",
        "arguments": {
            "remote": {"type": "string", "required": False, "default": "origin"},
            "branch": {"type": "string", "required": False, "example": "main"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_push": {
        "description": "Push to remote",
        "arguments": {
            "remote": {"type": "string", "required": False, "default": "origin"},
            "branch": {"type": "string", "required": False, "example": "main"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_branch": {
        "description": "Manage branches",
        "arguments": {
            "list": {"type": "bool", "required": False, "default": True},
            "create": {"type": "string", "required": False, "example": "feature/new-feature"},
            "delete": {"type": "string", "required": False, "example": "old-branch"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_stash": {
        "description": "Stash changes",
        "arguments": {
            "action": {"type": "string", "required": False, "enum": ["push", "pop", "list", "drop"], "default": "push"},
            "message": {"type": "string", "required": False, "example": "WIP: feature"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_log": {
        "description": "Show commit history",
        "arguments": {
            "path": {"type": "string", "required": False, "example": "src/main.py"},
            "limit": {"type": "int", "required": False, "default": 10},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_diff": {
        "description": "Show changes",
        "arguments": {
            "path": {"type": "string", "required": False, "example": "src/main.py"},
            "cached": {"type": "bool", "required": False, "default": False},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_checkout": {
        "description": "Switch branches",
        "arguments": {
            "branch": {"type": "string", "required": False, "example": "feature/new"},
            "file": {"type": "string", "required": False, "example": "src/main.py"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git_merge": {
        "description": "Merge branches",
        "arguments": {
            "branch": {"type": "string", "required": True, "example": "feature/new"},
            "no_ff": {"type": "bool", "required": False, "default": False},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "git": {
        "description": "Execute git command",
        "arguments": {
            "command": {"type": "string", "required": True, "example": "status"},
            "args": {"type": "array", "required": False, "example": ["--short"]},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
}

# Date/Time Tools
DATETIME_TOOLS = {
    "datetime": {
        "description": "Get date/time info",
        "arguments": {
            "action": {"type": "string", "required": False, "enum": ["now", "today", "yesterday", "tomorrow"], "default": "now"},
            "format": {"type": "string", "required": False, "example": "%Y-%m-%d %H:%M:%S"},
            "timezone": {"type": "string", "required": False, "example": "UTC"},
        }
    },
    "date_now": {
        "description": "Current date/time",
        "arguments": {
            "format": {"type": "string", "required": False, "example": "%Y-%m-%d %H:%M:%S"},
            "timezone": {"type": "string", "required": False, "example": "UTC"},
        }
    },
    "date_yesterday": {
        "description": "Yesterday's date",
        "arguments": {
            "format": {"type": "string", "required": False, "example": "%Y-%m-%d"},
            "timezone": {"type": "string", "required": False, "example": "UTC"},
        }
    },
    "date_tomorrow": {
        "description": "Tomorrow's date",
        "arguments": {
            "format": {"type": "string", "required": False, "example": "%Y-%m-%d"},
            "timezone": {"type": "string", "required": False, "example": "UTC"},
        }
    },
    "date_add": {
        "description": "Add time to date",
        "arguments": {
            "date": {"type": "string", "required": False, "example": "2024-01-01"},
            "days": {"type": "int", "required": False, "example": 7},
            "months": {"type": "int", "required": False, "example": 1},
            "format": {"type": "string", "required": False, "example": "%Y-%m-%d"},
        }
    },
    "date_diff": {
        "description": "Calculate date difference",
        "arguments": {
            "date1": {"type": "string", "required": True, "example": "2024-01-01"},
            "date2": {"type": "string", "required": True, "example": "2024-01-15"},
        }
    },
}

# Web Tools
WEB_TOOLS = {
    "web_search": {
        "description": "Search the web",
        "arguments": {
            "query": {"type": "string", "required": True, "example": "Spring Boot best practices"},
            "engine": {"type": "string", "required": False, "example": "google"},
            "max_results": {"type": "int", "required": False, "default": 5},
        }
    },
    "fetch": {
        "description": "HTTP request",
        "arguments": {
            "url": {"type": "string", "required": True, "example": "https://api.example.com"},
            "method": {"type": "string", "required": False, "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
            "headers": {"type": "object", "required": False},
            "body": {"type": "string", "required": False},
        }
    },
    "curl": {
        "description": "Curl request",
        "arguments": {
            "url": {"type": "string", "required": True, "example": "https://api.example.com"},
            "method": {"type": "string", "required": False, "example": "GET"},
            "data": {"type": "string", "required": False},
            "headers": {"type": "string", "required": False, "example": "Content-Type: application/json"},
        }
    },
    "ping": {
        "description": "Ping host",
        "arguments": {
            "host": {"type": "string", "required": True, "example": "google.com"},
            "count": {"type": "int", "required": False, "default": 4},
        }
    },
}

# MCP Tools
MCP_TOOLS = {
    "web_search": {
        "description": "Search the web via MCP",
        "arguments": {
            "query": {"type": "string", "required": True, "example": "React best practices"},
            "max_results": {"type": "int", "required": False, "default": 5},
        }
    },
    "fetch_web_content": {
        "description": "Fetch web content",
        "arguments": {
            "url": {"type": "string", "required": True, "example": "https://docs.example.com"},
            "prompt": {"type": "string", "required": True, "example": "Extract main content"},
        }
    },
    "bash_command": {
        "description": "Run bash command via MCP",
        "arguments": {
            "command": {"type": "string", "required": True, "example": "ls -la"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
            "timeout": {"type": "int", "required": False, "default": 60},
        }
    },
    "grep_search": {
        "description": "Search in files via MCP",
        "arguments": {
            "pattern": {"type": "string", "required": True, "example": "TODO"},
            "path": {"type": "string", "required": False, "example": "src"},
            "file_pattern": {"type": "string", "required": False, "example": "*.py"},
        }
    },
    "glob_files": {
        "description": "Find files via MCP",
        "arguments": {
            "pattern": {"type": "string", "required": True, "example": "src/**/*.py"},
            "working_dir": {"type": "string", "required": False, "example": "/project"},
        }
    },
    "understand_image": {
        "description": "Analyze image",
        "arguments": {
            "image_source": {"type": "string", "required": True, "example": "screenshot.png"},
            "prompt": {"type": "string", "required": True, "example": "Describe this image"},
        }
    },
    "webclaw_crawl": {
        "description": "Crawl webpage",
        "arguments": {
            "url": {"type": "string", "required": True, "example": "https://example.com"},
            "prompt": {"type": "string", "required": True, "example": "Extract data"},
        }
    },
}

# Combine all tools for easy selection
ALL_STANDARD_TOOLS = {
    **FILE_TOOLS,
    **SEARCH_TOOLS,
    **SYSTEM_TOOLS,
    **GIT_TOOLS,
    **DATETIME_TOOLS,
    **WEB_TOOLS,
}

TOOL_CATEGORIES = {
    "file": list(FILE_TOOLS.keys()),
    "search": list(SEARCH_TOOLS.keys()),
    "system": list(SYSTEM_TOOLS.keys()),
    "git": list(GIT_TOOLS.keys()),
    "datetime": list(DATETIME_TOOLS.keys()),
    "web": list(WEB_TOOLS.keys()),
}

# ─────────────────────────────────────────────
# AGENT ROLES & CAPABILITIES
# ─────────────────────────────────────────────

AGENT_ROLES = {
    "Senior Java Backend Developer": [
        "Expert in Java, Spring Boot, JPA/Hibernate, REST APIs, Microservices",
        "Strong in debugging, refactoring, and clean code principles",
        "Writes comprehensive unit tests with JUnit 5 and Mockito",
    ],
    "Senior React Developer": [
        "Expert in React, TypeScript, React Hooks, Redux, React Query",
        "Strong in component design, state management, and performance optimization",
        "Writes tests with React Testing Library and Jest",
    ],
    "Full Stack Engineer": [
        "Expert in Node.js, Express, React, PostgreSQL, Docker, AWS",
        "Strong in API design, database architecture, and CI/CD pipelines",
        "Writes full-stack applications with TypeScript",
    ],
    "Python Backend Developer": [
        "Expert in Python, Django, FastAPI, SQLAlchemy, Celery",
        "Strong in REST API development, data pipelines, and automation",
        "Writes tests with pytest and coverage reports",
    ],
    "Android Developer": [
        "Expert in Kotlin, Jetpack Compose, MVVM, Hilt, Coroutines, Room",
        "Strong in mobile UI/UX, offline-first architecture, and performance",
        "Writes instrumented and unit tests for Android",
    ],
    "DevOps Engineer": [
        "Expert in Kubernetes, Docker, Terraform, CI/CD, AWS/GCP, Linux",
        "Strong in infrastructure as code, monitoring, and incident response",
        "Automates deployments and manages cloud-native applications",
    ],
    "Next.js Developer": [
        "Expert in Next.js, React, TypeScript, Tailwind CSS, Prisma, Vercel",
        "Strong in SSR/SSG, API routes, and server components",
        "Writes integration and unit tests for web applications",
    ],
    "Frontend Engineer": [
        "Expert in JavaScript, TypeScript, React, CSS, Tailwind, Material UI",
        "Strong in responsive design, accessibility, and cross-browser compatibility",
        "Writes component tests and E2E tests with Playwright",
    ],
    "Data Engineer": [
        "Expert in Python, Spark, Airflow, Kafka, PostgreSQL, MongoDB",
        "Strong in ETL pipelines, data warehousing, and streaming analytics",
        "Builds scalable data platforms and ML pipelines",
    ],
    "Security Engineer": [
        "Expert in secure coding, SAST/DAST, OWASP Top 10, penetration testing",
        "Strong in authentication, authorization, and encryption best practices",
        "Reviews code for security vulnerabilities and writes security tests",
    ],
    "System Administrator": [
        "Expert in Linux, Bash, Shell scripting, systemd, nginx, Apache",
        "Strong in server management, monitoring, and automation",
        "Manages cloud infrastructure and containerized applications",
    ],
    "ML Engineer": [
        "Expert in Python, PyTorch, TensorFlow, scikit-learn, MLflow",
        "Strong in model training, evaluation, and deployment",
        "Builds end-to-end machine learning pipelines",
    ],
}

AGENT_INSTRUCTIONS = [
    "You are a CLI coding assistant that helps developers write, debug, refactor, and review code.",
    "You are an expert CLI assistant. Provide accurate, well-structured code with clear explanations.",
    "You are a coding agent CLI. Help users with programming tasks, following best practices.",
    "You are a CLI developer assistant. Break down complex problems into simple steps.",
    "You are an expert coding CLI. Write production-quality code with proper error handling.",
]

GUARDRAIL_SETS = [
    [
        "Never delete files without explicit user confirmation",
        "Always show diffs before applying any changes",
        "Never execute destructive commands without asking",
    ],
    [
        "Never commit secrets or credentials to version control",
        "Always use parameterized queries to prevent SQL injection",
        "Never disable security checks in production code",
        "Always write tests alongside new code",
    ],
    [
        "Focus on clean, readable code over clever hacks",
        "Prefer explicit over implicit",
        "Always consider accessibility in UI code",
        "Document complex logic with comments",
    ],
]

STRICT_RULES_SETS = [
    [
        "Always use the correct tool for the task",
        "Return ONLY the JSON response, no additional text",
        "For file operations, use the appropriate file tool",
        "For git operations, use the git tool",
        "For web searches, use web_search tool",
        "For system info, use the appropriate system tool",
        "For date/time queries, use datetime tool",
        "Follow CLI best practices for each command",
    ],
    [
        "Use the most specific tool available for the operation",
        "Return only the JSON object, never add explanations outside JSON",
        "File reads use read tool, file writes use write tool",
        "Git operations use git_* tools, not bash for git commands",
        "Web requests use web_search or fetch tools",
        "System monitoring uses top, df, free tools",
        "Date/time operations use datetime tool",
        "Validate all paths before file operations",
    ],
    [
        "Tool selection must match the operation type exactly",
        "Output format must be valid JSON only",
        "File tools for file operations, git tools for version control",
        "Web tools for network requests, system tools for OS operations",
        "DateTime tools for time-related queries",
        "Always specify complete arguments for the selected tool",
        "Use case-insensitive matching for tool names when needed",
    ],
]

BOT_TONES = ["professional", "friendly", "technical", "concise", "helpful"]

# ─────────────────────────────────────────────
# TASK TEMPLATES
# ─────────────────────────────────────────────

CODE_TASKS = {
    "python": {
        "easy": [
            "Write a function to reverse a string in Python",
            "Create a simple calculator function",
            "Write a function to check if a number is prime",
            "Create a list comprehension to filter even numbers",
        ],
        "medium": [
            "Write a REST API endpoint using FastAPI for user management",
            "Create a decorator that caches function results with TTL",
            "Implement a context manager for database connections",
        ],
        "hard": [
            "Implement a distributed rate limiter using Redis",
            "Create a custom ORM with relationship handling",
            "Build a message queue consumer with dead-letter queue support",
        ],
    },
    "java": {
        "easy": [
            "Create a simple POJO class for a User entity",
            "Write a method to calculate factorial using recursion",
            "Create an interface and its implementation",
        ],
        "medium": [
            "Write a REST controller with validation annotations",
            "Create a service layer with transaction management",
            "Implement a repository with custom JPA queries",
        ],
        "hard": [
            "Implement a Spring Security authentication filter chain",
            "Create an event-driven architecture with Spring ApplicationEvents",
            "Build a reactive REST API using Spring WebFlux",
        ],
    },
    "javascript": {
        "easy": [
            "Write a function to debounce another function",
            "Create a deep clone utility function",
            "Write a function to merge two objects",
        ],
        "medium": [
            "Create a custom React hook for data fetching",
            "Write a middleware function for Express.js",
            "Implement a promise-based retry mechanism",
        ],
        "hard": [
            "Build a virtual DOM implementation from scratch",
            "Create a Redux middleware for async actions",
            "Implement a reactive state system with proxies",
        ],
    },
    "typescript": {
        "easy": [
            "Write a generic function to swap two values",
            "Create an interface for a User with optional fields",
            "Write a type guard function",
        ],
        "medium": [
            "Write a generic API client with typed responses",
            "Create a form validation schema with Zod",
            "Implement a typed event emitter with generics",
        ],
        "hard": [
            "Build a type-safe SQL query builder",
            "Implement a compile-time validation system",
            "Create a DSL for building type-safe APIs",
        ],
    },
    "kotlin": {
        "easy": [
            "Create a data class for a Book entity",
            "Write a lambda expression to filter a list",
            "Create an extension function for String",
        ],
        "medium": [
            "Build a ViewModel with StateFlow for Jetpack Compose",
            "Create a repository with caching using Room",
            "Implement Hilt dependency injection setup",
        ],
        "hard": [
            "Build a Kotlin Multiplatform library with iOS support",
            "Implement a custom lint rule for code quality",
            "Build a real-time chat app with WebSockets",
        ],
    },
}

# Tool-specific user messages
TOOL_TASKS = {
    # File tools
    "read": [
        "Read the package.json file",
        "Show me the contents of App.js",
        "Read the main configuration file",
        "Display the user service file",
        "Look at the test file",
    ],
    "write": [
        "Write this content to a new file",
        "Create a new configuration file",
        "Save the API response to a JSON file",
    ],
    "edit": [
        "Update the CORS configuration",
        "Fix the typo in the main file",
        "Replace the old import with the new one",
    ],
    "copy": [
        "Copy the src directory to backup",
        "Duplicate the template file",
        "Copy the config to all environments",
    ],
    "move": [
        "Move the file to the components folder",
        "Rename the directory from old to new",
        "Relocate the assets to public folder",
    ],
    "delete": [
        "Remove the temporary files",
        "Delete the old backup directory",
        "Clean up the build artifacts",
    ],
    "mkdir": [
        "Create a new components directory",
        "Make the utils folder",
        "Create nested directories for the module",
    ],
    "list": [
        "List all files in src",
        "Show directory contents",
        "List the project root",
    ],
    # Search tools
    "grep": [
        "Search for TODO comments",
        "Find all console.log statements",
        "Search for the UserService class",
        "Find deprecated API usages",
    ],
    "find": [
        "Find all .java files",
        "Find the configuration files",
        "Search for test files",
    ],
    "glob": [
        "Find all JavaScript files",
        "Get all TypeScript files",
        "List all test files",
    ],
    # System tools
    "bash": [
        # Python commands
        "Run pip install -r requirements.txt",
        "Execute pytest tests",
        "Run python manage.py runserver",
        "Start FastAPI with uvicorn",
        "Run Django migrations",
        "Execute Python linter",
        "Run black formatter",
        "Run mypy type checker",
        "Execute Flask app",
        "Run Celery worker",
        # Java/Gradle commands
        "Run Gradle build",
        "Execute Maven test",
        "Run ./gradlew bootRun",
        "Compile with javac",
        "Run Spring Boot app",
        "Execute JAR file",
        "Run Gradle clean build",
        "Compile Kotlin code",
        "Run JUnit tests",
        # JavaScript/Node commands
        "Run npm install",
        "Execute npm test",
        "Run npm run dev",
        "Start Next.js dev server",
        "Build React app",
        "Run npm run build",
        "Execute npm start",
        "Run yarn install",
        "Execute npm run lint",
        "Start Express server",
        # TypeScript commands
        "Run npx tsc --build",
        "Execute TypeScript compiler",
        "Run npm run dev",
        "Build TypeScript project",
        "Check types with tsc",
        # Go commands
        "Run go mod download",
        "Execute go build",
        "Run go test ./...",
        "Start Go server",
        "Run gofmt formatter",
        # Rust commands
        "Run cargo build",
        "Execute cargo run",
        "Run cargo test",
        "Build release with cargo",
        "Run cargo clippy",
        # Kotlin commands
        "Run ./gradlew build",
        "Execute Kotlin compiler",
        "Run ./gradlew test",
        "Start Kotlin app",
        # Docker commands
        "Run docker build",
        "Execute docker-compose up",
        "Build Docker image",
        "Run docker-compose down",
        # Kubernetes commands
        "Apply kubectl config",
        "Get pod status",
        "View Kubernetes logs",
        # General commands
        "Check git status",
        "Run the linter",
        "Execute shell script",
        "Run make build",
    ],
    "top": [
        "Show top processes by CPU",
        "Check memory usage",
        "Display system load",
    ],
    "ps": [
        "List all Python processes",
        "Show running Node processes",
        "Check Java processes",
    ],
    "df": [
        "Check disk usage",
        "Show disk space",
        "Check available storage",
    ],
    "free": [
        "Show memory usage",
        "Check RAM availability",
    ],
    "uptime": [
        "Check system uptime",
        "Show system load",
    ],
    "whoami": [
        "Show current user",
        "Check logged in user",
    ],
    # Git tools
    "git_status": [
        "Check git status",
        "Show current branch and changes",
        "What files have been modified",
    ],
    "git_add": [
        "Stage all changes",
        "Add the modified files",
        "Stage specific files for commit",
    ],
    "git_commit": [
        "Commit the changes with a message",
        "Save the current state",
        "Commit with feat: add new feature",
    ],
    "git_pull": [
        "Pull latest changes",
        "Update from remote",
        "Sync with main branch",
    ],
    "git_push": [
        "Push commits to remote",
        "Upload local changes",
        "Push to origin",
    ],
    "git_branch": [
        "List all branches",
        "Create a new feature branch",
        "Show available branches",
    ],
    "git_stash": [
        "Stash current changes",
        "Save work in progress",
        "Stash to switch branches",
    ],
    "git_log": [
        "Show recent commits",
        "View commit history",
        "Check commit timeline",
    ],
    "git_diff": [
        "Show unstaged changes",
        "Check differences",
        "View file changes",
    ],
    "git_checkout": [
        "Switch to main branch",
        "Checkout feature branch",
        "Switch branches",
    ],
    "git_merge": [
        "Merge feature branch",
        "Combine branches",
        "Merge changes",
    ],
    # DateTime tools
    "datetime": [
        "What is the current date and time",
        "Get today's date",
        "Show current timestamp",
    ],
    "date_now": [
        "What time is it now",
        "Current date and time",
        "Show timestamp",
    ],
    "date_yesterday": [
        "What was yesterday's date",
        "Get date for yesterday",
        "Date 24 hours ago",
    ],
    "date_tomorrow": [
        "What is tomorrow's date",
        "Get next day's date",
        "Date 24 hours from now",
    ],
    "date_add": [
        "Add 7 days to today",
        "Calculate future date",
        "Add months to date",
    ],
    "date_diff": [
        "How many days between dates",
        "Calculate date difference",
        "Days between two dates",
    ],
    # Web tools
    "web_search": [
        "Search for React best practices",
        "Find Spring Boot documentation",
        "Search Python async patterns",
        "Look up GitHub Copilot usage",
    ],
    "fetch": [
        "Fetch the API documentation",
        "Get data from endpoint",
        "Request JSON data",
    ],
    "curl": [
        "Curl the health endpoint",
        "Test API with curl",
        "Make HTTP request",
    ],
    "ping": [
        "Ping google.com",
        "Check server connectivity",
        "Test network",
    ],
}

# Response-only tasks
RESPONSE_TASKS = {
    "explanation": [
        "What is the difference between var, let, and const in JavaScript?",
        "Explain the SOLID principles",
        "What is the CAP theorem?",
        "Explain microservices architecture",
    ],
    "review": [
        "Review this code for security issues",
        "Suggest performance improvements",
        "Check for SOLID violations",
    ],
    "comparison": [
        "Compare React Query vs SWR",
        "PostgreSQL vs MySQL",
        "Docker vs Kubernetes",
    ],
    "troubleshooting": [
        "Why CORS error in API calls?",
        "React component re-rendering too often",
        "Database query slow",
    ],
}

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def weighted_choice(weights_dict):
    """Select a random key based on weighted probabilities."""
    items = list(weights_dict.items())
    keys = [k for k, _ in items]
    weights = [w for _, w in items]
    return random.choices(keys, weights=weights, k=1)[0]


def pick_n(arr, n):
    """Pick n random items from array."""
    return random.sample(arr, min(n, len(arr)))


def generate_call_id():
    """Generate a unique call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def generate_mcp_id():
    """Generate a unique MCP call ID."""
    return f"mcp_{uuid.uuid4().hex[:8]}"


def format_capabilities(capabilities):
    """Format capabilities as newline-separated string."""
    return "\n".join(f"- {c}" for c in capabilities)


def format_guardrails(guardrails):
    """Format guardrails as newline-separated string."""
    return "\n".join(f"- {g}" for g in guardrails)


# Memory templates
MEMORY_TEMPLATES = [
    """- Project: {project_name}
- Language: {language}
- Framework: {framework}
- Current file: {current_file}
- Last modified: {last_modified}""",
    """- Project context: {project_desc}
- Working directory: {working_dir}
- Tech stack: {language} with {framework}
- Active files: {active_files}""",
    """- Session started: {session_time}
- User: {user_name}
- Project: {project_name}
- Current task: {current_task}""",
    """- Project: {project_name}
- Environment: {env}
- Branch: {branch}
- Last commit: {last_commit}""",
]

# Conversation history templates
CONVERSATION_HISTORY_TEMPLATES = [
    """user: Can you help me with this project?
assistant: I'll help you with that. Let me check the project first.""",
    """user: Show me the current file structure
assistant: I can see the project structure. What would you like to do?
tool_call: {"name": "list", "arguments": {"path": "src"}}
tool_response: src/main.py, src/utils.py, tests/
assistant: Found the project structure.""",
    """user: What is in the main configuration?
assistant: Let me check the configuration file.
tool_call: {"name": "read", "arguments": {"path": "config.yaml"}}
tool_response: app_name: my-project, version: 1.0.0
assistant: The configuration looks good.""",
    """user: Check the existing code
assistant: I'll analyze the code and provide suggestions.
tool_call: {"name": "read", "arguments": {"path": "src/main.py"}}
tool_response: class Main:\n    def __init__(self):\n        pass
assistant: The code structure is standard.""",
]


def generate_memory(language, framework):
    """Generate a memory block."""
    template = random.choice(MEMORY_TEMPLATES)
    project_names = ["user-service", "api-gateway", "web-app", "ml-pipeline", "data-processor"]
    current_files = {
        "python": "src/main.py",
        "java": "src/main/java/App.java",
        "javascript": "src/App.js",
        "typescript": "src/App.tsx",
        "kotlin": "src/main/kotlin/App.kt",
    }
    frameworks_map = {
        "django": "Django", "fastapi": "FastAPI", "flask": "Flask",
        "spring-boot": "Spring Boot", "express": "Express.js", "react": "React",
        "nextjs": "Next.js", "kotlin-android": "Kotlin Android"
    }

    memory = template.format(
        project_name=random.choice(project_names),
        language=language.title(),
        framework=frameworks_map.get(framework, framework.title()),
        current_file=current_files.get(language, "src/main.txt"),
        last_modified=random.choice(["2 hours ago", "yesterday", "today", "just now"]),
        project_desc=f"{frameworks_map.get(framework, framework)} application",
        working_dir="/project",
        active_files="src/main, tests/, config/",
        session_time="recent",
        user_name="developer",
        current_task="coding assistance",
        env=random.choice(["development", "staging", "production"]),
        branch=random.choice(["main", "feature/new-endpoint", "develop"]),
        last_commit=random.choice(["feat: add endpoint", "fix: resolve bug", "refactor: cleanup"]),
    )
    return memory


def generate_conversation_history(language, task_type):
    """Generate a conversation history block."""
    return random.choice(CONVERSATION_HISTORY_TEMPLATES)


def format_strict_rules(strict_rules):
    """Format strict rules as newline-separated string."""
    return "\n".join(f"- {r}" for r in strict_rules)


def build_prompt(agent_role, capabilities, bot_tone, language, instruction, guardrails, strict_rules, tools, memory, conversation_history, user_message):
    """Build the full structured prompt with memory and conversation history."""
    return f"""<init>
<agent_role>{agent_role}</agent_role>
<agent_capabilities>
{format_capabilities(capabilities)}
</agent_capabilities>
<localization>
  <bot_tone>{bot_tone}</bot_tone>
  <language>{language}</language>
</localization>
</init>
<agent_instruction>{instruction}</agent_instruction>
<guardrails>
{format_guardrails(guardrails)}
</guardrails>
<strict_rules>
{format_strict_rules(strict_rules)}
</strict_rules>
<humanize>false</humanize>
<tools>
  <list>{json.dumps(tools)}</list>
  <tool-call-format>json</tool-call-format>
</tools>
<memory>
{memory}
</memory>
<conversation-history>
{conversation_history}
</conversation-history>

---

User: {user_message}"""


def generate_tool_args(tool_name, language):
    """Generate appropriate arguments for a standard tool."""
    # File tools
    if tool_name == "read":
        file_paths = {
            "python": "src/main.py",
            "java": "src/main/java/App.java",
            "javascript": "src/App.js",
            "typescript": "src/App.tsx",
            "kotlin": "src/main/kotlin/App.kt",
        }
        return {
            "path": file_paths.get(language, "src/file.txt"),
            "offset": 1,
            "limit": random.choice([50, 100, 200]),
        }
    elif tool_name == "write":
        return {
            "path": f"src/new_file.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
            "content": "# New file content",
            "create_dirs": random.choice([True, False]),
        }
    elif tool_name == "edit":
        return {
            "path": "src/main.py",
            "old_string": "# TODO: implement",
            "new_string": "# IMPLEMENTED",
        }
    elif tool_name == "copy":
        return {
            "source": "src",
            "destination": "backup/src",
        }
    elif tool_name == "move":
        return {
            "source": "src/old",
            "destination": "src/new",
        }
    elif tool_name == "delete":
        return {
            "path": "temp/file.txt",
            "recursive": False,
        }
    elif tool_name == "mkdir":
        return {
            "path": "src/new_module",
            "parents": True,
        }
    elif tool_name == "stat":
        return {
            "path": "src/main.py",
        }
    elif tool_name == "exists":
        return {
            "path": "src/main.py",
        }
    elif tool_name == "list":
        return {
            "path": "src",
            "all": False,
        }

    # Search tools
    elif tool_name == "grep":
        return {
            "pattern": random.choice(["TODO", "FIXME", "console.log", "Exception"]),
            "path": "src",
            "file_pattern": f"*.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
            "case_insensitive": False,
        }
    elif tool_name == "find":
        return {
            "path": "src",
            "name": f"*.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
            "type": "f",
            "max_depth": 5,
        }
    elif tool_name == "glob":
        return {
            "pattern": f"src/**/*.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
            "working_dir": "/project",
        }

    # System tools
    elif tool_name == "bash":
        commands = {
            "python": random.choice(["python -m pytest", "python manage.py runserver", "pip install -r requirements.txt"]),
            "java": random.choice(["./mvnw test", "./gradlew build"]),
            "javascript": random.choice(["npm test", "npm run build", "yarn dev"]),
            "typescript": random.choice(["npm test", "npx tsc --noEmit"]),
        }
        return {
            "command": commands.get(language, "ls -la"),
            "working_dir": "/project",
            "timeout": random.choice([30, 60, 120]),
        }
    elif tool_name == "top":
        return {
            "process_count": 10,
            "sort_by": random.choice(["cpu", "memory"]),
        }
    elif tool_name == "ps":
        return {
            "aux": True,
            "filter": random.choice(["python", "java", "node", None]),
        }
    elif tool_name == "kill":
        return {
            "pid": random.randint(1000, 9999),
            "signal": "SIGTERM",
        }
    elif tool_name == "df":
        return {
            "path": "/",
            "human_readable": True,
        }
    elif tool_name == "du":
        return {
            "path": "src",
            "max_depth": 1,
        }
    elif tool_name == "free":
        return {
            "human_readable": True,
        }
    elif tool_name == "uptime":
        return {}
    elif tool_name == "uname":
        return {
            "all": True,
        }
    elif tool_name == "whoami":
        return {}

    # Git tools
    elif tool_name == "git_status":
        return {
            "working_dir": "/project",
        }
    elif tool_name == "git_add":
        return {
            "files": ["src/main.py"],
            "all": random.choice([True, False]),
            "working_dir": "/project",
        }
    elif tool_name == "git_commit":
        return {
            "message": random.choice([
                "feat: add user authentication",
                "fix: resolve null pointer exception",
                "refactor: improve code structure",
                "docs: update README",
                "test: add unit tests",
            ]),
            "working_dir": "/project",
        }
    elif tool_name == "git_pull":
        return {
            "remote": "origin",
            "branch": "main",
            "working_dir": "/project",
        }
    elif tool_name == "git_push":
        return {
            "remote": "origin",
            "branch": "main",
            "working_dir": "/project",
        }
    elif tool_name == "git_branch":
        return {
            "list": True,
            "create": None,
            "delete": None,
            "working_dir": "/project",
        }
    elif tool_name == "git_stash":
        return {
            "action": "push",
            "message": "WIP: feature implementation",
            "working_dir": "/project",
        }
    elif tool_name == "git_log":
        return {
            "path": "src/main.py",
            "limit": 10,
            "working_dir": "/project",
        }
    elif tool_name == "git_diff":
        return {
            "path": "src/main.py",
            "cached": False,
            "working_dir": "/project",
        }
    elif tool_name == "git_checkout":
        return {
            "branch": "feature/new-feature",
            "file": None,
            "working_dir": "/project",
        }
    elif tool_name == "git_merge":
        return {
            "branch": "feature/new-feature",
            "no_ff": False,
            "working_dir": "/project",
        }
    elif tool_name == "git":
        return {
            "command": "status",
            "args": ["--short"],
            "working_dir": "/project",
        }

    # DateTime tools
    elif tool_name == "datetime":
        return {
            "action": random.choice(["now", "today", "yesterday", "tomorrow"]),
            "format": "%Y-%m-%d %H:%M:%S",
            "timezone": random.choice(["UTC", "America/New_York", "Asia/Kolkata"]),
        }
    elif tool_name == "date_now":
        return {
            "format": "%Y-%m-%d %H:%M:%S",
            "timezone": "UTC",
        }
    elif tool_name == "date_yesterday":
        return {
            "format": "%Y-%m-%d",
            "timezone": "UTC",
        }
    elif tool_name == "date_tomorrow":
        return {
            "format": "%Y-%m-%d",
            "timezone": "UTC",
        }
    elif tool_name == "date_add":
        return {
            "days": random.choice([7, 14, 30]),
            "months": None,
            "format": "%Y-%m-%d",
        }
    elif tool_name == "date_diff":
        today = datetime.now()
        other = today - timedelta(days=random.randint(1, 30))
        return {
            "date1": today.strftime("%Y-%m-%d"),
            "date2": other.strftime("%Y-%m-%d"),
        }

    # Web tools
    elif tool_name == "web_search":
        return {
            "query": f"{language} best practices 2024",
            "engine": "google",
            "max_results": random.randint(3, 10),
        }
    elif tool_name == "fetch":
        return {
            "url": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Content-Type": "application/json"},
        }
    elif tool_name == "curl":
        return {
            "url": "https://api.example.com/health",
            "method": "GET",
        }
    elif tool_name == "ping":
        return {
            "host": "google.com",
            "count": 4,
        }

    else:
        return {"value": "example"}


def generate_mcp_args(tool_name, language):
    """Generate appropriate arguments for an MCP tool."""
    if tool_name == "web_search":
        return {
            "query": f"{language} best practices 2024",
            "max_results": random.randint(3, 10),
        }
    elif tool_name == "fetch_web_content":
        return {
            "url": "https://docs.example.com/api",
            "prompt": "Extract main content",
        }
    elif tool_name == "bash_command":
        return {
            "command": random.choice(["ls -la", "npm test", "python -m pytest"]),
            "working_dir": "/project",
            "timeout": 60,
        }
    elif tool_name == "grep_search":
        return {
            "pattern": random.choice(["TODO", "FIXME"]),
            "path": "src",
            "file_pattern": f"*.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
        }
    elif tool_name == "glob_files":
        return {
            "pattern": f"src/**/*.{'py' if language == 'python' else 'java' if language == 'java' else 'js'}",
            "working_dir": "/project",
        }
    elif tool_name == "understand_image":
        return {
            "image_source": "screenshot.png",
            "prompt": "Describe what you see",
        }
    elif tool_name == "webclaw_crawl":
        return {
            "url": "https://example.com/docs",
            "prompt": "Extract all information",
        }
    else:
        return {"query": "example"}


def get_task_for_language(language, difficulty):
    """Get a random task for the given language and difficulty."""
    tasks_by_lang = CODE_TASKS.get(language, CODE_TASKS["python"])
    tasks = tasks_by_lang.get(difficulty, tasks_by_lang["medium"])
    return random.choice(tasks)


def get_framework_for_language(language):
    """Get a framework that matches the language."""
    mapping = {
        "python": ["django", "fastapi", "flask"],
        "java": ["spring-boot"],
        "javascript": ["express", "react"],
        "typescript": ["nextjs", "express"],
        "kotlin": ["kotlin-android"],
        "go": ["none"],
        "rust": ["none"],
    }
    options = mapping.get(language, ["none"])
    return random.choice(options)


# ─────────────────────────────────────────────
# RESPONSE BUILDERS
# ─────────────────────────────────────────────

def build_normal_response(content, language, framework, difficulty, category, humanize=False):
    """Build a standard NORMAL message response."""
    return {
        "message_type": "normal",
        "content": content,
        "tool_call": None,
        "mcp_call": None,
        "turns": None,
        "humanize": humanize,
        "metadata": {
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }


def build_tool_call_response(tool_name, tool_args, language, framework, difficulty, category):
    """Build a standard TOOL_CALL response."""
    return {
        "message_type": "tool_call",
        "content": None,
        "tool_call": {
            "id": generate_call_id(),
            "name": tool_name,
            "arguments": tool_args,
        },
        "mcp_call": None,
        "turns": None,
        "humanize": False,
        "metadata": {
            "tool_provider": "standard",
            "tool_category": get_tool_category(tool_name),
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }


def build_mcp_call_response(tool_name, tool_args, language, framework, difficulty, category):
    """Build a standard MCP_CALL response."""
    return {
        "message_type": "mcp_call",
        "content": None,
        "tool_call": None,
        "mcp_call": {
            "id": generate_mcp_id(),
            "server": "extra_skills_mcp",
            "name": tool_name,
            "arguments": tool_args,
        },
        "turns": None,
        "humanize": False,
        "metadata": {
            "tool_provider": "mcp",
            "mcp_server": "extra_skills_mcp",
            "language": language,
            "framework": framework,
            "difficulty": difficulty,
            "category": category,
        }
    }


def build_multi_turn_response(turns, language, framework, difficulty, category):
    """Build a standard MULTI_TURN response."""
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


def get_tool_category(tool_name):
    """Get the category of a tool."""
    for category, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            return category
    return "unknown"


def build_response_response(task, language, difficulty, framework, category):
    """Build a general response."""
    content = f"""**Task: {task}**

Here's the information you requested:

**Key Points:**
1. This is a common development task
2. Follow best practices for implementation
3. Consider edge cases and error handling

**Recommendation:**
Based on best practices, I recommend the following approach for {language} development."""
    return build_normal_response(content, language, framework, difficulty, category, humanize=False)


# ─────────────────────────────────────────────
# MAIN DATA GENERATOR
# ─────────────────────────────────────────────

def generate_row(row_id):
    """Generate a single training row."""
    # 1. Select message type
    message_type = weighted_choice(MESSAGE_TYPES)

    # 2. Select language and framework
    language = weighted_choice(LANGUAGES)
    framework = get_framework_for_language(language)

    # 3. Select task type and difficulty
    task_type = weighted_choice(TASK_TYPES)
    difficulty = weighted_choice(DIFFICULTIES)

    # 4. Select category
    category = weighted_choice(CATEGORIES)

    # 5. Select agent role
    agent_role = random.choice(list(AGENT_ROLES.keys()))
    capabilities = AGENT_ROLES[agent_role]

    # 6. Select tone, instruction, guardrails, strict_rules
    bot_tone = random.choice(BOT_TONES)
    instruction = random.choice(AGENT_INSTRUCTIONS)
    guardrails = random.choice(GUARDRAIL_SETS)
    strict_rules = random.choice(STRICT_RULES_SETS)

    # 7. Select tools (5-10 random tools from all categories)
    num_tools = random.randint(5, 10)
    all_tool_names = list(ALL_STANDARD_TOOLS.keys())
    tools = pick_n(all_tool_names, num_tools)

    # 8. Generate memory and conversation history
    memory = generate_memory(language, framework)
    conversation_history = generate_conversation_history(language, task_type)

    # 9. Generate user message and response
    user_message = ""
    response = {}

    if message_type == "normal":
        # Normal response - no tool call needed
        response_category = random.choice(list(RESPONSE_TASKS.keys()))
        user_message = random.choice(RESPONSE_TASKS[response_category])
        response = build_response_response(user_message, language, difficulty, framework, category)

    elif message_type == "tool_call":
        # Standard tool call - pick from any standard tool
        tool_name = random.choice(list(ALL_STANDARD_TOOLS.keys()))
        tool_messages = TOOL_TASKS.get(tool_name, ["Perform the operation"])

        if tool_messages:
            user_message = random.choice(tool_messages)
        else:
            # Generate a generic message based on tool
            user_message = f"Use {tool_name} to complete the task"

        tool_args = generate_tool_args(tool_name, language)
        response = build_tool_call_response(tool_name, tool_args, language, framework, difficulty, category)

    elif message_type == "mcp_call":
        # MCP tool call
        tool_name = random.choice(list(MCP_TOOLS.keys()))
        tool_messages = TOOL_TASKS.get(tool_name, ["Search for information"])

        if tool_messages:
            user_message = random.choice(tool_messages)
        else:
            user_message = f"Use MCP to {tool_name.replace('_', ' ')}"

        tool_args = generate_mcp_args(tool_name, language)
        response = build_mcp_call_response(tool_name, tool_args, language, framework, difficulty, category)

    else:  # multi_turn
        # Multi-turn conversation
        num_turns = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
        user_message = get_task_for_language(language, difficulty)

        # Build conversation turns
        turns = []

        # First turn: tool call
        tool_name = random.choice(list(ALL_STANDARD_TOOLS.keys()))
        tool_args = generate_tool_args(tool_name, language)

        turns.append({
            "turn_id": 1,
            "type": "tool_call",
            "tool_call": {
                "id": generate_call_id(),
                "name": tool_name,
                "arguments": tool_args,
            }
        })

        # Second turn: tool response
        if tool_name in GIT_TOOLS:
            response_content = "[Git operation completed successfully]"
        elif tool_name in DATETIME_TOOLS:
            response_content = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif tool_name in SYSTEM_TOOLS:
            response_content = "[System command output]"
        elif tool_name in WEB_TOOLS:
            response_content = "[Web search results]"
        else:
            response_content = "[File operation completed]"

        turns.append({
            "turn_id": 2,
            "type": "tool_response",
            "content": response_content,
        })

        # Third turn: assistant response
        assistant_content = f"""Based on the results, here's my analysis:

**Summary:**
- Operation completed successfully
- Found relevant information in your {language} project

**Next Steps:**
1. Review the results
2. Apply any necessary changes
3. Run tests to verify

Would you like me to analyze any specific file in detail?"""

        turns.append({
            "turn_id": 3,
            "type": "assistant",
            "content": assistant_content,
            "humanize": False,
        })

        # Add extra turns if needed
        if num_turns > 3:
            for i in range(4, num_turns + 1):
                turns.append({
                    "turn_id": i,
                    "type": "assistant",
                    "content": f"Continuing analysis... (turn {i})",
                    "humanize": False,
                })

        response = build_multi_turn_response(turns, language, framework, difficulty, category)

    # Build full prompt with memory and conversation history
    prompt = build_prompt(
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

    # Build output JSON
    output_json = json.dumps(response, ensure_ascii=False)

    # Extract fields from response for CSV columns
    tool_call = response.get("tool_call")
    mcp_call = response.get("mcp_call")

    tool_name = ""
    if tool_call:
        tool_name = tool_call.get("name", "")
    elif mcp_call:
        tool_name = mcp_call.get("name", "")

    tool_provider = "standard" if tool_call else ("mcp" if mcp_call else "none")

    if tool_call:
        tool_args_json = json.dumps(tool_call.get("arguments", {}))
    elif mcp_call:
        tool_args_json = json.dumps(mcp_call.get("arguments", {}))
    else:
        tool_args_json = "{}"

    mcp_server = mcp_call.get("server") if mcp_call else None
    content = response.get("content") or ""
    turns_json = json.dumps(response.get("turns")) if response.get("turns") else ""

    # Build metadata
    metadata = json.dumps({
        "source": "synthetic",
        "generated_by": "dataset-generator-v3",
        "generated_at": datetime.now().isoformat(),
        "version": "3.0",
        "difficulty_score": {"easy": 1, "medium": 2, "hard": 3}[difficulty],
        "task_type": task_type,
    })

    return {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "language": language,
        "framework": framework,
        "task_type": task_type,
        "message_type": message_type,
        "tools_available": json.dumps(tools),
        "output_json": output_json,
        "content": content,
        "tool_name": tool_name,
        "tool_provider": tool_provider,
        "tool_args": tool_args_json,
        "mcp_server": mcp_server or "",
        "turns": turns_json,
        "humanize": str(response.get("humanize", False)).lower(),
        "category": category,
        "difficulty": difficulty,
        "agent_role": agent_role,
        "guardrails": json.dumps(guardrails),
        "strict_rules": json.dumps(strict_rules),
        "metadata": metadata,
    }


def generate_dataset(num_rows, output_file):
    """Generate the full dataset and write to CSV with progress bar."""
    print(f"Generating {num_rows:,} training rows...")
    print(f"Output: {output_file}")
    print(f"Estimated size: ~{num_rows * 3 // 1000}MB")
    print()

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    fieldnames = [
        "id", "prompt", "language", "framework", "task_type",
        "message_type", "tools_available", "output_json",
        "content", "tool_name", "tool_provider", "tool_args",
        "mcp_server", "turns", "humanize", "category",
        "difficulty", "agent_role", "guardrails", "strict_rules", "metadata",
    ]

    import time
    start_time = time.time()

    # Write CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        # Progress iterator
        if has_tqdm:
            iterator = tqdm(range(num_rows), desc="Generating", unit="rows", ncols=80)
        else:
            iterator = range(num_rows)

        for i in iterator:
            row = generate_row(i)
            writer.writerow(row)

    elapsed = time.time() - start_time
    rows_per_sec = num_rows / elapsed if elapsed > 0 else 0

    print()
    print(f"  " + "="*58)
    print(f"  [COMPLETE] Generated: {num_rows:,} rows in {elapsed:.1f}s ({rows_per_sec:,.0f} rows/sec)")
    print(f"  " + "="*58)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Coding Agent Dataset Generator v3")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate (default: 1000)")
    parser.add_argument("--output", type=str, default="training_data.csv", help="Output CSV file")
    parser.add_argument("--full", action="store_true", help="Generate full 5M dataset")

    args = parser.parse_args()

    num_rows = 5_000_000 if args.full else args.rows
    output_file = args.output

    print("=" * 60)
    print("CLI Coding Agent Training Data Generator v3")
    print(f"Model: Gemma 4 E4B (Google) | Framework: Unsloth + LoRA")
    print("Tools: File, System, Git, DateTime, Web + MCP")
    print("Format: Unified JSON with message_type + <strict_rules>")
    print("=" * 60)
    print()

    generate_dataset(num_rows, output_file)

    print()
    print("Dataset ready for training!")
    print(f"Next: Upload to RunPod and run training")
