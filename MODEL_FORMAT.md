# MODEL_FORMAT.md — CLI Coding Agent Fine-tuning

> **Base Model**: Google Gemma 4 E4B (8B)
> **Training Platform**: RunPod (Unsloth + LoRA)
> **Purpose**: Tool calling + multi-language coding CLI agent

---

# PART 1 — INPUT PROMPT FORMAT

The user prompt is always wrapped in a structured format with **memory** and **conversation history**.

```xml
<init>
<agent_role>Senior Java Backend Developer</agent_role>
<agent_capabilities>
- Expert in Java, Spring Boot, JPA/Hibernate, REST APIs, Microservices
- Strong in debugging, refactoring, and clean code principles
- Writes comprehensive unit tests with JUnit 5 and Mockito
</agent_capabilities>
<localization>
  <bot_tone>Professional and helpful</bot_tone>
  <language>en</language>
</localization>
</init>
<agent_instruction>You are a CLI coding assistant that helps developers write, debug, refactor, and review code.</agent_instruction>
<guardrails>
- Never delete files without explicit user confirmation
- Always show diffs before applying any changes
- Never execute destructive commands without asking
</guardrails>
<strict_rules>
- Always use the correct tool for the task
- Return ONLY the JSON response, no additional text
- For file operations, use the appropriate file tool
- For git operations, use the git tool
- For web searches, use web_search tool
- For system info, use the appropriate system tool
- For date/time queries, use datetime tool
</strict_rules>
<humanize>false</humanize>
<tools>
  <list>["read", "write", "bash", "vim", "grep", "find", "git", "datetime", "web_search", "fetch", "copy", "move", "delete"]</list>
  <tool-call-format>json</tool-call-format>
</tools>
<memory>
- Project: user-service (Spring Boot microservice)
- Language: Java 21
- Framework: Spring Boot 3.2
- Current file: src/main/java/com/example/UserController.java
- Last modified: 2 hours ago
</memory>
<conversation-history>
user: Can you create a user registration endpoint?
assistant: I'll help you create a user registration REST endpoint. Let me first check the existing User entity structure.
tool_call: {"name": "read", "arguments": {"path": "src/main/java/com/example/User.java"}}
tool_response: public class User { private Long id; private String email; private String name; }
assistant: I see the User entity. It has id, email, and name fields. Now I'll create the registration endpoint.
</conversation-history>

---

User: Add password field to the registration endpoint
```

### Variable Fields in Prompt Template

| Field | Description | Example |
|-------|-------------|---------|
| `agent_role` | The persona/role of the agent | "Senior React Developer" |
| `agent_capabilities` | Bullet list of skills | "Expert in Python, Django, FastAPI..." |
| `bot_tone` | Communication style | `professional`, `friendly`, `technical` |
| `language` | Output language code | `en`, `hi`, `es` |
| `guardrails` | Safety rules | 3-5 bullet rules |
| `humanize` | `true` or `false` | Whether to sound human |
| `tools.list` | Available tool names as array | JSON array of strings |
| `tool-call-format` | Tool call output format | `json`, `xml`, `mcp` |
| `memory` | Agent's memory/context | Project info, current file, recent actions |
| `conversation-history` | Prior turns | user/assistant exchanges with tool calls |
| `User:` | The actual user message | Any coding task |

---

# PART 2 — STANDARDIZED OUTPUT FORMAT

All model responses use a **unified JSON structure**. This makes parsing easy for developers.

## Response Object Structure

```json
{
  "message_type": "normal | tool_call | mcp_call | multi_turn",
  "content": "...",
  "tool_call": null | { ... },
  "mcp_call": null | { ... },
  "turns": null | [ ... ],
  "humanize": false,
  "metadata": { ... }
}
```

---

## FORMAT 1: NORMAL MESSAGE (No Tool Call)

Used for conversational responses, code explanations, direct answers.

```json
{
  "message_type": "normal",
  "content": "Here is a clean implementation of a User Service in Spring Boot:\n\n```java\n@Service\npublic class UserService {\n    private final UserRepository userRepository;\n    private final PasswordEncoder passwordEncoder;\n\n    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {\n        this.userRepository = userRepository;\n        this.passwordEncoder = passwordEncoder;\n    }\n\n    public User registerUser(RegistrationRequest request) {\n        if (userRepository.existsByEmail(request.getEmail())) {\n            throw new EmailAlreadyExistsException(request.getEmail());\n        }\n        User user = new User();\n        user.setEmail(request.getEmail());\n        user.setPassword(passwordEncoder.encode(request.getPassword()));\n        return userRepository.save(user);\n    }\n}\n```\n\n**Key Design Decisions:**\n1. **Constructor Injection** — Spring recommended approach\n2. **Password Encoding** — Never store plain passwords\n3. **Email Uniqueness Check** — Prevents duplicate registrations",
  "tool_call": null,
  "mcp_call": null,
  "turns": null,
  "humanize": false,
  "metadata": {
    "language": "java",
    "framework": "spring-boot",
    "difficulty": "medium",
    "category": "backend"
  }
}
```

---

## FORMAT 2: TOOL CALL (Standard CLI Tools)

Used for calling built-in CLI tools like `read_file`, `bash_command`, `grep_search`.

```json
{
  "message_type": "tool_call",
  "content": null,
  "tool_call": {
    "id": "call_abc123",
    "name": "read_file",
    "arguments": {
      "path": "src/main/java/com/example/UserController.java",
      "offset": 1,
      "limit": 100
    }
  },
  "mcp_call": null,
  "turns": null,
  "humanize": false,
  "metadata": {
    "tool_provider": "standard",
    "language": "java",
    "framework": "spring-boot",
    "difficulty": "easy"
  }
}
```

### Standard Tool Schema

#### File Operations
| Tool | Arguments | Description |
|------|-----------|-------------|
| `read` | `path`, `offset`, `limit` | Read file content |
| `write` | `path`, `content`, `create_dirs` | Write content to file |
| `edit` | `path`, `old_string`, `new_string` | Edit file content |
| `copy` | `source`, `destination` | Copy file or directory |
| `move` | `source`, `destination` | Move file or directory |
| `delete` | `path`, `recursive` | Delete file or directory |
| `mkdir` | `path`, `parents` | Create directory |
| `stat` | `path` | Get file/directory info |
| `exists` | `path` | Check if path exists |
| `list` | `path`, `all` | List directory contents |

#### Search Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `grep` | `pattern`, `path`, `file_pattern`, `case_insensitive` | Search in files |
| `find` | `path`, `name`, `type`, `max_depth` | Find files/directories |
| `glob` | `pattern`, `working_dir` | Find files by pattern |

#### System Commands
| Tool | Arguments | Description |
|------|-----------|-------------|
| `bash` | `command`, `working_dir`, `timeout`, `env` | Execute bash command |
| `vim` | `path`, `content`, `mode` | Edit file with vim |
| `top` | `process_count`, `sort_by` | Show system processes |
| `ps` | `aux`, `filter` | List running processes |
| `kill` | `pid`, `signal` | Terminate process |
| `df` | `path`, `human_readable` | Show disk usage |
| `du` | `path`, `max_depth` | Show directory size |
| `free` | `human_readable` | Show memory usage |
| `uptime` | - | System uptime |
| `uname` | `all` | System information |
| `whoami` | - | Current user |

#### Git Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `git` | `command`, `args`, `working_dir` | Execute git command |
| `git_status` | `working_dir` | Show git status |
| `git_add` | `files`, `all`, `working_dir` | Stage files |
| `git_commit` | `message`, `working_dir` | Commit changes |
| `git_pull` | `remote`, `branch`, `working_dir` | Pull from remote |
| `git_push` | `remote`, `branch`, `working_dir` | Push to remote |
| `git_branch` | `list`, `create`, `delete`, `working_dir` | Manage branches |
| `git_stash` | `action`, `message`, `working_dir` | Stash changes |
| `git_log` | `path`, `limit` | Show commit history |
| `git_diff` | `path`, `cached`, `working_dir` | Show changes |
| `git_checkout` | `branch`, `file`, `working_dir` | Switch branches |
| `git_merge` | `branch`, `no_ff`, `working_dir` | Merge branches |

#### Date/Time Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `datetime` | `action`, `format`, `timezone` | Get date/time info |
| `date_now` | `format`, `timezone` | Current date/time |
| `date_yesterday` | `format`, `timezone` | Yesterday's date |
| `date_tomorrow` | `format`, `timezone` | Tomorrow's date |
| `date_add` | `days`, `months`, `format` | Add time |
| `date_diff` | `date1`, `date2` | Calculate difference |

#### Web Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `web_search` | `query`, `engine`, `max_results` | Search the web |
| `fetch` | `url`, `method`, `headers`, `body` | HTTP request |
| `curl` | `url`, `method`, `data`, `headers` | Curl request |
| `ping` | `host`, `count` | Ping host |

#### Process Management
| Tool | Arguments | Description |
|------|-----------|-------------|
| `systemctl` | `action`, `service` | Systemd control |
| `service` | `action`, `name` | Service management |

---

## FORMAT 3: MCP CALL (MCP Server Tools)

Used for calling MCP server tools (port 7710 `extra_skills_mcp`).

```json
{
  "message_type": "mcp_call",
  "content": null,
  "tool_call": null,
  "mcp_call": {
    "id": "mcp_xyz789",
    "server": "extra_skills_mcp",
    "name": "web_search",
    "arguments": {
      "query": "Spring Boot 3 user registration best practices 2024",
      "max_results": 5
    }
  },
  "turns": null,
  "humanize": false,
  "metadata": {
    "tool_provider": "mcp",
    "mcp_server": "extra_skills_mcp",
    "language": "java",
    "framework": "spring-boot"
  }
}
```

### MCP Tool Schema

| MCP Tool | Arguments | Description |
|----------|-----------|-------------|
| `web_search` | `query`, `max_results` | Search the web |
| `fetch_web_content` | `url`, `prompt` | Extract content from URL |
| `bash_command` | `command`, `working_dir`, `timeout` | Run shell command |
| `run_command` | `command`, `args`, `env` | Run system command |
| `grep_search` | `pattern`, `path`, `file_pattern` | Search in files |
| `glob_files` | `pattern`, `working_dir` | Find files by pattern |
| `read_file` | `path`, `offset`, `limit` | Read file content |
| `understand_image` | `image_source`, `prompt` | Analyze image |
| `webclaw_crawl` | `url`, `prompt` | Crawl webpage |

---

## FORMAT 4: MULTI-TURN CONVERSATION

Used for conversation chains with multiple turns.

```json
{
  "message_type": "multi_turn",
  "content": null,
  "tool_call": null,
  "mcp_call": null,
  "turns": [
    {
      "turn_id": 1,
      "type": "tool_call",
      "tool_call": {
        "id": "call_001",
        "name": "bash_command",
        "arguments": {
          "command": "find src -name '*.java' -type f",
          "working_dir": "/project"
        }
      }
    },
    {
      "turn_id": 2,
      "type": "tool_response",
      "content": "src/main/java/com/example/App.java\nsrc/main/java/com/example/Controller.java\nsrc/main/java/com/example/Service.java"
    },
    {
      "turn_id": 3,
      "type": "assistant",
      "content": "I found 3 Java files in your project:\n\n**Files:**\n1. `App.java` — Main entry point\n2. `Controller.java` — REST endpoints\n3. `Service.java` — Business logic\n\nWhich file would you like me to analyze?",
      "humanize": false
    }
  ],
  "humanize": false,
  "metadata": {
    "language": "java",
    "framework": "spring-boot",
    "difficulty": "easy",
    "turn_count": 3
  }
}
```

---

## FORMAT 5: CODE GENERATION

Used when user asks to write/create code directly.

```json
{
  "message_type": "normal",
  "content": "Here is a React user registration form with Material UI:\n\n```jsx\nimport React, { useState } from 'react';\nimport { TextField, Button, Box, Typography, Alert } from '@mui/material';\n\nexport default function UserRegistration() {\n  const [formData, setFormData] = useState({\n    username: '',\n    email: '',\n    password: '',\n    confirmPassword: ''\n  });\n  const [error, setError] = useState('');\n  const [success, setSuccess] = useState(false);\n\n  const handleSubmit = async (e) => {\n    e.preventDefault();\n    if (formData.password !== formData.confirmPassword) {\n      setError('Passwords do not match');\n      return;\n    }\n    // API call logic...\n  };\n\n  return (\n    <Box component=\"form\" onSubmit={handleSubmit} sx={{ maxWidth: 400, mx: 'auto' }}>\n      <Typography variant=\"h5\" gutterBottom>User Registration</Typography>\n      {error && <Alert severity=\"error\">{error}</Alert>}\n      {success && <Alert severity=\"success\">Registration successful!</Alert>}\n      <TextField fullWidth label=\"Username\" name=\"username\" margin=\"normal\" />\n      <TextField fullWidth label=\"Email\" name=\"email\" type=\"email\" margin=\"normal\" />\n      <TextField fullWidth label=\"Password\" name=\"password\" type=\"password\" margin=\"normal\" />\n      <TextField fullWidth label=\"Confirm Password\" name=\"confirmPassword\" type=\"password\" margin=\"normal\" />\n      <Button type=\"submit\" variant=\"contained\" fullWidth sx={{ mt: 2 }}>Register</Button>\n    </Box>\n  );\n}\n```\n\n**Key Features:**\n- Material UI form with validation\n- Password confirmation check\n- Error/success state handling",
  "tool_call": null,
  "mcp_call": null,
  "turns": null,
  "humanize": false,
  "metadata": {
    "language": "javascript",
    "framework": "react",
    "difficulty": "medium",
    "category": "frontend",
    "task_type": "code_generation"
  }
}
```

---

# PART 3 — PARSING EXAMPLES

## Python Parser

```python
import json

def parse_response(response_text: str) -> dict:
    """Parse model response into structured format."""
    data = json.loads(response_text)

    message_type = data.get("message_type")

    if message_type == "normal":
        return {
            "type": "normal",
            "content": data["content"],
        }

    elif message_type == "tool_call":
        return {
            "type": "tool_call",
            "tool": data["tool_call"]["name"],
            "args": data["tool_call"]["arguments"],
            "provider": "standard",
        }

    elif message_type == "mcp_call":
        return {
            "type": "mcp_call",
            "server": data["mcp_call"]["server"],
            "tool": data["mcp_call"]["name"],
            "args": data["mcp_call"]["arguments"],
        }

    elif message_type == "multi_turn":
        return {
            "type": "multi_turn",
            "turns": data["turns"],
        }

    return {"type": "unknown"}


# Usage
response = '{"message_type":"tool_call","tool_call":{"name":"read_file","arguments":{"path":"test.java"}}}'
parsed = parse_response(response)
print(parsed)
# {'type': 'tool_call', 'tool': 'read_file', 'args': {'path': 'test.java'}, 'provider': 'standard'}
```

## TypeScript Parser

```typescript
interface StandardToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

interface MCPToolCall {
  id: string;
  server: string;
  name: string;
  arguments: Record<string, any>;
}

interface ParsedResponse {
  type: 'normal' | 'tool_call' | 'mcp_call' | 'multi_turn';
  content?: string;
  tool?: string;
  args?: Record<string, any>;
  provider?: string;
  server?: string;
  turns?: any[];
}

function parseResponse(text: string): ParsedResponse {
  const data = JSON.parse(text);

  switch (data.message_type) {
    case 'normal':
      return { type: 'normal', content: data.content };

    case 'tool_call':
      return {
        type: 'tool_call',
        tool: data.tool_call.name,
        args: data.tool_call.arguments,
        provider: 'standard',
      };

    case 'mcp_call':
      return {
        type: 'mcp_call',
        server: data.mcp_call.server,
        tool: data.mcp_call.name,
        args: data.mcp_call.arguments,
      };

    case 'multi_turn':
      return { type: 'multi_turn', turns: data.turns };

    default:
      throw new Error(`Unknown message type: ${data.message_type}`);
  }
}
```

## Java Parser

```java
import com.fasterxml.jackson.databind.ObjectMapper;

public class ResponseParser {
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void parse(String responseJson) throws Exception {
        JsonNode data = mapper.readTree(responseJson);
        String messageType = data.get("message_type").asText();

        switch (messageType) {
            case "normal":
                String content = data.get("content").asText();
                System.out.println("Normal response: " + content);
                break;

            case "tool_call":
                String toolName = data.get("tool_call").get("name").asText();
                JsonNode args = data.get("tool_call").get("arguments");
                System.out.println("Tool call: " + toolName + " with args: " + args);
                break;

            case "mcp_call":
                String mcpServer = data.get("mcp_call").get("server").asText();
                String mcpTool = data.get("mcp_call").get("name").asText();
                System.out.println("MCP call: " + mcpServer + "." + mcpTool);
                break;

            case "multi_turn":
                // Process turns array
                break;
        }
    }
}
```

---

# PART 4 — TRAINING CSV SCHEMA

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | ✅ | UUID v4 |
| `prompt` | string | ✅ | Full structured prompt (init + instruction + tools + user message) |
| `language` | string | ✅ | Primary language: `java`, `python`, `javascript`, `typescript`, `kotlin`, `go`, `rust` |
| `framework` | string | ✅ | Framework: `spring-boot`, `react`, `nextjs`, `django`, `fastapi`, `kotlin-android`, `express`, `none` |
| `task_type` | string | ✅ | `tool_call`, `code_generation`, `code_explanation`, `code_debug`, `code_refactor`, `code_review`, `architecture`, `devops` |
| `message_type` | string | ✅ | `normal`, `tool_call`, `mcp_call`, `multi_turn` |
| `tools_available` | string | ✅ | JSON array of available tool names |
| `output_json` | string | ✅ | The complete structured response as JSON string |
| `content` | string | ⚠️ | Text content (if message_type=normal) |
| `tool_name` | string | ⚠️ | Tool name (if message_type=tool_call or mcp_call) |
| `tool_provider` | string | ⚠️ | `standard`, `mcp` |
| `tool_args` | string | ⚠️ | JSON object of tool arguments |
| `mcp_server` | string | ⚠️ | MCP server name (if tool_provider=mcp) |
| `turns` | string | ⚠️ | JSON array of turns (if message_type=multi_turn) |
| `humanize` | boolean | ✅ | `true` or `false` |
| `category` | string | ✅ | `backend`, `frontend`, `fullstack`, `devops`, `database`, `mobile`, `general` |
| `difficulty` | string | ✅ | `easy`, `medium`, `hard` |
| `agent_role` | string | ✅ | The agent role used in this example |
| `guardrails` | string | ✅ | JSON array of guardrail rules |
| `metadata` | string | ✅ | JSON object with source, difficulty_score, etc. |

### Message Type Distribution (5M)

| Message Type | Count | Percentage |
|--------------|-------|------------|
| `normal` | 2,000,000 | 40% |
| `tool_call` | 1,500,000 | 30% |
| `mcp_call` | 1,000,000 | 20% |
| `multi_turn` | 500,000 | 10% |

---

# PART 5 — Gemma 4 E4B Training Config (RunPod)

```python
from unsloth import FastLanguageModel
import torch

# Model
model_name = "unsloth/gemma-4-E4B-better-fitting-v2"

# Quantization & LoRA
max_seq_length = 2048
dtype = None
load_in_4bit = True  # QLoRA 4-bit

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# LoRA Config
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# Training
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        save_steps=1000,
        output_dir="gemma4-e4b-coding-agent",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    ),
)

trainer.train()
```

### RunPod Recommended Settings

| Setting | Value | Reason |
|---------|-------|--------|
| GPU | RTX 4090 24GB or A100 40GB | Gemma 4 E4B needs ~17GB VRAM |
| Instance Type | `u-8-80-s-uncached` or `a100-80` | Sufficient VRAM + CPU |
| Runtime | Docker with CUDA 12.1+ | For Unsloth |
| Storage | 100GB+ | Model + dataset + checkpoints |
| Sequence Length | 2048 | Balance quality vs VRAM |
| Batch Size | 4 | With gradient accumulation 4 |
| Epochs | 3 | For coding-focused fine-tune |
| Learning Rate | 2e-4 | Standard for LoRA |

---

*Last Updated: 2026-04-15*
