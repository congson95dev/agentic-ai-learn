
import warnings
import logging
import time
import os
import litellm
from litellm import Router, completion, completion_cost
from litellm.caching import Cache
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

# Enable in-memory caching (you can also use Redis in production)
litellm.cache = Cache(type="local")

model_list = [
    {
        "model_name": "code-pool",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "code-openai-gpt-4o"},
    },
    {
        "model_name": "code-pool",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "code-openai-gpt-4o-mini"},
    },
    {
        "model_name": "code-pool",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "code-groq-gpt-oss-20b"},
    },
    {
        "model_name": "summary-pool",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "summary-openai-gpt-4o-mini"},
    },
    {
        "model_name": "summary-pool",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "summary-groq-gpt-oss-20b"},
    },
    {
        "model_name": "general-pool",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "general-groq-gpt-oss-20b"},
    },
    {
        "model_name": "general-pool",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "general-openai-gpt-4o-mini"},
    },
]

# Apply Load Balancing, type="least-busy".
# There is another type called "simple-shuffle", we can also use that.
router = Router(model_list=model_list, routing_strategy="least-busy")

# Classify the task type of the user query: code, summary, or general.
def classify_task(user_query: str) -> str:
    cls = completion(
        model="groq/openai/gpt-oss-20b",
        messages=[{
            "role": "user",
            "content": (
                "Classify this query as exactly one of: code, summary, general. "
                "Return only the category.\n"
                f"Query: {user_query}"
            )
        }],
        reasoning_effort="low",
        include_reasoning=False,
        max_tokens=100,
    )

    raw = cls.choices[0].message.content or ""
    task = raw.strip().lower()

    return task if task in {"code", "summary", "general"} else "general"

# If a model fails, try the next one in the chain. If all fail, raise the last error.
def call_with_fallbacks(model_chain, messages):
    """Try each least-busy model pool in order; return the first success."""
    last_error = None
    for model in model_chain:
        try:
            return router.completion(model=model, messages=messages)
        except Exception as e:
            print(f"   ⚠️  {model} failed ({type(e).__name__}), trying next...")
            last_error = e
            continue
    raise last_error


def smart_chat(user_query: str):
    """Routes to the right model based on task type, with fallbacks."""
    start = time.time()

    # Step 1: Classify the task
    task = classify_task(user_query)
    print(f"   Detected task: {task!r}")

    # Step 2: Route to the right model(s)
    routing = {
        "code":    ["code-pool"],
        "summary": ["summary-pool"],
        "general": ["general-pool"],
    }
    model_chain = routing.get(task, routing["general"])

    # Step 3: Call the model(s) with fallbacks
    response = call_with_fallbacks(
        model_chain=model_chain,
        messages=[{"role": "user", "content": user_query}]
    )
    latency = time.time() - start

    usage = response.usage
    usage_token = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }

    try:
        cost = completion_cost(completion_response=response)
        cost_str = f"${cost:.6f}"
    except Exception:
        cost_str = "n/a - maybe this is a free model so it don't take cost"

    return {
        "detected_task": task,
        "model_used":    response.model,
        "answer":        response.choices[0].message.content,
        "latency_sec":   round(latency, 2),
        "cost_usd":      cost_str,
        "usage_token":   usage_token
    }


# Try 3 queries
queries = [
    "Write a Python function to compute Fibonacci numbers.",
    "Write a Python function to compute Fibonacci numbers.", # This is a duplicate query to test caching.
    "Summarize the importance of attention mechanism in 2 sentences.",
    "Tell me a fun fact about elephants."
]
for query in queries:
    print("=" * 70)
    print("❓ Q:", query)
    result = smart_chat(query)
    print(f"🏷️  Task:    {result['detected_task']}")
    print(f"🤖 Model:    {result['model_used']}")
    print(f"⏱️  Latency: {result['latency_sec']}s")
    print(f"💰 Cost:    {result['cost_usd']}")
    print(f"💰 Usage Token:    {result['usage_token']}")
    print(f"💬 Answer:  {result['answer'][:200]}...")


# Output:
# ======================================================================
# ❓ Q: Write a Python function to compute Fibonacci numbers.
#    Detected task: 'code'
# 🏷️  Task:    code
# 🤖 Model:    gpt-4o-2024-08-06
# ⏱️  Latency: 5.42s
# 💰 Cost:    $0.003250
# 💰 Usage Token:    {'prompt_tokens': 16, 'completion_tokens': 321, 'total_tokens': 337}
# 💬 Answer:  Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, typically starting with 0 and 1. Below is a simple Python function to compute the nth F...
# ======================================================================
# ❓ Q: Write a Python function to compute Fibonacci numbers.
#    Detected task: 'code'
# 🏷️  Task:    code
# 🤖 Model:    gpt-4o-2024-08-06
# ⏱️  Latency: 0.01s
# 💰 Cost:    $0.003250
# 💰 Usage Token:    {'prompt_tokens': 16, 'completion_tokens': 321, 'total_tokens': 337}
# 💬 Answer:  Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, typically starting with 0 and 1. Below is a simple Python function to compute the nth F...
# ======================================================================
# ❓ Q: Summarize the importance of attention mechanism in 2 sentences.
#    Detected task: 'summary'
# 🏷️  Task:    summary
# 🤖 Model:    gpt-4o-mini-2024-07-18
# ⏱️  Latency: 2.68s
# 💰 Cost:    $0.000042
# 💰 Usage Token:    {'prompt_tokens': 20, 'completion_tokens': 65, 'total_tokens': 85}
# 💬 Answer:  The attention mechanism is crucial in neural networks as it allows models to focus on specific parts ofthe input data, enhancing their ability to capture relevant information while ignoring irrelevan...
# ======================================================================
# ❓ Q: Tell me a fun fact about elephants.
#    Detected task: 'general'
# 🏷️  Task:    general
# 🤖 Model:    openai/gpt-oss-20b
# ⏱️  Latency: 2.65s
# 💰 Cost:    n/a - maybe this is a free model so it don't take cost
# 💰 Usage Token:    {'prompt_tokens': 79, 'completion_tokens': 291, 'total_tokens': 370}
# 💬 Answer:  **Fun Fact:** An elephant’s trunk is not just a nose—it’s a super‑tool made of about 40,000 tiny muscles, no tendons or bones, and it can lift a whole adult human, pick up a single leaf, or even tap t...
