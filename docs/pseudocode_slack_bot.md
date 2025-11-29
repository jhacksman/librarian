# Librarian Slack Bot - Pseudocode

## Overview
Slack bot (@librarian) in #books channel for querying, recommending, and checking out digital books. Uses Bolt Python framework with threaded conversations for focused book discussions.

## Tech Stack
- Framework: Slack Bolt for Python
- LLM: Qwen3-72B via vLLM (shared with website backend)
- Vector DB: Qdrant (shared with ingest module)
- Embeddings: BGE-base-en-v1.5

---

## Application Structure

```python
# app.py - Main Slack Bot Application

import os
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from qdrant_client import QdrantClient
from openai import AsyncOpenAI

# ============================================
# INITIALIZATION
# ============================================

app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Shared services (same as website backend)
qdrant = QdrantClient(host="localhost", port=6333)
llm = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
embeddings = load_bge_model("BAAI/bge-base-en-v1.5")

# Conversation state tracking
# Key: thread_ts, Value: {mode: "global"|"book", book_id: str|None, history: []}
conversation_state = {}

# ============================================
# MESSAGE HANDLERS
# ============================================

@app.event("app_mention")
def handle_mention(event, say, client):
    """
    Handle @librarian mentions - entry point for all interactions.
    
    PSEUDOCODE:
    1. Check if this is in a thread or new message
    2. If new message: start new conversation thread
    3. If in thread: continue existing conversation
    4. Parse user intent and route to appropriate handler
    """
    
    user_id = event["user"]
    channel_id = event["channel"]
    text = event["text"].replace("<@BOT_ID>", "").strip()
    thread_ts = event.get("thread_ts") or event["ts"]
    
    # Get or create conversation state
    if thread_ts not in conversation_state:
        conversation_state[thread_ts] = {
            "mode": "global",  # Start in global/discovery mode
            "book_id": None,
            "history": [],
            "user_id": user_id
        }
    
    state = conversation_state[thread_ts]
    
    # Add user message to history
    state["history"].append({"role": "user", "content": text})
    
    # Route based on current mode and intent
    intent = classify_intent(text, state)
    
    if intent["type"] == "search":
        response = handle_search(text, state)
    elif intent["type"] == "recommend":
        response = handle_recommendation(text, state)
    elif intent["type"] == "focus_book":
        response = handle_focus_book(intent["book_id"], state, say, thread_ts)
        return  # focus_book handles its own response
    elif intent["type"] == "book_question":
        response = handle_book_question(text, state)
    elif intent["type"] == "read_chapter":
        response = handle_read_chapter(intent["chapter"], state)
    elif intent["type"] == "checkout":
        response = handle_checkout(state, user_id)
    elif intent["type"] == "list_checkouts":
        response = handle_list_checkouts(user_id)
    elif intent["type"] == "return_book":
        response = handle_return_book(intent["book_id"], user_id)
    elif intent["type"] == "help":
        response = get_help_message()
    else:
        response = handle_general_query(text, state)
    
    # Add assistant response to history
    state["history"].append({"role": "assistant", "content": response})
    
    # Reply in thread
    say(text=response, thread_ts=thread_ts)


def classify_intent(text: str, state: dict) -> dict:
    """
    Classify user intent from message text.
    
    PSEUDOCODE:
    1. Check for explicit commands (checkout, return, read chapter X)
    2. Check for recommendation keywords
    3. Check for book focus requests
    4. If in book mode, default to book question
    5. Otherwise, classify as search or general
    """
    
    text_lower = text.lower()
    
    # Explicit commands
    if "checkout" in text_lower or "download" in text_lower:
        return {"type": "checkout"}
    
    if "return" in text_lower and "book" in text_lower:
        book_match = re.search(r"return\s+(.+)", text_lower)
        return {"type": "return_book", "book_id": extract_book_id(book_match)}
    
    if re.match(r"read\s+chapter\s+(\d+)", text_lower):
        chapter = int(re.search(r"chapter\s+(\d+)", text_lower).group(1))
        return {"type": "read_chapter", "chapter": chapter}
    
    if "my checkouts" in text_lower or "my books" in text_lower:
        return {"type": "list_checkouts"}
    
    if "help" in text_lower:
        return {"type": "help"}
    
    # Recommendation keywords
    recommend_keywords = ["recommend", "suggest", "gimmee", "gimme", "give me", "find me", "looking for"]
    if any(kw in text_lower for kw in recommend_keywords):
        return {"type": "recommend"}
    
    # Book focus request (e.g., "let's talk about Black Hat Python")
    focus_patterns = [
        r"(?:let's|lets|want to)\s+(?:talk|chat|discuss)\s+(?:about\s+)?(.+)",
        r"(?:focus|switch)\s+(?:on|to)\s+(.+)",
        r"tell me (?:more )?about\s+(.+)"
    ]
    for pattern in focus_patterns:
        match = re.search(pattern, text_lower)
        if match:
            book_title = match.group(1).strip()
            book_id = find_book_by_title(book_title)
            if book_id:
                return {"type": "focus_book", "book_id": book_id}
    
    # If already in book mode, treat as book question
    if state["mode"] == "book" and state["book_id"]:
        return {"type": "book_question"}
    
    # Default to search
    return {"type": "search"}

# ============================================
# SEARCH & DISCOVERY (Global Mode)
# ============================================

def handle_search(query: str, state: dict) -> str:
    """
    Search the library for relevant books.
    
    PSEUDOCODE:
    1. Generate embedding for query
    2. Search Qdrant for matching books
    3. Format results with titles, authors, brief descriptions
    4. Suggest user can focus on a specific book for more details
    """
    
    query_embedding = embeddings.encode(query)
    
    # Search book metadata
    results = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter={"must": [{"is_metadata": True}]},
        limit=5,
        with_payload=True
    )
    
    if not results:
        return "I couldn't find any books matching that query. Try different keywords or ask me for recommendations!"
    
    # Format response
    response_parts = ["Here's what I found:\n"]
    
    for i, result in enumerate(results, 1):
        book = result.payload
        response_parts.append(
            f"*{i}. {book['title']}* by {', '.join(book['authors'])}\n"
            f"   Tags: {', '.join(book['tags'][:5])}\n"
            f"   _{book['summary'][:150]}..._\n"
        )
    
    response_parts.append(
        "\nWant to dive deeper? Say something like "
        "\"let's talk about [book title]\" to focus on a specific book."
    )
    
    return "\n".join(response_parts)


def handle_recommendation(query: str, state: dict) -> str:
    """
    Get personalized book recommendations.
    
    PSEUDOCODE:
    1. Use LLM to understand what user is looking for
    2. Search library with extracted criteria
    3. Use LLM to rank and explain recommendations
    4. Return top recommendations with reasons
    """
    
    # Step 1: Parse user intent
    intent_prompt = f"""
    A hackerspace member is looking for book recommendations.
    Their request: "{query}"
    
    Extract:
    - Topics they're interested in (e.g., Python, security, AI)
    - Skill level if mentioned (beginner/intermediate/advanced)
    - Any specific requirements or constraints
    
    Return JSON: {{"topics": [], "level": "", "requirements": []}}
    """
    
    intent_response = llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": intent_prompt}],
        response_format={"type": "json_object"}
    )
    
    parsed_intent = json.loads(intent_response.choices[0].message.content)
    
    # Step 2: Search library
    search_query = " ".join(parsed_intent["topics"])
    query_embedding = embeddings.encode(search_query)
    
    candidates = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter={"must": [{"is_metadata": True}]},
        limit=10,
        with_payload=True
    )
    
    if not candidates:
        return "Hmm, I couldn't find books matching that criteria. Could you tell me more about what you're looking for?"
    
    # Step 3: LLM ranks and explains
    books_context = "\n".join([
        f"- {c.payload['title']} by {', '.join(c.payload['authors'])}: {c.payload['summary'][:200]}"
        for c in candidates
    ])
    
    ranking_prompt = f"""
    User request: "{query}"
    
    Available books:
    {books_context}
    
    Select the top 3 most relevant books and explain why each would be good for this user.
    Be specific about what they'll learn and why it matches their needs.
    
    Format your response as a friendly recommendation, not JSON.
    """
    
    recommendation = llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": ranking_prompt}]
    )
    
    response = recommendation.choices[0].message.content
    response += "\n\nWant to learn more about any of these? Just say \"let's talk about [book title]\"!"
    
    return response

# ============================================
# BOOK FOCUS MODE (Per-Book Chat)
# ============================================

def handle_focus_book(book_id: str, state: dict, say, thread_ts: str) -> None:
    """
    Switch conversation to focus on a specific book.
    
    PSEUDOCODE:
    1. Fetch book details
    2. Update conversation state to book mode
    3. Send book overview with available actions
    4. Clear conversation history for fresh context
    """
    
    # Fetch book details
    book = get_book_details(book_id)
    
    if not book:
        say(text="I couldn't find that book. Could you check the title?", thread_ts=thread_ts)
        return
    
    # Update state
    state["mode"] = "book"
    state["book_id"] = book_id
    state["history"] = []  # Fresh start for book-focused chat
    
    # Build overview message
    overview = f"""
*Now focusing on: {book['title']}*
by {', '.join(book['authors'])}

*Summary:*
{book['summary']}

*Topics:* {', '.join(book['tags'][:8])}
*Difficulty:* {book['difficulty_level']}

*What you can do:*
- Ask me anything about this book
- Say "read chapter X" to get the full chapter text
- Say "key quotes" to see notable quotes
- Say "checkout" to download the book
- Say "back to search" to return to browsing

What would you like to know?
"""
    
    say(text=overview, thread_ts=thread_ts)


def handle_book_question(question: str, state: dict) -> str:
    """
    Answer a question about the focused book using RAG.
    
    PSEUDOCODE:
    1. Search book's chunks for relevant context
    2. Build prompt with context and conversation history
    3. Generate response with citations
    4. Format citations as references to chapters/pages
    """
    
    book_id = state["book_id"]
    
    # Search for relevant chunks
    query_embedding = embeddings.encode(question)
    
    relevant_chunks = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter={
            "must": [
                {"book_id": book_id},
                {"is_metadata": False}
            ]
        },
        limit=5,
        with_payload=True
    )
    
    # Build context with citations
    context_parts = []
    citations = []
    for i, chunk in enumerate(relevant_chunks):
        citation_id = f"[{i+1}]"
        context_parts.append(f"{citation_id} {chunk.payload['content']}")
        citations.append({
            "id": citation_id,
            "chapter": chunk.payload.get("chapter_title", "Unknown"),
            "page": chunk.payload.get("start_page")
        })
    
    context = "\n\n".join(context_parts)
    
    # Build conversation history for context
    history_context = ""
    if state["history"]:
        recent_history = state["history"][-6:]  # Last 3 exchanges
        history_context = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent_history
        ])
    
    # Generate response
    book_title = get_book_title(book_id)
    
    system_prompt = f"""
    You are a helpful librarian assistant discussing the book "{book_title}".
    
    Use the following context from the book to answer questions.
    Always cite your sources using markers like [1], [2], etc.
    If the context doesn't contain the answer, say so honestly.
    Keep responses concise but informative - this is Slack, not an essay.
    
    Book Context:
    {context}
    
    Recent Conversation:
    {history_context}
    """
    
    response = llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    
    # Append citation references
    if citations:
        citation_refs = "\n\n_References:_\n" + "\n".join([
            f"_{c['id']} Chapter: {c['chapter']}" + (f", Page {c['page']}" if c['page'] else "") + "_"
            for c in citations
        ])
        answer += citation_refs
    
    return answer


def handle_read_chapter(chapter_number: int, state: dict) -> str:
    """
    Return full text of a specific chapter.
    
    PSEUDOCODE:
    1. Verify we're in book mode
    2. Fetch all chunks for this chapter
    3. Reconstruct and return chapter text
    4. If chapter is long, offer to split into parts
    """
    
    if state["mode"] != "book" or not state["book_id"]:
        return "Please focus on a book first by saying \"let's talk about [book title]\""
    
    book_id = state["book_id"]
    
    # Fetch chapter chunks
    chunks = qdrant.scroll(
        collection_name="librarian_books",
        filter={
            "must": [
                {"book_id": book_id},
                {"chapter_number": chapter_number},
                {"is_metadata": False}
            ]
        },
        order_by="chunk_index"
    )
    
    if not chunks:
        return f"I couldn't find chapter {chapter_number}. This book might have fewer chapters."
    
    chapter_title = chunks[0].payload.get("chapter_title", f"Chapter {chapter_number}")
    chapter_text = "\n\n".join([c.payload["content"] for c in chunks])
    
    # Slack has message length limits (~4000 chars for best display)
    if len(chapter_text) > 3500:
        # Split into parts
        parts = split_text_into_parts(chapter_text, max_length=3000)
        response = f"*{chapter_title}* (Part 1 of {len(parts)})\n\n{parts[0]}"
        response += f"\n\n_Say \"continue\" for the next part._"
        
        # Store remaining parts in state for continuation
        state["pending_parts"] = parts[1:]
    else:
        response = f"*{chapter_title}*\n\n{chapter_text}"
    
    return response

# ============================================
# CHECKOUT SYSTEM
# ============================================

def handle_checkout(state: dict, user_id: str) -> str:
    """
    Check out the currently focused book.
    
    PSEUDOCODE:
    1. Verify we're in book mode
    2. Record checkout with 2-week expiry
    3. Generate and return download link
    4. Notify user of expiry date
    """
    
    if state["mode"] != "book" or not state["book_id"]:
        return "Please focus on a book first by saying \"let's talk about [book title]\", then say \"checkout\"."
    
    book_id = state["book_id"]
    book = get_book_details(book_id)
    
    # Record checkout
    checkout_record = {
        "book_id": book_id,
        "user_id": user_id,
        "checkout_time": datetime.now(),
        "expiry_time": datetime.now() + timedelta(weeks=2),
        "status": "active",
        "source": "slack"
    }
    
    save_checkout_record(checkout_record)
    
    # Generate download link
    download_url = generate_secure_download_link(book_id)
    expiry_date = checkout_record["expiry_time"].strftime("%B %d, %Y")
    
    return f"""
*Checked out: {book['title']}*

Here's your download link (expires in 24 hours):
{download_url}

The book is checked out to you until *{expiry_date}*.
It will auto-return after that, but you can say "return {book['title']}" anytime to return it early.

Enjoy the read!
"""


def handle_list_checkouts(user_id: str) -> str:
    """
    List user's current checkouts.
    
    PSEUDOCODE:
    1. Query checkout records for this user
    2. Filter to active checkouts
    3. Format with titles and expiry dates
    """
    
    checkouts = get_user_checkouts(user_id)
    active = [c for c in checkouts if c["status"] == "active"]
    
    if not active:
        return "You don't have any books checked out right now. Browse the library with \"@librarian recommend [topic]\"!"
    
    response_parts = ["*Your checked out books:*\n"]
    
    for checkout in active:
        book = get_book_details(checkout["book_id"])
        days_left = (checkout["expiry_time"] - datetime.now()).days
        
        response_parts.append(
            f"- *{book['title']}*\n"
            f"  Due: {checkout['expiry_time'].strftime('%B %d')} ({days_left} days left)\n"
        )
    
    response_parts.append("\nSay \"return [book title]\" to return a book early.")
    
    return "\n".join(response_parts)


def handle_return_book(book_identifier: str, user_id: str) -> str:
    """
    Return a checked-out book.
    
    PSEUDOCODE:
    1. Find the book by title or ID
    2. Verify user has it checked out
    3. Mark as returned
    4. Confirm return
    """
    
    # Find book
    book_id = find_book_by_title(book_identifier) or book_identifier
    
    # Find active checkout
    checkout = get_active_checkout(book_id, user_id)
    
    if not checkout:
        return f"I don't see that book checked out to you. Say \"my checkouts\" to see your current books."
    
    # Mark as returned
    checkout["status"] = "returned"
    checkout["return_time"] = datetime.now()
    update_checkout_record(checkout)
    
    book = get_book_details(book_id)
    
    return f"*{book['title']}* has been returned. Thanks for reading!"

# ============================================
# WEB SEARCH INTEGRATION
# ============================================

def handle_general_query(query: str, state: dict) -> str:
    """
    Handle queries that might need web search.
    
    PSEUDOCODE:
    1. First check if query relates to library content
    2. If not, perform web search
    3. Combine library and web results
    4. Generate response with citations to both sources
    """
    
    # Check if this is a library-related query
    library_results = search_library(query, limit=3)
    
    # Determine if we need web search
    needs_web = should_search_web(query, library_results)
    
    web_results = []
    if needs_web:
        web_results = perform_web_search(query)
    
    # Build context
    context_parts = []
    
    if library_results:
        context_parts.append("From the library:")
        for r in library_results:
            context_parts.append(f"- {r['title']}: {r['content'][:200]}")
    
    if web_results:
        context_parts.append("\nFrom the web:")
        for r in web_results:
            context_parts.append(f"- [{r['title']}]({r['url']}): {r['snippet']}")
    
    # Generate response
    prompt = f"""
    User question: "{query}"
    
    Available information:
    {chr(10).join(context_parts)}
    
    Provide a helpful answer. If using web sources, include the links.
    If the question is about books we have, mention them.
    Keep it concise for Slack.
    """
    
    response = llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def should_search_web(query: str, library_results: list) -> bool:
    """
    Determine if web search is needed.
    
    PSEUDOCODE:
    1. If query asks about current events, news -> yes
    2. If query is about something not in library -> yes
    3. If library results are highly relevant -> no
    4. Default to no for book-focused queries
    """
    
    web_indicators = [
        "latest", "recent", "news", "current", "2024", "2025",
        "what is", "how to", "tutorial", "guide"
    ]
    
    query_lower = query.lower()
    
    # Check for web indicators
    if any(indicator in query_lower for indicator in web_indicators):
        return True
    
    # If no library results, try web
    if not library_results:
        return True
    
    # If library results have low relevance scores
    if library_results and library_results[0].get("score", 0) < 0.5:
        return True
    
    return False

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_help_message() -> str:
    """Return help message with available commands."""
    
    return """
*Librarian Bot Help*

*Discovery:*
- `@librarian [search query]` - Search the library
- `@librarian recommend [topic]` - Get personalized recommendations
- `@librarian gimmee infosec books on python` - Natural language requests

*Book Focus:*
- `let's talk about [book title]` - Focus on a specific book
- `read chapter X` - Get full chapter text
- `key quotes` - See notable quotes from the book
- `back to search` - Return to browsing mode

*Checkout:*
- `checkout` - Download the focused book (2-week checkout)
- `my checkouts` - See your checked out books
- `return [book title]` - Return a book early

*Other:*
- Ask any question - I'll search the library and web
- All conversations happen in threads for easy tracking

Happy reading!
"""


def get_book_details(book_id: str) -> dict | None:
    """Fetch full book details from Qdrant."""
    
    result = qdrant.retrieve(
        collection_name="librarian_books",
        ids=[f"{book_id}_metadata"]
    )
    
    return result[0].payload if result else None


def get_book_title(book_id: str) -> str:
    """Get just the title for a book ID."""
    
    book = get_book_details(book_id)
    return book["title"] if book else "Unknown Book"


def find_book_by_title(title: str) -> str | None:
    """Find a book ID by fuzzy title match."""
    
    # Search for books with similar titles
    query_embedding = embeddings.encode(title)
    
    results = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter={"must": [{"is_metadata": True}]},
        limit=1,
        with_payload=True
    )
    
    if results and results[0].score > 0.7:
        return results[0].payload["book_id"]
    
    return None


# ============================================
# BACKGROUND TASKS
# ============================================

def auto_return_expired_checkouts():
    """
    Scheduled task to auto-return books after 2-week expiry.
    Run daily via cron or scheduler.
    
    PSEUDOCODE:
    1. Query all active checkouts past expiry
    2. Mark each as auto-returned
    3. Optionally notify users via DM
    """
    
    expired = get_expired_checkouts()
    
    for checkout in expired:
        checkout["status"] = "auto-returned"
        checkout["return_time"] = datetime.now()
        update_checkout_record(checkout)
        
        # Optional: DM user about auto-return
        # client.chat_postMessage(
        #     channel=checkout["user_id"],
        #     text=f"Your checkout of '{get_book_title(checkout['book_id'])}' has been auto-returned."
        # )
        
        log.info(f"Auto-returned book {checkout['book_id']} for user {checkout['user_id']}")


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Socket Mode for local development (no public URL needed)
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    
    print("Librarian bot is running!")
    handler.start()
```

---

## Slack App Configuration

```yaml
# manifest.yaml - Slack App Manifest

display_information:
  name: Librarian
  description: Hackerspace book library assistant
  background_color: "#2c3e50"

features:
  bot_user:
    display_name: librarian
    always_online: true
  
  app_home:
    home_tab_enabled: true
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false

oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - channels:history
      - channels:read
      - chat:write
      - files:write
      - im:history
      - im:read
      - im:write
      - users:read

settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.im
  
  interactivity:
    is_enabled: true
  
  socket_mode_enabled: true
```

---

## Conversation Flow Examples

### Example 1: Discovery to Checkout

```
User: @librarian gimmee infosec books on python

Librarian: Based on your interest in Python for security, here are my top recommendations:

1. *Black Hat Python* by Justin Seitz
   Perfect for learning offensive security techniques with Python. Covers network 
   sniffing, web hacking, and creating trojans. Intermediate level.

2. *Gray Hat Python* by Justin Seitz  
   Focuses on debugging, reverse engineering, and fuzzing. Great companion to 
   Black Hat Python.

3. *Violent Python* by TJ O'Connor
   Cookbook-style with practical recipes for penetration testing.

Want to learn more about any of these? Just say "let's talk about [book title]"!

---

User: let's talk about Black Hat Python

Librarian: *Now focusing on: Black Hat Python*
by Justin Seitz

*Summary:*
A guide to using Python for offensive security tasks including network analysis,
web application attacks, and creating custom security tools...

*Topics:* Python, Security, Networking, Penetration Testing, Malware
*Difficulty:* Intermediate

*What you can do:*
- Ask me anything about this book
- Say "read chapter X" to get the full chapter text
- Say "key quotes" to see notable quotes
- Say "checkout" to download the book
- Say "back to search" to return to browsing

What would you like to know?

---

User: what does it cover about network sniffing?

Librarian: Black Hat Python covers network sniffing extensively in Chapters 3-4 [1][2].

The book teaches you to build a raw socket sniffer from scratch using Python's 
`socket` module, then progresses to using Scapy for more sophisticated packet 
manipulation. Key topics include:

- Capturing packets on Windows and Linux
- Decoding IP and ICMP headers
- Building an ARP cache poisoner
- PCAP file processing

The author emphasizes understanding the underlying protocols rather than just 
using tools, which makes this knowledge transferable. [1]

_References:_
_[1] Chapter 3: The Network - Raw Sockets and Sniffing_
_[2] Chapter 4: Owning the Network with Scapy_

---

User: checkout

Librarian: *Checked out: Black Hat Python*

Here's your download link (expires in 24 hours):
https://librarian.hackerspace.local/download/abc123xyz

The book is checked out to you until *December 13, 2025*.
It will auto-return after that, but you can say "return Black Hat Python" anytime.

Enjoy the read!
```

### Example 2: Direct Question with Web Search

```
User: @librarian what's the latest version of Python and should I use it?

Librarian: The latest stable Python version is *3.12.1* (as of late 2024).

From our library, several books are written for Python 3.8-3.10, which remain 
fully compatible. *Black Hat Python* and *Automate the Boring Stuff* both work 
great with 3.12.

Key 3.12 features relevant to our collection:
- Improved error messages (great for learning)
- Faster startup time
- Better typing support

I'd recommend Python 3.11 or 3.12 for new projects. The security books in our 
library will work fine with either.

Sources: [Python.org Release Notes](https://python.org/downloads/)
```

---

## Key Features Summary

1. **Threaded Conversations**: All interactions happen in threads for organized discussions
2. **Global vs Book Mode**: Start broad, narrow to specific book for focused Q&A
3. **Natural Language**: "gimmee infosec books" works as well as formal queries
4. **RAG-Based Answers**: Questions about books cite specific chapters/pages
5. **Chapter Reading**: Full chapter text on demand
6. **High-Trust Checkout**: Simple download with 2-week auto-return
7. **Web Search Fallback**: Non-library queries get web results with citations
8. **Conversation Memory**: Context maintained within threads

## Environment Variables

```bash
SLACK_BOT_TOKEN=xoxb-...      # Bot User OAuth Token
SLACK_APP_TOKEN=xapp-...      # App-Level Token (for Socket Mode)
QDRANT_HOST=localhost
QDRANT_PORT=6333
VLLM_BASE_URL=http://localhost:8000/v1
```
