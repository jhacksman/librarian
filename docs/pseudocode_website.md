# Librarian Website - Pseudocode

## Overview
Self-hosted web interface (Open WebUI style) for browsing, querying, recommending, and checking out digital books. Mirrors Slack functionality with per-book chat capabilities.

## Tech Stack
- Backend: FastAPI (Python)
- Frontend: React/Next.js or simple Jinja2 templates
- Auth: Initially open, future SSO to Slack/Authentik
- LLM: Qwen3-72B via vLLM
- Vector DB: Qdrant (shared with ingest module)

---

## Backend API (FastAPI)

```python
# main.py - FastAPI Application

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import AsyncOpenAI  # vLLM OpenAI-compatible client

app = FastAPI(title="Librarian API")

# ============================================
# MODELS
# ============================================

class SearchQuery:
    query: str
    filters: dict | None  # {tags: [], genres: [], difficulty: str}
    limit: int = 10

class ChatMessage:
    role: str  # "user" or "assistant"
    content: str

class BookChatRequest:
    book_id: str
    messages: list[ChatMessage]
    include_web_search: bool = False

class CheckoutRequest:
    book_id: str
    user_id: str  # Slack user ID or web session ID

class BookRecommendationRequest:
    query: str  # e.g., "infosec books on python"
    limit: int = 5

# ============================================
# INITIALIZATION
# ============================================

def initialize():
    # Connect to Qdrant (same instance as ingest module)
    qdrant = QdrantClient(host="localhost", port=6333)
    
    # Connect to vLLM server
    llm = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    
    # Load embedding model for queries
    embeddings = load_bge_model("BAAI/bge-base-en-v1.5")
    
    return qdrant, llm, embeddings

# ============================================
# SEARCH & DISCOVERY
# ============================================

@app.get("/api/books")
async def list_books(
    page: int = 1,
    per_page: int = 20,
    sort_by: str = "title",  # title, author, date_added
    filters: dict = None
):
    """
    List all books with pagination and filtering.
    
    PSEUDOCODE:
    1. Query Qdrant for all book metadata points (is_metadata=True)
    2. Apply filters if provided (tags, genres, difficulty)
    3. Sort results by specified field
    4. Paginate and return
    """
    
    # Build filter conditions
    filter_conditions = [{"is_metadata": True}]
    if filters:
        if filters.get("tags"):
            filter_conditions.append({"tags": {"$in": filters["tags"]}})
        if filters.get("genres"):
            filter_conditions.append({"genres": {"$in": filters["genres"]}})
        if filters.get("difficulty"):
            filter_conditions.append({"difficulty_level": filters["difficulty"]})
    
    # Query Qdrant
    results = qdrant.scroll(
        collection_name="librarian_books",
        filter={"must": filter_conditions},
        limit=per_page,
        offset=(page - 1) * per_page
    )
    
    # Format response
    books = []
    for result in results:
        books.append({
            "id": result.payload["book_id"],
            "title": result.payload["title"],
            "authors": result.payload["authors"],
            "summary": result.payload["summary"][:200] + "...",
            "tags": result.payload["tags"],
            "genres": result.payload["genres"],
            "cover_image": result.payload.get("cover_image"),
            "checkout_status": get_checkout_status(result.payload["book_id"])
        })
    
    return {"books": books, "total": total_count, "page": page}


@app.post("/api/search")
async def search_books(query: SearchQuery):
    """
    Semantic search across book library.
    
    PSEUDOCODE:
    1. Generate embedding for search query
    2. Search Qdrant with hybrid search (vector + keyword)
    3. Return ranked results with relevance scores
    """
    
    # Generate query embedding
    query_embedding = embeddings.encode(query.query)
    
    # Build filters
    filters = None
    if query.filters:
        filters = build_qdrant_filter(query.filters)
    
    # Hybrid search: combine vector similarity with keyword matching
    results = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter=filters,
        limit=query.limit,
        with_payload=True
    )
    
    # Format and return results
    return {
        "results": [
            {
                "book_id": r.payload["book_id"],
                "title": r.payload["title"],
                "authors": r.payload["authors"],
                "relevance_score": r.score,
                "matched_content": r.payload.get("content", "")[:300],
                "chapter": r.payload.get("chapter_title")
            }
            for r in results
        ]
    }


@app.post("/api/recommend")
async def get_recommendations(request: BookRecommendationRequest):
    """
    Get book recommendations based on natural language query.
    
    PSEUDOCODE:
    1. Use LLM to understand user intent and extract search criteria
    2. Search library for matching books
    3. Use LLM to rank and explain recommendations
    4. Return recommendations with explanations
    """
    
    # Step 1: Parse user intent with LLM
    intent_prompt = f"""
    User is looking for book recommendations: "{request.query}"
    
    Extract:
    - Topics/subjects they're interested in
    - Skill level (beginner/intermediate/advanced)
    - Any specific requirements
    
    Return as JSON: {{"topics": [], "level": "", "requirements": []}}
    """
    
    intent = await llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": intent_prompt}],
        response_format={"type": "json_object"}
    )
    
    parsed_intent = json.loads(intent.choices[0].message.content)
    
    # Step 2: Search library
    search_results = await search_books(SearchQuery(
        query=request.query,
        filters={"tags": parsed_intent["topics"]},
        limit=request.limit * 2  # Get extra for LLM to filter
    ))
    
    # Step 3: LLM ranks and explains recommendations
    ranking_prompt = f"""
    User query: "{request.query}"
    
    Available books:
    {format_books_for_prompt(search_results)}
    
    Select the top {request.limit} most relevant books and explain why each is recommended.
    Consider: relevance to query, difficulty level, practical value.
    
    Return as JSON array: [{{"book_id": "", "recommendation_reason": ""}}]
    """
    
    rankings = await llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": ranking_prompt}],
        response_format={"type": "json_object"}
    )
    
    return {"recommendations": json.loads(rankings.choices[0].message.content)}

# ============================================
# BOOK DETAILS & CHAT
# ============================================

@app.get("/api/books/{book_id}")
async def get_book_details(book_id: str):
    """
    Get full details for a specific book.
    
    PSEUDOCODE:
    1. Fetch book metadata from Qdrant
    2. Include: full summary, chapter list, tags, themes, key quotes
    3. Include checkout status and availability
    """
    
    # Query book metadata
    result = qdrant.retrieve(
        collection_name="librarian_books",
        ids=[f"{book_id}_metadata"]
    )
    
    if not result:
        raise HTTPException(404, "Book not found")
    
    book = result[0].payload
    
    return {
        "id": book_id,
        "title": book["title"],
        "authors": book["authors"],
        "publisher": book.get("publisher"),
        "publication_year": book.get("publication_year"),
        "isbn": book.get("isbn"),
        "summary": book["summary"],
        "main_topics": book["main_topics"],
        "key_takeaways": book["key_takeaways"],
        "tags": book["tags"],
        "genres": book["genres"],
        "difficulty_level": book["difficulty_level"],
        "prerequisites": book.get("prerequisites", []),
        "primary_themes": book["primary_themes"],
        "chapter_summaries": get_chapter_summaries(book_id),
        "key_quotes": get_key_quotes(book_id),
        "checkout_status": get_checkout_status(book_id),
        "total_chunks": book["total_chunks"]
    }


@app.post("/api/books/{book_id}/chat")
async def chat_with_book(book_id: str, request: BookChatRequest):
    """
    Chat with a specific book using RAG.
    
    PSEUDOCODE:
    1. Get user's latest message
    2. Search book's chunks for relevant context
    3. Optionally include web search results
    4. Generate response with citations
    5. Return response with inline citations to book sections
    """
    
    user_message = request.messages[-1].content
    
    # Step 1: Retrieve relevant chunks from this book
    query_embedding = embeddings.encode(user_message)
    
    relevant_chunks = qdrant.search(
        collection_name="librarian_books",
        query_vector=query_embedding,
        query_filter={"must": [{"book_id": book_id}, {"is_metadata": False}]},
        limit=5,
        with_payload=True
    )
    
    # Step 2: Format context with citations
    context_parts = []
    citations = []
    for i, chunk in enumerate(relevant_chunks):
        citation_id = f"[{i+1}]"
        context_parts.append(f"{citation_id} {chunk.payload['content']}")
        citations.append({
            "id": citation_id,
            "chapter": chunk.payload.get("chapter_title"),
            "page": chunk.payload.get("start_page"),
            "content_preview": chunk.payload["content"][:100]
        })
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Optional web search
    web_context = ""
    if request.include_web_search:
        web_results = await perform_web_search(user_message)
        web_context = format_web_results(web_results)
    
    # Step 4: Generate response with LLM
    system_prompt = f"""
    You are a helpful librarian assistant discussing the book "{get_book_title(book_id)}".
    
    Use the following context from the book to answer questions.
    Always cite your sources using the citation markers like [1], [2], etc.
    If the context doesn't contain the answer, say so.
    
    Book Context:
    {context}
    
    {f"Web Search Results: {web_context}" if web_context else ""}
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": m.role, "content": m.content} for m in request.messages])
    
    response = await llm.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        temperature=0.7
    )
    
    return {
        "response": response.choices[0].message.content,
        "citations": citations,
        "web_sources": web_results if request.include_web_search else []
    }


@app.post("/api/books/{book_id}/read-chapter")
async def read_chapter(book_id: str, chapter_number: int):
    """
    Get full text of a specific chapter.
    
    PSEUDOCODE:
    1. Query all chunks for this book and chapter
    2. Reconstruct chapter text in order
    3. Return formatted chapter content
    """
    
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
    
    # Reconstruct chapter text
    chapter_text = "\n\n".join([c.payload["content"] for c in chunks])
    chapter_title = chunks[0].payload.get("chapter_title") if chunks else f"Chapter {chapter_number}"
    
    return {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "content": chapter_text
    }

# ============================================
# CHECKOUT SYSTEM
# ============================================

@app.post("/api/checkout")
async def checkout_book(request: CheckoutRequest):
    """
    Check out a book for download.
    
    PSEUDOCODE:
    1. Verify book exists
    2. Check if already checked out by someone else (high-trust: just track)
    3. Record checkout in database with 2-week expiry
    4. Return download link
    """
    
    # Verify book exists
    book = await get_book_details(request.book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    
    # Record checkout (high-trust environment - no blocking)
    checkout_record = {
        "book_id": request.book_id,
        "user_id": request.user_id,
        "checkout_time": datetime.now(),
        "expiry_time": datetime.now() + timedelta(weeks=2),
        "status": "active"
    }
    
    # Store in database (could be Qdrant or separate SQLite)
    save_checkout_record(checkout_record)
    
    # Generate download link
    download_url = generate_secure_download_link(request.book_id)
    
    return {
        "success": True,
        "book_id": request.book_id,
        "title": book["title"],
        "download_url": download_url,
        "expiry": checkout_record["expiry_time"].isoformat(),
        "message": f"Book checked out until {checkout_record['expiry_time'].strftime('%Y-%m-%d')}"
    }


@app.post("/api/return/{book_id}")
async def return_book(book_id: str, user_id: str):
    """
    Return a checked-out book (manual return before expiry).
    
    PSEUDOCODE:
    1. Find active checkout record
    2. Mark as returned
    3. Confirm return
    """
    
    checkout = get_active_checkout(book_id, user_id)
    if not checkout:
        raise HTTPException(404, "No active checkout found")
    
    checkout["status"] = "returned"
    checkout["return_time"] = datetime.now()
    update_checkout_record(checkout)
    
    return {"success": True, "message": "Book returned successfully"}


@app.get("/api/checkouts")
async def list_checkouts(user_id: str = None):
    """
    List all checkouts, optionally filtered by user.
    
    PSEUDOCODE:
    1. Query checkout records
    2. Filter by user if specified
    3. Include book details
    4. Return list with expiry status
    """
    
    checkouts = get_all_checkouts(user_id=user_id)
    
    return {
        "checkouts": [
            {
                "book_id": c["book_id"],
                "title": get_book_title(c["book_id"]),
                "user_id": c["user_id"],
                "checkout_time": c["checkout_time"],
                "expiry_time": c["expiry_time"],
                "status": c["status"],
                "days_remaining": (c["expiry_time"] - datetime.now()).days
            }
            for c in checkouts
        ]
    }


# Background task: Auto-return expired checkouts
async def auto_return_expired():
    """
    Scheduled task to auto-return books after 2-week expiry.
    
    PSEUDOCODE:
    1. Query all active checkouts past expiry
    2. Mark each as auto-returned
    3. Log the auto-return
    """
    
    expired = get_expired_checkouts()
    
    for checkout in expired:
        checkout["status"] = "auto-returned"
        checkout["return_time"] = datetime.now()
        update_checkout_record(checkout)
        
        log.info(f"Auto-returned book {checkout['book_id']} for user {checkout['user_id']}")

# ============================================
# WEB SEARCH INTEGRATION
# ============================================

async def perform_web_search(query: str) -> list[dict]:
    """
    Perform web search for non-book queries or supplementary info.
    
    PSEUDOCODE:
    1. Detect if query is about something not in library
    2. Use search API (SearXNG, DuckDuckGo, etc.)
    3. Return results with source URLs
    """
    
    # Use local SearXNG instance or similar
    search_results = await searxng_client.search(query, num_results=5)
    
    return [
        {
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "source": r["source"]
        }
        for r in search_results
    ]
```

---

## Frontend (React/Next.js)

```jsx
// pages/index.jsx - Home/Browse Page

function HomePage() {
    /*
    PSEUDOCODE:
    1. Fetch book list on mount
    2. Display grid of book cards
    3. Sidebar with filters (tags, genres, difficulty)
    4. Search bar at top
    5. Click book -> navigate to book detail page
    */
    
    const [books, setBooks] = useState([])
    const [filters, setFilters] = useState({})
    const [searchQuery, setSearchQuery] = useState("")
    
    useEffect(() => {
        fetchBooks(filters).then(setBooks)
    }, [filters])
    
    const handleSearch = async (query) => {
        const results = await searchBooks(query, filters)
        setBooks(results)
    }
    
    return (
        <Layout>
            <SearchBar onSearch={handleSearch} />
            
            <div className="flex">
                <FilterSidebar 
                    filters={filters} 
                    onChange={setFilters}
                    availableTags={["Python", "InfoSec", "AI/ML", ...]}
                    availableGenres={["Technical", "Tutorial", ...]}
                />
                
                <BookGrid books={books} />
            </div>
        </Layout>
    )
}


// pages/books/[id].jsx - Book Detail Page

function BookDetailPage({ bookId }) {
    /*
    PSEUDOCODE:
    1. Fetch book details on mount
    2. Display: title, authors, summary, tags, chapters
    3. "Chat with this book" button -> opens chat panel
    4. "Checkout/Download" button -> initiates checkout
    5. Chapter list -> click to read full chapter
    */
    
    const [book, setBook] = useState(null)
    const [chatOpen, setChatOpen] = useState(false)
    const [chatMessages, setChatMessages] = useState([])
    
    useEffect(() => {
        fetchBookDetails(bookId).then(setBook)
    }, [bookId])
    
    const handleCheckout = async () => {
        const result = await checkoutBook(bookId, currentUser.id)
        if (result.success) {
            window.open(result.download_url)
            showNotification(`Checked out until ${result.expiry}`)
        }
    }
    
    const handleChat = async (message) => {
        setChatMessages([...chatMessages, { role: "user", content: message }])
        
        const response = await chatWithBook(bookId, [...chatMessages, { role: "user", content: message }])
        
        setChatMessages([
            ...chatMessages,
            { role: "user", content: message },
            { role: "assistant", content: response.response, citations: response.citations }
        ])
    }
    
    return (
        <Layout>
            <BookHeader 
                title={book.title}
                authors={book.authors}
                coverImage={book.cover_image}
            />
            
            <div className="grid grid-cols-3 gap-4">
                <div className="col-span-2">
                    <BookSummary summary={book.summary} />
                    <KeyTakeaways items={book.key_takeaways} />
                    <ChapterList 
                        chapters={book.chapter_summaries}
                        onReadChapter={(num) => openChapterModal(num)}
                    />
                </div>
                
                <div className="col-span-1">
                    <BookMetadata 
                        tags={book.tags}
                        genres={book.genres}
                        difficulty={book.difficulty_level}
                        prerequisites={book.prerequisites}
                    />
                    
                    <CheckoutButton 
                        status={book.checkout_status}
                        onCheckout={handleCheckout}
                    />
                    
                    <ChatButton onClick={() => setChatOpen(true)} />
                </div>
            </div>
            
            {chatOpen && (
                <ChatPanel
                    bookTitle={book.title}
                    messages={chatMessages}
                    onSendMessage={handleChat}
                    onClose={() => setChatOpen(false)}
                />
            )}
        </Layout>
    )
}


// components/ChatPanel.jsx - Book Chat Interface

function ChatPanel({ bookTitle, messages, onSendMessage, onClose }) {
    /*
    PSEUDOCODE:
    1. Display chat history with citations
    2. Input field for new messages
    3. Toggle for "include web search"
    4. Citations shown as clickable references
    */
    
    const [input, setInput] = useState("")
    const [includeWebSearch, setIncludeWebSearch] = useState(false)
    
    return (
        <div className="chat-panel">
            <div className="chat-header">
                <h3>Chat with "{bookTitle}"</h3>
                <button onClick={onClose}>Close</button>
            </div>
            
            <div className="chat-messages">
                {messages.map((msg, i) => (
                    <ChatMessage 
                        key={i}
                        role={msg.role}
                        content={msg.content}
                        citations={msg.citations}
                    />
                ))}
            </div>
            
            <div className="chat-input">
                <label>
                    <input 
                        type="checkbox" 
                        checked={includeWebSearch}
                        onChange={(e) => setIncludeWebSearch(e.target.checked)}
                    />
                    Include web search
                </label>
                
                <input 
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about this book..."
                    onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                            onSendMessage(input, includeWebSearch)
                            setInput("")
                        }
                    }}
                />
            </div>
        </div>
    )
}


// pages/recommend.jsx - Recommendation Page

function RecommendPage() {
    /*
    PSEUDOCODE:
    1. Natural language input for what user is looking for
    2. Submit -> LLM processes and returns recommendations
    3. Display recommendations with explanations
    4. Click recommendation -> go to book detail
    */
    
    const [query, setQuery] = useState("")
    const [recommendations, setRecommendations] = useState([])
    const [loading, setLoading] = useState(false)
    
    const handleRecommend = async () => {
        setLoading(true)
        const results = await getRecommendations(query)
        setRecommendations(results.recommendations)
        setLoading(false)
    }
    
    return (
        <Layout>
            <h1>Get Book Recommendations</h1>
            
            <div className="recommend-input">
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Describe what you're looking for... e.g., 'I want to learn Python for security testing'"
                />
                <button onClick={handleRecommend} disabled={loading}>
                    {loading ? "Finding books..." : "Get Recommendations"}
                </button>
            </div>
            
            {recommendations.length > 0 && (
                <div className="recommendations">
                    {recommendations.map((rec) => (
                        <RecommendationCard
                            key={rec.book_id}
                            bookId={rec.book_id}
                            reason={rec.recommendation_reason}
                        />
                    ))}
                </div>
            )}
        </Layout>
    )
}
```

---

## Key Features Summary

1. **Browse/Search**: Grid view of all books, semantic search, filtering by tags/genres/difficulty
2. **Book Details**: Full metadata, summaries, chapter list, key quotes, themes
3. **Per-Book Chat**: RAG-based Q&A with citations to specific sections
4. **Recommendations**: Natural language query -> LLM-powered book suggestions
5. **Checkout System**: High-trust download with 2-week auto-return tracking
6. **Web Search**: Fallback for queries outside library scope, with source citations
7. **Chapter Reading**: Full chapter text retrieval on demand

## Future Enhancements (Stubs)
- TTS: Piper/XTTS-v2 integration for audio playback
- SSO: Slack/Authentik authentication
- Reading Progress: Track user's reading position
- Notes/Highlights: User annotations per book
