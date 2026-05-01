from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import json
import logging
import os
import asyncio
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
import uvicorn
from dotenv import load_dotenv

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_history")

load_dotenv()

app = FastAPI()

# --- Inference backend config ---
# We talk to Open WebUI's Ollama passthrough. The /ollama/* routes forward
# requests to the upstream Ollama instance with no transformation, so the
# request/response shapes are byte-for-byte identical to native Ollama —
# we just need to point at OWUI's host and add a Bearer token.
#
# OLLAMA_URL is the BASE URL of the OWUI host (no /ollama suffix). We append
# /ollama/api/embed (and /ollama/api/embeddings as legacy fallback) below.
OLLAMA_URL = os.getenv('OLLAMA_URL', 'https://oui.gpu.garden')
MODEL_ID_EMBEDDING = os.getenv('MODEL_ID_EMBEDDING', 'qwen3-embedding:latest')

# Where saved images live on disk. The DB only stores relative paths here;
# bytes go to the filesystem because SQLite isn't a great BLOB store at scale.
# Layout: ./image_store/<channel_id>/<saved_image_id>.<ext>
IMAGE_STORE_DIR = os.getenv('IMAGE_STORE_DIR', './image_store')

# Optional auto-prune: when the on-disk store exceeds this size, oldest images
# are deleted (DB row + file) until we're back under the cap. 0 disables.
# Note: we measure the directory size, not just the count of rows, since the
# DB and the FS can drift if a bot crashed between writes.
IMAGE_STORE_MAX_MB = int(os.getenv('IMAGE_STORE_MAX_MB', '0'))

# Bearer token for OWUI. Generate in OWUI > Settings > Account > API Keys
# and put in .env as OPENWEBUI_API_KEY=sk-...
# The chat history service uses its own env var (separate from the bot's
# per-bot key) because conceptually any bot's chat history can flow through
# this same service, and embedding calls are infrastructure-level.
OPENWEBUI_API_KEY = os.getenv('OPENWEBUI_API_KEY_Cirno')
if not OPENWEBUI_API_KEY:
    logger.warning("OPENWEBUI_API_KEY not set — embedding calls will likely 401. "
                   "Set it in your .env file.")

HEADERS = {"Content-Type": "application/json"}
if OPENWEBUI_API_KEY:
    HEADERS["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"


# user_id canonicalization.
# Discord IDs are 17-20 digit snowflakes. Older versions of the bot stored
# user_ids in their Discord-mention form ("<@396774290588041228>" or
# "<@!396...>") but the bot's queries pass bare IDs. Result: stored values
# never match queried values, every WHERE user_id = ? returns 0 rows, and
# every personality summary call silently does nothing.
#
# Fix: normalize at the service boundary — any incoming user_id (whether
# from a path param or a JSON body) gets stripped to its bare numeric form
# before touching SQL. Bot side keeps using whatever form is convenient.
import re as _re_uid
_USER_ID_WRAP_RE = _re_uid.compile(r'^<@!?(\d{15,25})>$')


def _canon_uid(uid):
    """
    Strip <@id> / <@!id> wrapping if present, returning the bare ID. Pure
    digit strings pass through unchanged. Anything else passes through as-is
    so we don't accidentally mangle legitimate non-mention strings (like
    a future Discord ID format).
    """
    if uid is None:
        return None
    s = str(uid).strip()
    m = _USER_ID_WRAP_RE.match(s)
    if m:
        return m.group(1)
    return s


def _wrap_uid(uid):
    """
    Re-wrap a bare numeric user ID as <@id> for API output. The bot's prompt
    assembly compares strings like `f"<@{author.id}>"` against the `user`
    field of returned rows; without re-wrapping, those comparisons all break.

    Pass through None and already-wrapped strings unchanged so this is safe
    to apply unconditionally.
    """
    if uid is None:
        return None
    s = str(uid).strip()
    if not s:
        return s
    if s.startswith("<@"):
        return s  # already wrapped
    if s.isdigit():
        return f"<@{s}>"
    return s  # not numeric — leave alone


def _is_opted_out(channel_id, user_id):
    """
    Returns True if the (channel_id, user_id) pair is in the opted_out table.
    Called from /add_message and /save_image to silently drop writes from
    opted-out users. Pass user_id in any form — we canonicalize internally.
    """
    uid = _canon_uid(user_id)
    if not uid:
        return False
    c.execute("SELECT 1 FROM opted_out WHERE channel_id = ? AND user_id = ? LIMIT 1",
              (str(channel_id), uid))
    return c.fetchone() is not None

# --- Database setup ---
# check_same_thread=False lets FastAPI share the connection across async handlers.
# For higher concurrency, consider aiosqlite or switching to Postgres.
#
# IMPORTANT: When you change MODEL_ID_EMBEDDING, the new vectors live in a
# different vector space than the old ones — cosine similarity between them
# will be garbage. Either (a) wipe chat_history.db before first run, or
# (b) keep separate DBs per embedding model.
conn = sqlite3.connect('chat_history.db', check_same_thread=False)
c = conn.cursor()

# Main message table.
# `discord_message_id` (added in v2) lets the bot dedupe its own current
# message out of fetched history reliably. Nullable for backward-compat
# with rows written before the upgrade.
c.execute('''CREATE TABLE IF NOT EXISTS messages
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              channel_id TEXT,
              user_id TEXT,
              timestamp TEXT,
              replying_to TEXT,
              content TEXT,
              embedding TEXT,
              role TEXT,
              discord_message_id TEXT)''')

# Migration: add discord_message_id column to existing databases that don't have it.
try:
    c.execute("SELECT discord_message_id FROM messages LIMIT 1")
except sqlite3.OperationalError:
    logger.info("Migrating: adding discord_message_id column to messages table")
    c.execute("ALTER TABLE messages ADD COLUMN discord_message_id TEXT")
    conn.commit()

# Personality summary table — one row per (channel_id, user_id).
# `summary` is the freeform text the bot generated about this user.
# `last_summarized_at` and `messages_at_last_summary` let us decide when
# to re-summarize.
c.execute('''CREATE TABLE IF NOT EXISTS personality_summaries
             (channel_id TEXT,
              user_id TEXT,
              summary TEXT,
              last_summarized_at TEXT,
              messages_at_last_summary INTEGER DEFAULT 0,
              PRIMARY KEY (channel_id, user_id))''')

# Saved images table — image memory for "Cirno remembers that meme" RAG.
# Image bytes live on disk; this row stores metadata + the caption embedding
# we retrieve against. file_size_bytes lets us compute store size for prune
# decisions without statting the FS.
c.execute('''CREATE TABLE IF NOT EXISTS saved_images
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              channel_id TEXT,
              user_id TEXT,
              discord_message_id TEXT,
              image_path TEXT,
              media_type TEXT,
              caption TEXT,
              embedding TEXT,
              file_size_bytes INTEGER,
              saved_at TEXT)''')

# Opt-out registry. A row here means the user has explicitly opted out of
# data collection in this channel — every subsequent /add_message and
# /save_image silently no-ops for them. The opt-out is permanent until they
# call /clear_opt_out (the bot's `/optin` command). Per-channel scope was
# the design decision: opting out of channel A doesn't affect channel B.
c.execute('''CREATE TABLE IF NOT EXISTS opted_out
             (channel_id TEXT,
              user_id TEXT,
              opted_out_at TEXT,
              PRIMARY KEY (channel_id, user_id))''')

# Server emoji captions. Populated incrementally: when the bot observes a
# custom emoji it hasn't seen before, it captions the image once and stores
# the result here. Subsequent uses of the same emoji ID skip captioning.
# The caption embedding is what RAG searches against on every chat turn,
# so the bot can suggest semantically-relevant emojis Cirno can use in her
# reply.
#
# Schema notes:
#   - emoji_id is the Discord snowflake (TEXT to avoid integer-overflow
#     concerns on 32-bit Python builds and to match how we pass it elsewhere)
#   - animated is 0/1; we use it to render the right `<a:name:id>` vs
#     `<:name:id>` syntax
#   - last_seen_at lets us prune emojis that have been deleted from the guild
#     by checking gaps relative to bot startup (or just delete them when
#     on_guild_emojis_update fires)
c.execute('''CREATE TABLE IF NOT EXISTS server_emojis
             (emoji_id TEXT PRIMARY KEY,
              guild_id TEXT,
              name TEXT,
              animated INTEGER DEFAULT 0,
              caption TEXT,
              embedding TEXT,
              captioned_at TEXT,
              last_seen_at TEXT)''')

# Helpful indexes for the queries we actually run
c.execute('CREATE INDEX IF NOT EXISTS idx_channel_ts ON messages(channel_id, timestamp DESC)')
c.execute('CREATE INDEX IF NOT EXISTS idx_channel_role ON messages(channel_id, role)')
c.execute('CREATE INDEX IF NOT EXISTS idx_channel_user ON messages(channel_id, user_id)')
c.execute('CREATE INDEX IF NOT EXISTS idx_dmid ON messages(discord_message_id)')
c.execute('CREATE INDEX IF NOT EXISTS idx_img_channel ON saved_images(channel_id, saved_at DESC)')
c.execute('CREATE INDEX IF NOT EXISTS idx_img_user ON saved_images(channel_id, user_id)')
c.execute('CREATE INDEX IF NOT EXISTS idx_optout ON opted_out(channel_id, user_id)')
c.execute('CREATE INDEX IF NOT EXISTS idx_emoji_guild ON server_emojis(guild_id)')
conn.commit()

# Make sure the image store directory exists. Created lazily per-channel below.
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)


def _migrate_canonicalize_user_ids():
    """
    One-time migration: any messages/personality_summaries/saved_images rows
    that hold wrapped Discord mentions ("<@123>") get rewritten to the bare
    numeric form. Cheap because rare — an UPDATE with a LIKE filter that
    most tables won't match at all.

    Without this, rows recorded under the old buggy bot remain unmatched
    by the new bot's bare-id queries, and personality summaries / image
    deletes / user-message lookups appear broken even though the code
    itself is fixed.
    """
    try:
        # messages: the most affected table — every user_id was wrapped.
        c.execute("UPDATE messages SET user_id = "
                  "REPLACE(REPLACE(REPLACE(user_id, '<@!', ''), '<@', ''), '>', '') "
                  "WHERE user_id LIKE '<@%>'")
        msg_n = c.rowcount
        # replying_to is also stored as <@id> — normalize it too so queries
        # that join on it match.
        c.execute("UPDATE messages SET replying_to = "
                  "REPLACE(REPLACE(REPLACE(replying_to, '<@!', ''), '<@', ''), '>', '') "
                  "WHERE replying_to LIKE '<@%>'")
        rt_n = c.rowcount
        # personality_summaries: same fix
        c.execute("UPDATE personality_summaries SET user_id = "
                  "REPLACE(REPLACE(REPLACE(user_id, '<@!', ''), '<@', ''), '>', '') "
                  "WHERE user_id LIKE '<@%>'")
        ps_n = c.rowcount
        # saved_images: same fix
        c.execute("UPDATE saved_images SET user_id = "
                  "REPLACE(REPLACE(REPLACE(user_id, '<@!', ''), '<@', ''), '>', '') "
                  "WHERE user_id LIKE '<@%>'")
        img_n = c.rowcount
        conn.commit()
        if msg_n or rt_n or ps_n or img_n:
            logger.info(f"user_id migration: rewrote messages={msg_n}, "
                        f"replying_to={rt_n}, personalities={ps_n}, "
                        f"saved_images={img_n}")
    except sqlite3.OperationalError as e:
        logger.warning(f"user_id migration skipped: {e}")


_migrate_canonicalize_user_ids()


class Message(BaseModel):
    channel_id: str
    user_id: str
    content: str
    replying_to: Optional[str] = None
    role: str
    discord_message_id: Optional[str] = None  # Discord snowflake, if available


class SummaryUpsert(BaseModel):
    channel_id: str
    user_id: str
    summary: str


class SavedImageUpload(BaseModel):
    """Payload for /save_image. Image bytes are sent as base64 since FastAPI
    JSON bodies can't carry raw binary. Keeps the API consistent with the
    rest of the service (everything else is JSON)."""
    channel_id: str
    user_id: str
    discord_message_id: str
    media_type: str          # "image/png", "image/jpeg", etc.
    extension: str           # "png", "jpg", "webp" — used in the on-disk filename
    image_b64: str           # raw bytes encoded as base64 (no data URI prefix)
    caption: str             # captioner's description of what's in the image


class SaveEmojiUpload(BaseModel):
    """Payload for /save_emoji. Emoji bytes are NOT stored — only the caption
    embedding. Discord renders `<:name:id>` itself when the bot has access
    to the emoji, so we don't need to keep the image around."""
    emoji_id: str
    guild_id: str
    name: str
    animated: bool
    caption: str


async def get_embedding(text: str):
    """
    Fetch an embedding from Open WebUI's Ollama passthrough.

    We try /ollama/api/embed first (current Ollama API; supports batching,
    returns list under "embeddings"). If OWUI's underlying Ollama is older
    and 404s, we fall back to /ollama/api/embeddings (legacy single-prompt
    endpoint that returns "embedding" singular).

    On failure, returns None and logs the SPECIFIC reason. Common causes:
      - Ollama timeout (model not loaded, slow GPU). Cloudflare 524.
      - Model name typo / not pulled. 404 from upstream Ollama.
      - Auth issue (401/403). Token expired or revoked.
      - Response shape mismatch — unusual but possible across Ollama versions.
    """
    if not text or not text.strip():
        logger.warning("get_embedding called with empty text")
        return None

    base = OLLAMA_URL.rstrip('/')
    # Generous timeout — qwen3-embedding cold-load can take 30s+ on first call.
    # The fixed 30s default of aiohttp would 524 us before the model finishes
    # loading. We mirror the chat-side approach: large total, large sock_read.
    timeout = aiohttp.ClientTimeout(total=300, sock_read=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Preferred: current endpoint
        try:
            async with session.post(
                f"{base}/ollama/api/embed",
                headers=HEADERS,
                json={"model": MODEL_ID_EMBEDDING, "input": text},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Newer Ollama: {"embeddings": [[...]]}
                    embeddings = data.get("embeddings")
                    if embeddings and len(embeddings) > 0:
                        return embeddings[0]
                    # Some forks/versions return {"embedding": [...]} (singular)
                    # even on /api/embed. Handle that gracefully.
                    single = data.get("embedding")
                    if single and isinstance(single, list):
                        logger.info("/ollama/api/embed returned singular 'embedding' "
                                    "key — accepting it")
                        return single
                    logger.error(f"/ollama/api/embed: 200 OK but no embedding in "
                                 f"response. keys={list(data.keys())} "
                                 f"text_preview={text[:60]!r}")
                elif response.status == 404:
                    logger.info("/ollama/api/embed not found, trying legacy /ollama/api/embeddings")
                elif response.status in (401, 403):
                    body = await response.text()
                    logger.error(f"Auth failure on /ollama/api/embed: {response.status} {body[:300]}. "
                                 f"Check OPENWEBUI_API_KEY env var.")
                    return None
                elif response.status in (502, 503, 504, 524):
                    body = await response.text()
                    logger.error(f"Upstream timeout/unavailable on /ollama/api/embed: "
                                 f"{response.status}. Likely qwen3-embedding model is "
                                 f"slow to load or Ollama is overloaded. body={body[:300]}")
                else:
                    body = await response.text()
                    logger.error(f"/ollama/api/embed returned {response.status}: {body[:300]}")
        except asyncio.TimeoutError:
            logger.error(f"/ollama/api/embed timed out after 300s — qwen3-embedding "
                         f"model may be stuck loading or unavailable")
            return None
        except Exception as e:
            logger.error(f"/ollama/api/embed request failed: {e}")

        # Legacy fallback
        try:
            async with session.post(
                f"{base}/ollama/api/embeddings",
                headers=HEADERS,
                json={"model": MODEL_ID_EMBEDDING, "prompt": text},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    emb = data.get("embedding")
                    if emb:
                        return emb
                    logger.error(f"Legacy /ollama/api/embeddings: 200 OK but no "
                                 f"embedding in response. keys={list(data.keys())}")
                    return None
                body = await response.text()
                logger.error(f"Legacy /ollama/api/embeddings returned {response.status}: {body[:300]}")
        except asyncio.TimeoutError:
            logger.error(f"Legacy /ollama/api/embeddings timed out after 300s")
        except Exception as e:
            logger.error(f"Legacy /ollama/api/embeddings request failed: {e}")

    return None


@app.post("/add_message")
async def add_message(message: Message):
    # Canonicalize ids — see _canon_uid docstring. Without this, the bot's
    # bare-id queries never match and personality summaries silently fail.
    user_id_canon = _canon_uid(message.user_id)
    replying_to_canon = _canon_uid(message.replying_to)

    # Opt-out check — if this user opted out of this channel, drop the write
    # silently. Returning success keeps the bot's flow happy without revealing
    # to other observers that someone has opted out.
    if _is_opted_out(message.channel_id, user_id_canon):
        return {"status": "skipped", "reason": "opted_out"}

    timestamp = datetime.now(timezone.utc).isoformat()
    embedding = await get_embedding(message.content)

    if embedding is None:
        logger.error(f"Could not generate embedding for message in channel {message.channel_id}")
        # Still store the message — we just lose semantic searchability for this row.
        # get_relevant_context filters out rows with null embeddings so this is safe.

    # Also blank out replying_to if the TARGET user opted out of this channel.
    # Without this, an opted-out user's mention can keep appearing in the
    # `replying_to` field of other people's messages even after the opt-out.
    if replying_to_canon and _is_opted_out(message.channel_id, replying_to_canon):
        replying_to_canon = None

    c.execute('''INSERT INTO messages
                 (channel_id, user_id, timestamp, replying_to, content, embedding, role,
                  discord_message_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (message.channel_id, user_id_canon, timestamp, replying_to_canon,
               message.content,
               json.dumps(embedding) if embedding is not None else None,
               message.role,
               message.discord_message_id))
    conn.commit()

    return {"status": "success", "message_id": c.lastrowid,
            "discord_message_id": message.discord_message_id}


@app.get("/get_conversation_history/{channel_id}")
async def get_conversation_history(channel_id: str, limit: int = 10):
    c.execute('''SELECT id, user_id, timestamp, replying_to, content, role,
                        discord_message_id
                 FROM messages WHERE channel_id = ?
                 ORDER BY timestamp DESC LIMIT ?''', (channel_id, limit))
    messages = c.fetchall()

    return [{"id": msg[0], "user": _wrap_uid(msg[1]), "timestamp": msg[2],
             "replying_to": _wrap_uid(msg[3]),
             "content": msg[4], "role": msg[5], "discord_message_id": msg[6]}
            for msg in reversed(messages)]


@app.get("/get_relevant_context/{channel_id}")
async def get_relevant_context(
    channel_id: str,
    query: str,
    limit: int = 5,
    exclude_role: List[str] = Query(default_factory=list),
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    """
    Semantic search over this channel's history.

    Args:
        exclude_role: zero or more roles to drop from results. Pass multiple
                      times in the URL: ?exclude_role=user&exclude_role=assistant
        after:        ISO-8601 datetime; only messages newer than this are
                      considered.
        before:       ISO-8601 datetime; only messages older than this are
                      considered.

    The time filters are applied at the SQL level BEFORE the vector search,
    so a query like "what did Alice say last week" doesn't waste effort
    embedding messages outside the time range.
    """
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    # Build SQL dynamically based on which filters are active.
    # Cleaner than nested if/else and harder to mess up.
    sql_parts = [
        "SELECT id, channel_id, user_id, timestamp, replying_to, content, embedding, role, "
        "discord_message_id "
        "FROM messages WHERE channel_id = ? AND embedding IS NOT NULL"
    ]
    params: list = [channel_id]

    if exclude_role:
        # Build the right number of placeholders for the IN clause
        placeholders = ",".join("?" for _ in exclude_role)
        sql_parts.append(f"AND role NOT IN ({placeholders})")
        params.extend(exclude_role)

    if after:
        sql_parts.append("AND timestamp >= ?")
        params.append(after)
    if before:
        sql_parts.append("AND timestamp <= ?")
        params.append(before)

    sql = " ".join(sql_parts)
    c.execute(sql, params)
    rows = c.fetchall()
    if not rows:
        return []

    # Defensive: if rows happen to contain mixed-dimension embeddings (e.g. from
    # an embedding-model swap with a partially-wiped DB), cosine_similarity
    # would crash. Filter out any rows whose vector dimensionality doesn't
    # match the query.
    query_dim = len(query_embedding)
    valid_rows = []
    valid_embeddings = []
    for r in rows:
        try:
            v = np.array(json.loads(r[6]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if v.shape[0] != query_dim:
            # Stale row from a different embedding model. Skip silently —
            # logging every one would be very noisy after a model change.
            continue
        valid_rows.append(r)
        valid_embeddings.append(v)

    if not valid_rows:
        return []

    similarities = cosine_similarity([query_embedding], valid_embeddings)[0]

    # Cap limit so we don't return absurd numbers if a buggy client sends limit=10000
    limit = max(1, min(limit, 50))
    top_indices = np.argsort(similarities)[-limit:][::-1]

    # Schema: (id, channel_id, user_id, timestamp, replying_to, content, embedding, role,
    #         discord_message_id)
    out = []
    for i in top_indices:
        r = valid_rows[i]
        out.append({
            "id": r[0],
            "user": _wrap_uid(r[2]),
            "timestamp": r[3],
            "replying_to": _wrap_uid(r[4]),
            "content": r[5],
            "role": r[7],
            "discord_message_id": r[8],
            "score": float(similarities[i]),  # let client apply its own threshold
        })
    return out


@app.get("/search_user_messages/{channel_id}/{user_id}")
async def search_user_messages(
    channel_id: str,
    user_id: str,
    query: str,
    limit: int = 10,
):
    """
    Semantic search restricted to ONE user's messages in this channel.
    Used by the bot's "what did @user say about X" intent path. Same shape
    as /get_relevant_context but with an extra WHERE user_id = ? filter
    applied at the SQL level — much more efficient than fetching all and
    post-filtering, and (more importantly) much more relevant: we get the
    top-N matches FROM that user, not the top-N from the channel that
    happen to be from that user.
    """
    user_id = _canon_uid(user_id)
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    c.execute('''SELECT id, channel_id, user_id, timestamp, replying_to, content,
                        embedding, role, discord_message_id
                 FROM messages
                 WHERE channel_id = ? AND user_id = ? AND embedding IS NOT NULL''',
              (channel_id, user_id))
    rows = c.fetchall()
    if not rows:
        return []

    query_dim = len(query_embedding)
    valid_rows = []
    valid_embeddings = []
    for r in rows:
        try:
            v = np.array(json.loads(r[6]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if v.shape[0] != query_dim:
            continue
        valid_rows.append(r)
        valid_embeddings.append(v)

    if not valid_rows:
        return []

    similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
    limit = max(1, min(limit, 30))
    top_indices = np.argsort(similarities)[-limit:][::-1]
    out = []
    for i in top_indices:
        r = valid_rows[i]
        out.append({
            "id": r[0],
            "user": _wrap_uid(r[2]),
            "timestamp": r[3],
            "replying_to": _wrap_uid(r[4]),
            "content": r[5],
            "role": r[7],
            "discord_message_id": r[8],
            "score": float(similarities[i]),
        })
    return out


@app.get("/get_messages_about_user/{channel_id}/{user_id}")
async def get_messages_about_user(
    channel_id: str,
    user_id: str,
    limit: int = 20,
):
    """
    Return messages where OTHER users replied to or mentioned `user_id`.
    Used by the "what do you think of @user" intent — to form an opinion of
    someone, the bot needs both what they said and what others said about
    them. This endpoint provides the latter half.

    "Mentioned" here means either (a) `replying_to = user_id` (a direct
    reply), or (b) the message content contains `<@user_id>`. We avoid
    full-text search on content for performance — it's a LIKE scan, but
    the channel filter already keeps the working set small.
    """
    user_id = _canon_uid(user_id)
    mention_token = f"<@{user_id}>"
    mention_token_alt = f"<@!{user_id}>"  # legacy mention form
    limit = max(1, min(limit, 100))

    c.execute('''SELECT id, channel_id, user_id, timestamp, replying_to, content,
                        role, discord_message_id
                 FROM messages
                 WHERE channel_id = ?
                   AND user_id != ?
                   AND (replying_to = ?
                        OR content LIKE ?
                        OR content LIKE ?)
                 ORDER BY timestamp DESC
                 LIMIT ?''',
              (channel_id, user_id, user_id,
               f"%{mention_token}%", f"%{mention_token_alt}%", limit))
    rows = c.fetchall()
    return [
        {
            "id": r[0],
            "user": _wrap_uid(r[2]),
            "timestamp": r[3],
            "replying_to": _wrap_uid(r[4]),
            "content": r[5],
            "role": r[6],
            "discord_message_id": r[7],
        }
        for r in reversed(rows)  # oldest first for natural reading order
    ]


@app.get("/get_user_messages/{channel_id}/{user_id}")
async def get_user_messages(channel_id: str, user_id: str, limit: int = 100):
    """
    Return the most recent messages from a single user in a channel.
    Used for personality summarization.
    """
    user_id = _canon_uid(user_id)
    c.execute('''SELECT id, user_id, timestamp, replying_to, content, role,
                        discord_message_id
                 FROM messages
                 WHERE channel_id = ? AND user_id = ?
                 ORDER BY timestamp DESC LIMIT ?''',
              (channel_id, user_id, limit))
    rows = c.fetchall()
    return [{"id": r[0], "user": _wrap_uid(r[1]), "timestamp": r[2],
             "replying_to": _wrap_uid(r[3]),
             "content": r[4], "role": r[5], "discord_message_id": r[6]}
            for r in reversed(rows)]


@app.get("/count_user_messages/{channel_id}/{user_id}")
async def count_user_messages(channel_id: str, user_id: str):
    """How many messages from this user are in this channel? Used to decide
    when it's time to re-run the personality summary."""
    user_id = _canon_uid(user_id)
    c.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    return {"count": c.fetchone()[0]}


@app.get("/get_personality/{channel_id}/{user_id}")
async def get_personality(channel_id: str, user_id: str):
    """Return the stored personality summary for this user, or null if none."""
    user_id = _canon_uid(user_id)
    c.execute('''SELECT summary, last_summarized_at, messages_at_last_summary
                 FROM personality_summaries
                 WHERE channel_id = ? AND user_id = ?''',
              (channel_id, user_id))
    row = c.fetchone()
    if not row:
        return None
    return {
        "summary": row[0],
        "last_summarized_at": row[1],
        "messages_at_last_summary": row[2] or 0,
    }


@app.post("/set_personality")
async def set_personality(payload: SummaryUpsert):
    """Upsert a personality summary. Updates last_summarized_at and message count."""
    now = datetime.now(timezone.utc).isoformat()
    user_id = _canon_uid(payload.user_id)
    # Get current message count for this user/channel so we know when to refresh
    c.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ? AND user_id = ?",
              (payload.channel_id, user_id))
    msg_count = c.fetchone()[0]

    c.execute('''INSERT INTO personality_summaries
                   (channel_id, user_id, summary, last_summarized_at, messages_at_last_summary)
                 VALUES (?, ?, ?, ?, ?)
                 ON CONFLICT(channel_id, user_id) DO UPDATE SET
                   summary = excluded.summary,
                   last_summarized_at = excluded.last_summarized_at,
                   messages_at_last_summary = excluded.messages_at_last_summary''',
              (payload.channel_id, user_id, payload.summary, now, msg_count))
    conn.commit()
    return {"status": "success", "messages_at_last_summary": msg_count}


def _channel_image_dir(channel_id: str) -> str:
    """Returns the directory where this channel's images live; creates it if needed."""
    d = os.path.join(IMAGE_STORE_DIR, str(channel_id))
    os.makedirs(d, exist_ok=True)
    return d


def _maybe_prune_image_store():
    """
    If IMAGE_STORE_MAX_MB > 0 and the on-disk store exceeds that cap, delete
    the oldest saved_images rows (and their files) until we're under the cap.
    Called after every successful /save_image. Cheap when the store is small.
    """
    if IMAGE_STORE_MAX_MB <= 0:
        return
    cap_bytes = IMAGE_STORE_MAX_MB * 1024 * 1024

    # Use the DB's tracked sizes — fast and lets us pick the oldest by saved_at.
    # If the FS and DB drift, we may overshoot or undershoot the cap slightly,
    # but the prune still makes forward progress.
    c.execute("SELECT COALESCE(SUM(file_size_bytes), 0) FROM saved_images")
    total = int(c.fetchone()[0] or 0)
    if total <= cap_bytes:
        return

    logger.info(f"Image store at {total} bytes (cap {cap_bytes}) — pruning")
    # Walk oldest-first, deleting until we're back under
    c.execute("SELECT id, image_path, file_size_bytes FROM saved_images "
              "ORDER BY saved_at ASC")
    rows = c.fetchall()
    deleted_count = 0
    for row_id, image_path, size in rows:
        if total <= cap_bytes:
            break
        try:
            full_path = os.path.join(IMAGE_STORE_DIR, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        except OSError as e:
            logger.warning(f"prune: could not remove {image_path}: {e}")
        c.execute("DELETE FROM saved_images WHERE id = ?", (row_id,))
        total -= int(size or 0)
        deleted_count += 1
    conn.commit()
    logger.info(f"Image store prune: deleted {deleted_count} images, now {total} bytes")


@app.post("/save_image")
async def save_image(payload: SavedImageUpload):
    """
    Save one image to memory:
      1. Decode and write bytes to ./image_store/<channel_id>/<id>.<ext>
      2. Embed the caption (qwen3) for retrieval
      3. Insert a saved_images row pointing at the file
      4. Maybe prune oldest if we're over the cap
    Returns the new row's id and on-disk path (relative to IMAGE_STORE_DIR).
    """
    import base64 as _b64

    # Opt-out check — silently no-op for opted-out users. Same pattern as
    # /add_message; we return a 200 so the bot's caller doesn't error.
    if _is_opted_out(payload.channel_id, payload.user_id):
        return {"status": "skipped", "reason": "opted_out"}

    # 1. Embed first — if embedding fails we don't want a stranded file on disk.
    #    Caption must be non-empty (we wouldn't be able to retrieve a captionless image).
    caption = (payload.caption or "").strip()
    if not caption:
        raise HTTPException(status_code=400, detail="caption is required and cannot be empty")
    embedding = await get_embedding(caption)
    if embedding is None:
        raise HTTPException(status_code=503, detail="could not generate caption embedding")

    # 2. Insert the row first WITHOUT image_path — this gets us an autoincrement id
    #    we can use in the filename. Fill image_path in a second update step.
    saved_at = datetime.now(timezone.utc).isoformat()
    try:
        raw = _b64.b64decode(payload.image_b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")

    file_size = len(raw)
    # Sanitize extension — only allow safe alphanumeric extensions to prevent
    # path traversal via cleverly-crafted "extensions" like "../../foo".
    ext = "".join(c for c in (payload.extension or "png").lower() if c.isalnum())
    if not ext or len(ext) > 5:
        ext = "png"

    c.execute('''INSERT INTO saved_images
                 (channel_id, user_id, discord_message_id, image_path, media_type,
                  caption, embedding, file_size_bytes, saved_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (payload.channel_id, _canon_uid(payload.user_id), payload.discord_message_id,
               "",  # placeholder, updated below once we have the id
               payload.media_type, caption, json.dumps(embedding),
               file_size, saved_at))
    new_id = c.lastrowid

    # 3. Compute the relative path (relative to IMAGE_STORE_DIR) and write bytes.
    rel_path = os.path.join(str(payload.channel_id), f"{new_id}.{ext}")
    abs_path = os.path.join(IMAGE_STORE_DIR, rel_path)
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(raw)
    except OSError as e:
        # Roll back the row — we don't want a DB entry pointing at nothing
        c.execute("DELETE FROM saved_images WHERE id = ?", (new_id,))
        conn.commit()
        raise HTTPException(status_code=500, detail=f"could not write image: {e}")

    c.execute("UPDATE saved_images SET image_path = ? WHERE id = ?", (rel_path, new_id))
    conn.commit()

    _maybe_prune_image_store()

    return {"status": "success", "id": new_id, "image_path": rel_path,
            "file_size_bytes": file_size}


@app.get("/get_relevant_images/{channel_id}")
async def get_relevant_images(channel_id: str, query: str, limit: int = 3):
    """
    Semantic search over saved image captions. Returns rows with caption +
    metadata; the bot fetches the bytes separately via /get_image_bytes/{id}
    only when it actually decides to re-attach. Avoids shipping image data
    on every retrieval.
    """
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    c.execute('''SELECT id, channel_id, user_id, discord_message_id, image_path,
                        media_type, caption, embedding, file_size_bytes, saved_at
                 FROM saved_images
                 WHERE channel_id = ? AND embedding IS NOT NULL''',
              (channel_id,))
    rows = c.fetchall()
    if not rows:
        return []

    query_dim = len(query_embedding)
    valid_rows = []
    valid_embeddings = []
    for r in rows:
        try:
            v = np.array(json.loads(r[7]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if v.shape[0] != query_dim:
            continue
        valid_rows.append(r)
        valid_embeddings.append(v)

    if not valid_rows:
        return []

    similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
    limit = max(1, min(limit, 10))
    top_indices = np.argsort(similarities)[-limit:][::-1]

    out = []
    for i in top_indices:
        r = valid_rows[i]
        out.append({
            "id": r[0],
            "channel_id": r[1],
            "user_id": r[2],
            "discord_message_id": r[3],
            "image_path": r[4],
            "media_type": r[5],
            "caption": r[6],
            "file_size_bytes": r[8],
            "saved_at": r[9],
            "score": float(similarities[i]),
        })
    return out


@app.get("/get_image_bytes/{image_id}")
async def get_image_bytes(image_id: int):
    """
    Return the raw bytes of a saved image (base64-encoded, with media_type)
    so the bot can re-attach it. Separate call from get_relevant_images to
    keep the retrieval path light — we only ship bytes when we're actually
    going to use them.
    """
    import base64 as _b64
    c.execute("SELECT image_path, media_type, caption FROM saved_images WHERE id = ?",
              (image_id,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    image_path, media_type, caption = row
    abs_path = os.path.join(IMAGE_STORE_DIR, image_path)
    if not os.path.exists(abs_path):
        # DB row exists but file is gone — clean up the orphan row so we don't
        # keep retrieving it. Could happen if the FS was wiped or pruned out
        # of band.
        logger.warning(f"saved_image id={image_id} is orphaned (file missing); removing row")
        c.execute("DELETE FROM saved_images WHERE id = ?", (image_id,))
        conn.commit()
        raise HTTPException(status_code=404, detail="image file missing on disk")
    try:
        with open(abs_path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not read image: {e}")
    return {
        "id": image_id,
        "media_type": media_type,
        "caption": caption,
        "image_b64": _b64.b64encode(raw).decode('utf-8'),
    }


@app.post("/delete_user_images/{channel_id}/{user_id}")
async def delete_user_images(channel_id: str, user_id: str):
    """
    Delete all saved images uploaded by `user_id` in `channel_id`. Used for
    privacy commands like ?forget_images. Removes both DB rows and on-disk
    files. Returns count of deletions.
    """
    user_id = _canon_uid(user_id)
    c.execute("SELECT id, image_path FROM saved_images "
              "WHERE channel_id = ? AND user_id = ?", (channel_id, user_id))
    rows = c.fetchall()
    deleted = 0
    for row_id, image_path in rows:
        try:
            full_path = os.path.join(IMAGE_STORE_DIR, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        except OSError as e:
            logger.warning(f"forget_images: could not remove {image_path}: {e}")
        c.execute("DELETE FROM saved_images WHERE id = ?", (row_id,))
        deleted += 1
    conn.commit()
    return {"status": "success", "deleted": deleted}


@app.get("/optout_summary/{channel_id}/{user_id}")
async def optout_summary(channel_id: str, user_id: str):
    """
    Pre-flight count of what `/optout` would delete. The bot calls this
    before showing the confirmation prompt so the user knows the scope of
    the deletion ('this would delete 312 messages, 4 saved images, and 1
    personality summary'). Pure read; no state change.
    """
    user_id = _canon_uid(user_id)
    c.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    msg_count = int(c.fetchone()[0] or 0)

    c.execute("SELECT COUNT(*) FROM saved_images WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    img_count = int(c.fetchone()[0] or 0)

    c.execute("SELECT COUNT(*) FROM personality_summaries "
              "WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    personality_count = int(c.fetchone()[0] or 0)

    c.execute("SELECT 1 FROM opted_out WHERE channel_id = ? AND user_id = ? LIMIT 1",
              (channel_id, user_id))
    already_opted_out = c.fetchone() is not None

    return {
        "messages": msg_count,
        "saved_images": img_count,
        "personality_summaries": personality_count,
        "already_opted_out": already_opted_out,
    }


@app.post("/optout/{channel_id}/{user_id}")
async def optout(channel_id: str, user_id: str):
    """
    Permanent per-channel opt-out. Performs four destructive operations,
    then registers the opt-out so future writes are dropped:

      1. Delete every saved_images row for this user in this channel,
         AND remove the corresponding files from disk.
      2. Delete every messages row for this user in this channel.
      3. NULL out replying_to in every messages row that was replying to
         this user (don't delete the OTHER user's content — just sever the
         link, which protects everyone's reasonable expectations).
      4. Delete the personality_summaries row.
      5. Insert into opted_out so future writes are silently dropped.

    Returns counts of what was actually deleted. Idempotent: running it
    twice is safe; the second call deletes 0 rows.
    """
    user_id = _canon_uid(user_id)
    now = datetime.now(timezone.utc).isoformat()

    # 1. saved_images: delete files first, then rows
    c.execute("SELECT id, image_path FROM saved_images "
              "WHERE channel_id = ? AND user_id = ?", (channel_id, user_id))
    img_rows = c.fetchall()
    img_files_removed = 0
    for row_id, image_path in img_rows:
        try:
            full_path = os.path.join(IMAGE_STORE_DIR, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                img_files_removed += 1
        except OSError as e:
            logger.warning(f"optout: could not remove {image_path}: {e}")
    c.execute("DELETE FROM saved_images WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    img_rows_deleted = c.rowcount

    # 2. messages: delete the user's own messages
    c.execute("DELETE FROM messages WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    msg_rows_deleted = c.rowcount

    # 3. messages: null out replying_to where it referenced this user.
    # We update rather than delete because the message authors are other
    # users and we have no right to wipe their content — just the link.
    c.execute("UPDATE messages SET replying_to = NULL "
              "WHERE channel_id = ? AND replying_to = ?",
              (channel_id, user_id))
    replyto_rows_nulled = c.rowcount

    # 4. personality_summaries
    c.execute("DELETE FROM personality_summaries "
              "WHERE channel_id = ? AND user_id = ?", (channel_id, user_id))
    personality_rows_deleted = c.rowcount

    # 5. opted_out registry: idempotent insert via INSERT OR REPLACE
    c.execute("INSERT OR REPLACE INTO opted_out (channel_id, user_id, opted_out_at) "
              "VALUES (?, ?, ?)", (channel_id, user_id, now))

    conn.commit()

    logger.info(f"optout: channel={channel_id} user={user_id} "
                f"messages={msg_rows_deleted} images={img_rows_deleted} "
                f"image_files={img_files_removed} personalities={personality_rows_deleted} "
                f"replyto_nulled={replyto_rows_nulled}")

    return {
        "status": "success",
        "messages_deleted": msg_rows_deleted,
        "images_deleted": img_rows_deleted,
        "image_files_removed": img_files_removed,
        "personalities_deleted": personality_rows_deleted,
        "replying_to_nulled": replyto_rows_nulled,
        "opted_out_at": now,
    }


@app.post("/optin/{channel_id}/{user_id}")
async def optin(channel_id: str, user_id: str):
    """
    Reverse a per-channel opt-out: remove the row from `opted_out` so future
    writes for this user in this channel succeed normally.
    Does NOT restore previously deleted data — that's gone forever.
    """
    user_id = _canon_uid(user_id)
    c.execute("DELETE FROM opted_out WHERE channel_id = ? AND user_id = ?",
              (channel_id, user_id))
    deleted = c.rowcount
    conn.commit()
    if deleted:
        logger.info(f"optin: channel={channel_id} user={user_id}")
    return {"status": "success", "was_opted_out": bool(deleted)}


@app.get("/check_opt_out/{channel_id}/{user_id}")
async def check_opt_out(channel_id: str, user_id: str):
    """Public read of the opt-out flag. Used by the bot's `/optout` command
    to detect 'already opted out' state and offer `/optin` instead."""
    return {"opted_out": _is_opted_out(channel_id, user_id)}


@app.get("/check_emoji_known/{emoji_id}")
async def check_emoji_known(emoji_id: str):
    """
    Cheap presence check: is this emoji_id already captioned? The bot calls
    this before kicking off a captioning task so we don't waste vision-model
    cycles on emojis we've already learned. Also bumps last_seen_at as a
    side effect — useful for prune logic.
    """
    c.execute("SELECT caption FROM server_emojis WHERE emoji_id = ? LIMIT 1",
              (emoji_id,))
    row = c.fetchone()
    if row is None:
        return {"known": False, "has_caption": False}
    has_caption = bool((row[0] or "").strip())
    # Bump last_seen_at so we know this emoji is still in active use
    now = datetime.now(timezone.utc).isoformat()
    c.execute("UPDATE server_emojis SET last_seen_at = ? WHERE emoji_id = ?",
              (now, emoji_id))
    conn.commit()
    return {"known": True, "has_caption": has_caption}


@app.post("/save_emoji")
async def save_emoji(payload: SaveEmojiUpload):
    """
    Upsert a server emoji's caption + embedding. Idempotent: calling with
    the same emoji_id twice updates the existing row (useful when the bot's
    captioner improves and you want to recaption).

    Note: animated is stored as INTEGER (0/1) rather than BOOLEAN because
    SQLite doesn't have a native boolean type and INTEGER is the canonical
    representation. We accept Python bool in the payload and convert.
    """
    caption = (payload.caption or "").strip()
    if not caption:
        raise HTTPException(status_code=400, detail="caption is required")
    embedding = await get_embedding(caption)
    if embedding is None:
        # The actual failure reason is logged by get_embedding(). We surface
        # a 503 with a hint pointing at the service logs so the bot's log
        # alone is enough to know where to look.
        raise HTTPException(
            status_code=503,
            detail=f"could not generate caption embedding for emoji "
                   f":{payload.name}: ({payload.emoji_id}) — see "
                   f"chat_history_service logs for the underlying cause "
                   f"(model not loaded, OWUI auth, timeout, etc.)"
        )

    now = datetime.now(timezone.utc).isoformat()
    animated_int = 1 if payload.animated else 0

    c.execute('''INSERT INTO server_emojis
                   (emoji_id, guild_id, name, animated, caption, embedding,
                    captioned_at, last_seen_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(emoji_id) DO UPDATE SET
                   guild_id = excluded.guild_id,
                   name = excluded.name,
                   animated = excluded.animated,
                   caption = excluded.caption,
                   embedding = excluded.embedding,
                   captioned_at = excluded.captioned_at,
                   last_seen_at = excluded.last_seen_at''',
              (str(payload.emoji_id), str(payload.guild_id), payload.name,
               animated_int, caption, json.dumps(embedding), now, now))
    conn.commit()
    return {"status": "success", "emoji_id": payload.emoji_id}


@app.get("/get_relevant_emojis/{guild_id}")
async def get_relevant_emojis(guild_id: str, query: str, limit: int = 8,
                              min_score: float = 0.0):
    """
    Semantic search over server emoji captions. Returns rows with name + id
    + animated + caption + score. The bot applies its own min-score gate
    (we accept one too via `min_score` param so the service can do the
    initial filtering and avoid sending obvious junk over the wire).

    Returns up to `limit` (capped at 30) rows sorted by descending score.
    """
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    c.execute('''SELECT emoji_id, guild_id, name, animated, caption, embedding
                 FROM server_emojis
                 WHERE guild_id = ? AND embedding IS NOT NULL''', (guild_id,))
    rows = c.fetchall()
    if not rows:
        return []

    query_dim = len(query_embedding)
    valid_rows = []
    valid_embeddings = []
    for r in rows:
        try:
            v = np.array(json.loads(r[5]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if v.shape[0] != query_dim:
            # Stale embedding from a previous model — skip silently. Re-running
            # the bot's startup recaption would refresh these.
            continue
        valid_rows.append(r)
        valid_embeddings.append(v)

    if not valid_rows:
        return []

    similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
    limit = max(1, min(limit, 30))
    # Sort descending by score, keep all (caller filters by min_score)
    order = np.argsort(similarities)[::-1]

    out = []
    for i in order:
        score = float(similarities[i])
        if score < min_score:
            break  # since we're sorted descending, everything after is also too low
        if len(out) >= limit:
            break
        r = valid_rows[i]
        out.append({
            "emoji_id": r[0],
            "guild_id": r[1],
            "name": r[2],
            "animated": bool(r[3]),
            "caption": r[4],
            "score": score,
        })
    return out


@app.post("/delete_guild_emojis_except")
async def delete_guild_emojis_except(guild_id: str, keep_ids: List[str] = Query(default_factory=list)):
    """
    Prune emojis that no longer exist in the guild. Called from the bot
    on startup (after fetching the live emoji list) and from
    on_guild_emojis_update. Pass guild_id and the list of currently-valid
    emoji_ids; everything else for that guild is deleted.

    If keep_ids is empty, ALL emojis for this guild are deleted (nuke).
    This is intentional — a guild with zero custom emojis is a valid state.
    """
    if keep_ids:
        # Build the IN clause safely
        placeholders = ",".join("?" for _ in keep_ids)
        sql = (f"DELETE FROM server_emojis WHERE guild_id = ? "
               f"AND emoji_id NOT IN ({placeholders})")
        c.execute(sql, [guild_id, *keep_ids])
    else:
        c.execute("DELETE FROM server_emojis WHERE guild_id = ?", (guild_id,))
    deleted = c.rowcount
    conn.commit()
    return {"status": "success", "deleted": deleted}


@app.get("/emoji_store_stats/{guild_id}")
async def emoji_store_stats(guild_id: str):
    """Diagnostic: how many emojis are captioned for this guild + a few samples."""
    c.execute("SELECT COUNT(*) FROM server_emojis WHERE guild_id = ?", (guild_id,))
    count = int(c.fetchone()[0] or 0)
    c.execute("SELECT emoji_id, name, animated, caption, captioned_at "
              "FROM server_emojis WHERE guild_id = ? "
              "ORDER BY captioned_at DESC LIMIT 5", (guild_id,))
    samples = [
        {"emoji_id": r[0], "name": r[1], "animated": bool(r[2]),
         "caption": r[3], "captioned_at": r[4]}
        for r in c.fetchall()
    ]
    return {"count": count, "samples": samples}


@app.get("/image_store_stats/{channel_id}")
async def image_store_stats(channel_id: str):
    """
    Diagnostic endpoint: how many saved images are stored for this channel,
    plus a sample of the most recent few captions. Used by the bot's
    /diagnose_images command to figure out whether the save flow is working.
    """
    c.execute("SELECT COUNT(*), COALESCE(SUM(file_size_bytes), 0) "
              "FROM saved_images WHERE channel_id = ?", (channel_id,))
    row = c.fetchone()
    count = int(row[0] or 0)
    total_bytes = int(row[1] or 0)

    # Most recent samples (cap at 5 for readability)
    c.execute("SELECT id, user_id, caption, file_size_bytes, saved_at "
              "FROM saved_images WHERE channel_id = ? "
              "ORDER BY saved_at DESC LIMIT 5", (channel_id,))
    samples = [
        {
            "id": r[0],
            "user_id": _wrap_uid(r[1]),
            "caption": r[2],
            "file_size_bytes": r[3],
            "saved_at": r[4],
        }
        for r in c.fetchall()
    ]

    # Embedding-dimension sanity: if we have rows, peek at the dim of the
    # most recent one. The bot's current embedding model should produce the
    # same dim; if there's drift, RAG retrieval will silently filter all
    # rows out and recall will appear to "do nothing".
    last_dim = None
    c.execute("SELECT embedding FROM saved_images WHERE channel_id = ? "
              "AND embedding IS NOT NULL ORDER BY saved_at DESC LIMIT 1",
              (channel_id,))
    er = c.fetchone()
    if er and er[0]:
        try:
            last_dim = len(json.loads(er[0]))
        except Exception:
            last_dim = None

    return {
        "count": count,
        "total_bytes": total_bytes,
        "samples": samples,
        "last_embedding_dim": last_dim,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/embeddings")
async def health_embeddings():
    """
    Live test of the embedding pipeline. Useful when the bot starts hitting
    503s and you want to know whether the embedding service is currently
    reachable at all. Hits OWUI with a tiny test prompt.

    Returns:
      {"status": "ok", "model": "...", "dim": N}     on success
      {"status": "error", "model": "...", "reason": "..."} on failure
    """
    test_text = "ping"
    embedding = await get_embedding(test_text)
    if embedding is not None:
        return {
            "status": "ok",
            "model": MODEL_ID_EMBEDDING,
            "dim": len(embedding),
        }
    return {
        "status": "error",
        "model": MODEL_ID_EMBEDDING,
        "reason": "see chat_history_service logs for specific error "
                  "(timeout, model not loaded, auth, etc.)",
    }


@app.on_event("startup")
async def _startup_self_test():
    """
    One-shot self-test on service startup: try a single embedding call and
    log whether it works. Catches 'qwen3-embedding model not pulled' and
    'OWUI token wrong' immediately at startup instead of when the first
    emoji save attempt fails minutes later.

    Doesn't block startup — if embeddings are down, the service still comes
    up; only embedding-dependent endpoints will return 503 until they work.
    """
    try:
        logger.info(f"startup: testing embedding pipeline (model={MODEL_ID_EMBEDDING})…")
        emb = await get_embedding("startup test")
        if emb is None:
            logger.error("startup: embedding test FAILED. The chat_history "
                         "service will return 503 on any endpoint that needs "
                         "embeddings. Check the messages above for the underlying "
                         "cause, then verify: (1) MODEL_ID_EMBEDDING is correct, "
                         "(2) OPENWEBUI_API_KEY is set + valid, "
                         "(3) qwen3-embedding (or whatever you set) is pulled "
                         "into your Ollama instance.")
        else:
            logger.info(f"startup: embedding test OK — {MODEL_ID_EMBEDDING} "
                        f"returns {len(emb)}-dim vectors")
    except Exception as e:
        logger.error(f"startup: embedding self-test crashed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)