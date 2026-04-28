import discord
from discord.ext import commands
from discord import app_commands
import aiohttp
import os
from dotenv import load_dotenv
import logging
import logging.handlers
import asyncio
import json
from datetime import datetime, timedelta as datetime_timedelta
import time
import random
import base64

# Setup and configuration
logger = logging.getLogger('MetaLLM')
logger.setLevel(logging.DEBUG)
file_handler = logging.handlers.RotatingFileHandler(
    filename='MetaLLM.log',
    maxBytes=5*1024*1024,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

load_dotenv()

BOT_NAME = "Cirno"

# Inference backend.
# We talk to Open WebUI's *Ollama passthrough* at /ollama/api/chat — this is a
# transparent forward to the underlying Ollama instance, so the request and
# response shapes stay identical to native Ollama. That lets us keep using the
# `think` parameter and the separate `thinking` / `content` response fields,
# which our chain-of-thought rescue logic depends on.
#
# LM_STUDIO_URL is the BASE URL only (no path). Default points at the hosted
# OWUI instance. Override with LM_STUDIO_URL_<BOT_NAME> in .env if needed.
LM_STUDIO_URL = os.getenv(f'LM_STUDIO_URL_{BOT_NAME}', 'https://oui.gpu.garden')
MODEL_ID_LLM = os.getenv(f'MODEL_ID_LLM_{BOT_NAME}', "gemma4:26b-a4b-it-q4_K_M")

# Optional separate model for personality summaries. Default = same as chat model,
# but you can point this at a plain instruct model in Ollama (e.g. `llama3.1:8b`,
# `qwen2.5:7b`, etc.) so summaries are written in a neutral analyst voice without
# any roleplay flavor leaking through.
MODEL_ID_SUMMARIZER = os.getenv(f'MODEL_ID_SUMMARIZER_{BOT_NAME}', MODEL_ID_LLM)

# Bearer token for Open WebUI. NEVER commit this — read from env only.
# Generate one in OWUI > Settings > Account > API Keys, then put in .env as:
#   OPENWEBUI_API_KEY_Cirno=sk-...
# (Or as the full JWT if you're using JWT auth.)
OPENWEBUI_API_KEY = os.getenv(f'OPENWEBUI_API_KEY_{BOT_NAME}')
if not OPENWEBUI_API_KEY:
    logger.warning(f"OPENWEBUI_API_KEY_{BOT_NAME} not set — LLM calls will likely fail with 401. "
                   f"Set it in your .env file.")

DISCORD_TOKEN = os.getenv(f'DISCORD_TOKEN_{BOT_NAME}')
OTHER_BOTS = ["HERMES", "NEMO", "NEMOTRON"]

# Slash command sync — comma-separated guild IDs to register the slash tree
# to. Per-guild registration is INSTANT (changes show up in the slash menu
# immediately). Global registration takes up to ~1 hour to propagate the
# first time. Recommended workflow: set to your test guild IDs during
# development, leave empty for production deploy.
_raw_guild_ids = os.getenv(f'GUILD_IDS_FOR_SLASH_SYNC_{BOT_NAME}', '')
GUILD_IDS_FOR_SLASH_SYNC = []
for _gid in _raw_guild_ids.split(','):
    _gid = _gid.strip()
    if _gid.isdigit():
        GUILD_IDS_FOR_SLASH_SYNC.append(int(_gid))

# Names/aliases that trigger a response when mentioned in a message.
# Case-insensitive. Matches whole words only (so "cirnolike" won't trigger on "cirno").
BOT_ALIASES = ["cirno", "cirbot", "cirnotest", "⑨", "baka", "gm"]

# RAG tuning knobs
RAG_MIN_SIMILARITY = 0.75       # drop retrieved excerpts below this score (if service returns scores)
RAG_MAX_RESULTS = 20             # absolute cap on excerpts injected into context
HISTORY_WINDOW = 30             # how many recent messages to pull as sliding window
RAG_INCLUDE_ASSISTANT = True    # if True, RAG can return Cirno's own past replies
                                # (turn off if you see Cirno parroting old answers)

# Unprompted-chime knobs. The bot ALWAYS replies when mentioned or replied-to.
# Separately, a background task periodically checks if she should chime in
# unprompted. Tune these so it feels rare and natural, not chatty.
ENABLE_UNPROMPTED_CHIME = True
CHIME_CHECK_INTERVAL_SEC = 60 * 30     # how often to consider chiming (every 30 min)
CHIME_PROBABILITY = 0.05               # 5% chance per check that she actually chimes
CHIME_COOLDOWN_SEC = 60 * 60 * 2       # never chime more than once per channel per 2h
CHIME_MIN_RECENT_MSGS = 3              # need at least this many messages since her last reply
CHIME_QUIET_AFTER_SEC = 60 * 5         # last activity must be older than this (don't interrupt active chats)
CHIME_RECENT_WINDOW_SEC = 60 * 60 * 6  # only chime if there's been activity in the last 6h

# Personality-summary knobs.
ENABLE_PERSONALITY_SUMMARIES = True
SUMMARY_MIN_NEW_MESSAGES = 5          # re-summarize after N new messages from a user
SUMMARY_MAX_USER_MESSAGES = 80         # how many recent user messages to feed into the summarizer
SUMMARY_MAX_TOKENS = 400               # cap on the generated summary's length

# Image memory knobs.
# Cirno saves an image to long-term memory only when she's pinged on the
# message that contained it (so the dataset is biased toward "interesting"
# images rather than everyone's screenshots). Captions are generated by the
# vision model with a neutral prompt and embedded for retrieval.
ENABLE_IMAGE_MEMORY = True
IMAGE_RAG_LIMIT = 3                    # how many captioned images to fetch as RAG hits
IMAGE_RAG_MIN_SCORE = 0.55             # caption-similarity threshold (lower than text RAG;
                                       #   captions are short and lossy by nature)
IMAGE_CAPTION_MAX_TOKENS = 120         # caption length budget — one or two sentences
IMAGE_MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024  # don't try to caption huge files (>8MB)
# When the bot's reply contains [recall_image: N] for an image we offered,
# we strip the marker from text and attach the bytes. This regex is permissive
# about whitespace inside the brackets.
import re as _re_module
RECALL_IMAGE_RE = _re_module.compile(r'\[\s*recall[_\s]?image\s*:\s*(\d+)\s*\]', _re_module.IGNORECASE)

# Poll-creation marker. Looks like:
#   [create_poll: question="..." options="A|B|C" duration_h=24 multi=false]
# Permissive: order of kwargs flexible, missing fields use defaults, both " and ' quotes work.
# We grab everything between [create_poll: ... ] and parse out the kwargs in
# Python — a single regex tight enough to validate everything would be brittle.
CREATE_POLL_RE = _re_module.compile(
    r'\[\s*create[_\s]?poll\s*:\s*(.+?)\s*\]',
    _re_module.IGNORECASE | _re_module.DOTALL,
)
# Sub-pattern for kwargs inside the brace. Matches:  key="value"  key='value'  key=bare_word  key=number
_POLL_KWARG_RE = _re_module.compile(
    r'''(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s\]]+))''',
    _re_module.IGNORECASE,
)


def _parse_create_poll_markers(text):
    """
    Find every `[create_poll: ...]` marker in `text`. Returns a list of
    parsed poll specs (dicts) and the cleaned text (markers removed).

    Each spec dict has keys:
        question (str, required, validated to <= POLL_MAX_QUESTION_CHARS)
        options (list[str], required, 2-10 items, each <= POLL_MAX_OPTION_CHARS)
        duration_h (int, optional, default POLL_DEFAULT_DURATION_HOURS, capped)
        multi (bool, optional, default False)

    Specs that fail validation are dropped (and logged) rather than raising —
    we'd rather show the bot's text without a poll than silently fail the whole
    reply because the model emitted bad syntax.
    """
    if not text:
        return [], text

    specs = []

    def _replace(m):
        body = m.group(1)
        kwargs = {}
        for km in _POLL_KWARG_RE.finditer(body):
            key = km.group(1).lower()
            # Quoted forms or bare value — pick whichever group matched.
            value = km.group(2) or km.group(3) or km.group(4) or ""
            kwargs[key] = value

        question = (kwargs.get("question") or "").strip()
        options_raw = (kwargs.get("options") or "").strip()
        duration_raw = (kwargs.get("duration_h") or kwargs.get("duration") or "").strip()
        multi_raw = (kwargs.get("multi") or kwargs.get("multiselect") or "").strip().lower()

        # Validate
        if not question:
            logger.debug(f"poll: dropped marker (no question): {body!r}")
            return ""
        if len(question) > POLL_MAX_QUESTION_CHARS:
            question = question[:POLL_MAX_QUESTION_CHARS - 1].rstrip() + "…"

        # Options come pipe-delimited. Trim each, drop empty.
        options = [o.strip() for o in options_raw.split("|") if o.strip()]
        if len(options) < 2:
            logger.debug(f"poll: dropped marker (need 2+ options): {body!r}")
            return ""
        if len(options) > POLL_MAX_OPTIONS:
            options = options[:POLL_MAX_OPTIONS]
        # Per-option char cap
        options = [
            (o if len(o) <= POLL_MAX_OPTION_CHARS else o[:POLL_MAX_OPTION_CHARS - 1].rstrip() + "…")
            for o in options
        ]

        try:
            duration_h = int(float(duration_raw)) if duration_raw else POLL_DEFAULT_DURATION_HOURS
        except (TypeError, ValueError):
            duration_h = POLL_DEFAULT_DURATION_HOURS
        duration_h = max(1, min(duration_h, POLL_MAX_DURATION_HOURS))

        multi = multi_raw in ("true", "yes", "1", "y", "t")

        specs.append({
            "question": question,
            "options": options,
            "duration_h": duration_h,
            "multi": multi,
        })
        return ""  # always strip the marker

    cleaned = CREATE_POLL_RE.sub(_replace, text)
    # Tidy whitespace runs
    cleaned = _re_module.sub(r'[ \t]+', ' ', cleaned)
    cleaned = _re_module.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return specs, cleaned

# Vision context knobs (for stickers + loosened emoji vision).
VISION_INCLUDE_STICKERS = True
VISION_MAX_EMOJIS_INLINE = 3           # how many custom emoji images to attach when present

# Poll feature knobs.
# Cirno can create native Discord polls. To keep the feature understated rather
# than annoying, we only PRIME the model with the tool description once every
# POLL_PRIME_EVERY_N user messages in a channel. The note tells her she can
# emit `[create_poll: question="..." options="A|B|C" ...]` markers; same
# strip-after-stream pattern as [recall_image: N].
ENABLE_POLLS = True
POLL_PRIME_EVERY_N = 20                # show the poll-tool note every Nth user message
POLL_DEFAULT_DURATION_HOURS = 24
POLL_MAX_OPTIONS = 10                  # Discord's hard limit
POLL_MAX_QUESTION_CHARS = 300          # Discord's hard limit
POLL_MAX_OPTION_CHARS = 55             # Discord's hard limit
POLL_MAX_DURATION_HOURS = 24 * 32      # Discord's max poll duration
# Per-channel message counter for poll priming. In-memory only — survives
# until process restart, which is fine since a fresh bot would just prime
# on the very next message anyway.
_poll_prime_counter = {}  # {channel_id_str: int}
import re as _re_module2  # _re_module already used elsewhere; harmless duplicate alias


def _load_persona():
    """
    Load the persona prompt from an external file.

    Lookup order:
      1. PERSONA_FILE_<BOT_NAME> env var (absolute or relative path)
      2. ./persona.txt next to this script
      3. ./persona_<bot_name>.txt next to this script (lowercase bot name)

    Substitutes {BOT_NAME} with the actual bot name. Falls back to a minimal
    default and logs a warning if no file is found.
    """
    candidates = []
    env_path = os.getenv(f'PERSONA_FILE_{BOT_NAME}')
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "persona.txt"))
    candidates.append(os.path.join(here, f"persona_{BOT_NAME.lower()}.txt"))

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            if raw.strip():
                logger.info(f"Loaded persona from {path}")
                return raw.format(BOT_NAME=BOT_NAME)
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(f"Could not read persona file {path}: {e}")
            continue
        except (KeyError, IndexError) as e:
            # format() failed because of a stray brace in the persona text.
            # Return raw content so the user at least gets *something*.
            logger.warning(f"Persona file {path} contains unescaped braces: {e}. "
                           f"Use {{{{ and }}}} for literal braces, or {{BOT_NAME}} for substitution.")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError:
                continue

    logger.warning("No persona file found — falling back to minimal default. "
                   "Create persona.txt next to the script.")
    return f"You are {BOT_NAME}. Be helpful and friendly."


CIRNO_PERSONA = _load_persona()


def _rewrite_self_mentions(text):
    """
    Replace the bot's own Discord mention with a clear self-reference
    so the LLM understands when it's being pinged/referenced.
    """
    if not text or not bot.user:
        return text
    bot_id = str(bot.user.id)
    for pattern in (f"<@{bot_id}>", f"<@!{bot_id}>"):
        text = text.replace(pattern, "@CIRNO(you)")
    return text


# Cache: user_id (str) -> display_name (str)
# discord.py already caches Member objects, but we keep our own light cache
# so we don't re-do the lookup+format work on every message.
_username_cache = {}
_MENTION_RE = __import__('re').compile(r'<@!?(\d+)>')

# Cache rewritten content keyed by message ID — avoids re-resolving mentions
# for every retrieved message every turn. RAG retrieval is the hottest path here.
_rewritten_content_cache = {}  # {message_id: rewritten_text}


async def _resolve_user_name(user_id_str, guild=None):
    """
    Return a display name for a Discord user ID. Checks cache first, then guild
    members, then falls back to a global API fetch. Returns None if unresolvable.
    """
    if user_id_str in _username_cache:
        return _username_cache[user_id_str]

    # Bot's own ID — special-case so we never replace self-mentions via this path
    if bot.user and user_id_str == str(bot.user.id):
        _username_cache[user_id_str] = BOT_NAME
        return BOT_NAME

    try:
        uid = int(user_id_str)
    except ValueError:
        return None

    # Try guild member first (fast, already cached by discord.py)
    if guild is not None:
        member = guild.get_member(uid)
        if member is not None:
            name = member.display_name or member.name
            _username_cache[user_id_str] = name
            return name

    # Fallback: global user fetch (hits Discord API, rate-limited)
    try:
        user = bot.get_user(uid) or await bot.fetch_user(uid)
        if user:
            name = getattr(user, 'display_name', None) or user.name
            _username_cache[user_id_str] = name
            return name
    except discord.NotFound:
        pass
    except Exception as e:
        logger.debug(f"Could not resolve user {user_id_str}: {e}")

    return None


async def _rewrite_mentions(text, guild=None):
    """
    Replace all <@id> and <@!id> mentions in `text` with @DisplayName.
    Also rewrite custom emoji <:name:id> to :name: so the LLM can read it.
    The bot's own mention is replaced with @CIRNOBOT(you) via _rewrite_self_mentions.
    """
    if not text:
        return text

    # Self-mention first (keeps the (you) marker intact)
    text = _rewrite_self_mentions(text)

    # Custom emoji → :name: substitution (cheap, always applied)
    text = _rewrite_custom_emojis(text)

    # Find all remaining user mentions
    ids = set(_MENTION_RE.findall(text))
    if not ids:
        return text

    # Resolve each unique ID
    for uid in ids:
        name = await _resolve_user_name(uid, guild=guild)
        if name:
            text = text.replace(f"<@{uid}>", f"@{name}")
            text = text.replace(f"<@!{uid}>", f"@{name}")

    return text


async def _rewrite_mentions_cached(text, message_id, guild=None):
    """
    Same as _rewrite_mentions but cached by message_id to avoid re-resolving
    the same retrieved message's mentions on every turn.
    """
    if not message_id:
        return await _rewrite_mentions(text, guild=guild)
    cached = _rewritten_content_cache.get(message_id)
    if cached is not None:
        return cached
    rewritten = await _rewrite_mentions(text, guild=guild)
    # Bound the cache so it doesn't grow forever
    if len(_rewritten_content_cache) > 5000:
        # drop ~half by clearing oldest insertions (dict is insertion-ordered)
        for k in list(_rewritten_content_cache.keys())[:2500]:
            del _rewritten_content_cache[k]
    _rewritten_content_cache[message_id] = rewritten
    return rewritten


# Custom emoji patterns: <:name:id> for static, <a:name:id> for animated
_EMOJI_RE = __import__('re').compile(r'<(a?):([A-Za-z0-9_~]+):(\d+)>')

# Set to True to also fetch the emoji image and feed it to the vision model
# when the message is *mostly* just emojis. False = text-only substitution.
VISION_EMOJIS = True


def _rewrite_custom_emojis(text):
    """
    Replace Discord custom emoji syntax <:name:id> with :name: so the LLM
    at least knows it's an emoji with a name rather than seeing raw markup.
    """
    if not text:
        return text
    return _EMOJI_RE.sub(r':\2:', text)


def _extract_emoji_urls(text, max_emojis=3):
    """
    Pull out custom emoji image URLs from message text.
    Returns list of (name, url, is_animated) tuples, capped at max_emojis
    so we don't blow token budget on a spam message full of emojis.
    """
    if not text:
        return []
    urls = []
    for match in _EMOJI_RE.finditer(text):
        animated, name, emoji_id = match.groups()
        ext = "gif" if animated == "a" else "png"
        urls.append((name, f"https://cdn.discordapp.com/emojis/{emoji_id}.{ext}", bool(animated)))
        if len(urls) >= max_emojis:
            break
    return urls


def _message_is_mostly_emoji(text):
    """
    Heuristic: is this message essentially just emojis (no real content)?
    If so, worth sending the emoji image to the vision model. Otherwise the
    :name: text substitution is enough.
    """
    if not text:
        return False
    stripped = _EMOJI_RE.sub('', text).strip()
    # After removing emoji syntax, is there <= 3 chars of other content?
    return len(stripped) <= 3


async def _fetch_emoji_as_base64(url):
    """Download an emoji from Discord's CDN and return it as base64."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    raw = await response.read()
                    return base64.b64encode(raw).decode('utf-8')
                else:
                    logger.debug(f"Emoji fetch {url} returned {response.status}")
    except Exception as e:
        logger.debug(f"Emoji fetch failed for {url}: {e}")
    return None


async def _fetch_url_as_base64(url):
    """Generic URL → base64 fetch. Returns None on failure. Used for stickers."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    raw = await response.read()
                    return base64.b64encode(raw).decode('utf-8')
                logger.debug(f"Asset fetch {url} returned {response.status}")
    except Exception as e:
        logger.debug(f"Asset fetch failed for {url}: {e}")
    return None


async def _gather_sticker_visuals(message):
    """
    For each sticker on the message, return (text_annotation, base64_or_None).
    Discord has three sticker formats:
      - PNG / APNG: rasterizable (vision-readable; for APNG we just get
        the first frame from the standard sticker URL)
      - LOTTIE: vector animation (NOT vision-readable; we only return text)
    The caller decides whether to attach the bytes to the vision payload.
    """
    if not VISION_INCLUDE_STICKERS or not getattr(message, "stickers", None):
        return []

    out = []
    for sticker in message.stickers:
        name = getattr(sticker, "name", "") or "sticker"
        # discord.py exposes sticker.format which is a discord.StickerFormatType
        # enum; we use its .name string for text and its identity for routing.
        fmt = getattr(sticker, "format", None)
        fmt_name = getattr(fmt, "name", str(fmt)).lower() if fmt is not None else "unknown"

        # Some sticker objects also have a description field (the Discord
        # creator may have written one). Adds context the model can use.
        description = (getattr(sticker, "description", "") or "").strip()
        if description:
            text = f"[sticker: {name} — {description}]"
        else:
            text = f"[sticker: {name}]"

        b64 = None
        # Lottie stickers can't be read by vision models. Skip the bytes.
        if fmt_name != "lottie":
            url = getattr(sticker, "url", None)
            if url:
                b64 = await _fetch_url_as_base64(url)

        out.append((text, b64, name))
    return out


_ALIAS_RE = None  # built lazily from BOT_ALIASES

def _message_mentions_bot_by_name(text):
    """
    Returns True if the message text contains any of BOT_ALIASES as a whole word
    (case-insensitive). "cirno" matches "hey cirno!" and "cirno?" but not "cirnolike".
    """
    global _ALIAS_RE
    if not text or not BOT_ALIASES:
        return False
    if _ALIAS_RE is None:
        import re
        # Build one regex with word boundaries. Escape each alias in case of special chars.
        pattern = r'(?:^|\W)(' + '|'.join(re.escape(a) for a in BOT_ALIASES) + r')(?:\W|$)'
        _ALIAS_RE = re.compile(pattern, re.IGNORECASE)
    return bool(_ALIAS_RE.search(text))


# Per-channel map of display-name -> user_id for participants we've seen recently.
# Used to turn the LLM's plain-text "@name" back into a real Discord mention.
# Only names that have actually appeared in this channel's conversation are
# eligible — prevents accidental cross-channel pings.
_channel_participants = {}  # {channel_id: {lowercase_name: user_id}}


def _register_participant(channel_id, user):
    """Record a user as a participant in this channel so the bot can later ping them."""
    if not user or not channel_id:
        return
    name = (user.display_name or user.name or "").strip()
    if not name:
        return
    _channel_participants.setdefault(str(channel_id), {})[name.lower()] = str(user.id)


def _strip_leaked_reply_markers(text):
    """
    Remove reply-context artifacts Cirno might have copied from the prompt
    into her own output. The pipeline shows her things like blockquoted target
    messages ('> @x: previous text') and the older parenthetical markers — the
    model occasionally parrots them.

    Strips:
      - leading blockquote lines that look like the prompt's reply quotes
      - "(↳ to @x)" / "[reply to @x]" / "[in reply to @x]" markers anywhere
    """
    if not text:
        return text
    import re

    # 1. Strip leading blockquote lines (one or more "> ..." at the start).
    # These are how we render the reply target in the prompt; if the model
    # repeats them it's just echoing context.
    lines = text.splitlines()
    while lines and lines[0].lstrip().startswith('>'):
        lines.pop(0)
    text = '\n'.join(lines).lstrip()

    # 2. Inline marker patterns (legacy, in case any leak through)
    patterns = [
        r'\(\s*↳\s*to\s+@?\S+(?::\s*"[^"]*")?\s*\)',
        r'\[\s*reply(?:ing)?\s+to\s+@?\S+(?::\s*"[^"]*")?\s*\]',
        r'\[\s*in\s+reply\s+to\s+@?\S+(?::\s*"[^"]*")?\s*\]',
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    # Collapse any whitespace runs left by the removal
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _rewrite_output_mentions(text, channel_id):
    """
    Convert plain-text '@name' in the bot's reply back into Discord mention syntax
    '<@user_id>' so it becomes an actual ping. Only pings participants we've seen
    in this channel; unknown names are left as plain text.

    Also strips @everyone / @here to prevent accidental mass-pings.
    """
    if not text:
        return text

    import re

    # 1. Neutralize mass-ping attempts first (even in-character, these are dangerous)
    # Insert zero-width space between @ and everyone/here so Discord doesn't parse it.
    # Note: replacement is NOT a raw string — we want \u200b interpreted as the ZWSP char,
    # and \\1 is the escaped backreference.
    text = re.sub(r'@(everyone|here)\b', '@\u200b\\1', text, flags=re.IGNORECASE)

    participants = _channel_participants.get(str(channel_id), {})
    if not participants:
        return text

    # 2. Replace @name with <@id> for known participants.
    names_by_length = sorted(participants.keys(), key=len, reverse=True)
    for lname in names_by_length:
        user_id = participants[lname]
        pattern = r'@' + re.escape(lname) + r'(?=\W|$)'
        text = re.sub(pattern, f'<@{user_id}>', text, flags=re.IGNORECASE)

    return text

CHAT_HISTORY_SERVICE_URL = "http://localhost:8000"

# Headers for OWUI calls. Keep auth in here so all callsites pick it up.
HEADERS = {"Content-Type": "application/json"}
if OPENWEBUI_API_KEY:
    HEADERS["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"

# Flip this to enable chain-of-thought. When on, num_predict gets bumped
# automatically since thinking eats a lot of tokens before the answer appears.
ENABLE_THINKING = True
THINK_NUM_PREDICT = 3000   # budget when thinking is on (trace + answer)
FAST_NUM_PREDICT = 1000    # budget when thinking is off

# --- Streaming knobs ---
# We stream from Ollama (so Cloudflare's ~100s idle-timeout never triggers) and
# also stream into Discord by editing a placeholder message.
#
# Discord's edit rate limit is ~5 edits per 5 seconds per channel, so we
# throttle: a partial edit fires at most once per STREAM_EDIT_INTERVAL_SEC, or
# whenever the accumulated content has grown by STREAM_EDIT_MIN_CHARS, whichever
# comes second. The final edit at end-of-stream is unconditional.
STREAM_EDIT_INTERVAL_SEC = 1.5
STREAM_EDIT_MIN_CHARS = 60        # don't edit if barely any new text
STREAM_PLACEHOLDER = "⑨ thinking..."  # shown until the first content token arrives
# Alternate placeholder used when /api/ps says the model isn't loaded —
# warns the user that the cold-load delay is coming. Switches to STREAM_PLACEHOLDER
# once any chunk arrives.
STREAM_PLACEHOLDER_WAKING = "⑨ waking up..."
# Per-request HTTP timeout for the streaming call. Generous because thinking
# can take 60s+ before the first content token. The stream itself reads with a
# longer total budget (sock_read) to account for slow tokens.
STREAM_TIMEOUT_TOTAL_SEC = 600
STREAM_TIMEOUT_SOCK_READ_SEC = 120  # max gap between two streamed chunks


async def _check_model_loaded(model_name):
    """
    Hit /ollama/api/ps and return True if `model_name` is in the loaded list,
    False if not, None on error (caller should treat unknown as "loaded" and
    not show the waking-up message — better than spamming false alarms).

    The call is cheap (~50ms typically) and runs once per chat call, before we
    start streaming. The result tells _stream_to_discord which placeholder
    text to start with.
    """
    if not model_name:
        return None
    url = f"{LM_STUDIO_URL.rstrip('/')}/ollama/api/ps"
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=HEADERS) as response:
                if response.status != 200:
                    logger.debug(f"/api/ps returned {response.status}; assuming loaded")
                    return None
                data = await response.json()
                loaded = data.get("models") or []
                # Match either by exact name or by `model` field. Ollama tags
                # like "gemma4:26b-..." can appear in either field depending
                # on version. We accept any case-insensitive match.
                target = model_name.lower()
                for m in loaded:
                    n = (m.get("name") or "").lower()
                    mm = (m.get("model") or "").lower()
                    if target == n or target == mm:
                        return True
                return False
    except asyncio.TimeoutError:
        logger.debug("/api/ps timed out")
        return None
    except Exception as e:
        logger.debug(f"/api/ps error: {e}")
        return None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # needed for resolving <@id> mentions to display names reliably
bot = commands.Bot(command_prefix='?', intents=intents)

message_queue = {}
processing_channels = set()
last_response_time = {}

# Per-channel ctxbreak cutoff: any message with a timestamp <= this is hidden
# from both sliding-window history and RAG retrieval. Survives only in memory —
# bot restart wipes it. If you want it persistent, store it in the chat history
# service or a small json file.
_ctx_cutoff = {}  # {channel_id (str): datetime (timezone-aware UTC)}

MAX_MESSAGE_LENGTH = 1900


async def get_base64_images(message):
    base64_images = []
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_bytes = await attachment.read()
                b64_string = base64.b64encode(image_bytes).decode('utf-8')
                base64_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}
                })
    return base64_images


async def record_chat_history(channel_id, message):
    """
    POST a message to the history service. Returns the service's response JSON
    on success (which may include the assigned message id), else None.

    `message` may include a `discord_message_id` field — if your service stores
    and returns it, we use it for reliable dedupe in get_structured_input.
    """
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "channel_id": channel_id,
                "user_id": message['user_id'],
                "content": message['content'],
                "replying_to": message.get('replying_to'),
                "role": message['role'],
            }
            # Only include if the caller provided one (keeps payload backward-compat
            # with services that ignore unknown fields).
            if message.get('discord_message_id'):
                payload['discord_message_id'] = message['discord_message_id']
            if message.get('replying_to_message_id'):
                payload['replying_to_message_id'] = message['replying_to_message_id']
            async with session.post(f"{CHAT_HISTORY_SERVICE_URL}/add_message", json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to record chat history: {await response.text()}")
                    return None
                return await response.json()
    except Exception as e:
        logger.error(f"Failed to record chat history: {str(e)}")
        return None


async def get_conversation_history(channel_id, limit=HISTORY_WINDOW):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{CHAT_HISTORY_SERVICE_URL}/get_conversation_history/{channel_id}",
                                   params={"limit": limit}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get conversation history: {await response.text()}")
                    return []
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        return []


async def get_relevant_context(channel_id, query, limit=RAG_MAX_RESULTS,
                               after=None, before=None):
    """
    Fetch semantically-relevant prior messages.

    Always excludes other-bot chatter (role="user" — see record_user_message).
    Excludes Cirno's own past replies (role="assistant") only when
    RAG_INCLUDE_ASSISTANT is False.

    Optional `after`/`before` are tz-aware datetimes, sent to the service as
    ISO strings. The service is expected to apply them as a SQL WHERE clause
    over the timestamp column. If your service doesn't yet support these, the
    params will be ignored (most frameworks happily ignore unknown query args)
    and you'll fall back to pure semantic search.
    """
    try:
        async with aiohttp.ClientSession() as session:
            params = [
                ("query", query),
                ("limit", str(limit)),
                ("exclude_role", "user"),   # other bots are stored as role="user"
            ]
            if not RAG_INCLUDE_ASSISTANT:
                params.append(("exclude_role", "assistant"))
            if after is not None:
                params.append(("after", after.isoformat()))
            if before is not None:
                params.append(("before", before.isoformat()))

            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/get_relevant_context/{channel_id}",
                params=params,
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get relevant context: {await response.text()}")
                    return []
    except Exception as e:
        logger.error(f"Failed to get relevant context: {str(e)}")
        return []


# --- Image memory HTTP clients ----------------------------------------------
# These talk to the chat history service's /save_image, /get_relevant_images,
# /get_image_bytes, and /delete_user_images endpoints. Kept separate from the
# message RAG calls because they have different shapes (POST a binary blob,
# GET multipart-ish data with bytes fetched on demand).

async def save_image_to_memory(channel_id, user_id, discord_message_id,
                               media_type, extension, image_b64, caption):
    """POST a captioned image to the history service. Returns the new id on
    success, None on failure. Caller is responsible for generating the caption."""
    payload = {
        "channel_id": str(channel_id),
        "user_id": str(user_id),
        "discord_message_id": str(discord_message_id),
        "media_type": media_type,
        "extension": extension,
        "image_b64": image_b64,
        "caption": caption,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHAT_HISTORY_SERVICE_URL}/save_image", json=payload
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return data.get("id")
                logger.error(f"save_image failed: {r.status} {await r.text()}")
    except Exception as e:
        logger.error(f"save_image_to_memory failed: {e}")
    return None


async def get_relevant_images(channel_id, query, limit=IMAGE_RAG_LIMIT):
    """Semantic search over saved image captions for this channel. Returns a
    list of dicts (caption, id, score, image_path, ...). Bytes are NOT fetched
    here — call fetch_image_bytes(id) when you actually want to attach."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/get_relevant_images/{channel_id}",
                params={"query": query, "limit": limit},
            ) as r:
                if r.status == 200:
                    return await r.json()
                logger.error(f"get_relevant_images: {r.status}")
    except Exception as e:
        logger.error(f"get_relevant_images failed: {e}")
    return []


async def fetch_image_bytes(image_id):
    """Fetch a saved image's raw bytes (returns dict with media_type +
    image_b64 + caption) or None on failure / not found."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/get_image_bytes/{image_id}"
            ) as r:
                if r.status == 200:
                    return await r.json()
                if r.status == 404:
                    return None
                logger.error(f"get_image_bytes: {r.status}")
    except Exception as e:
        logger.error(f"fetch_image_bytes failed: {e}")
    return None


async def delete_user_images(channel_id, user_id):
    """Delete all saved images for this user in this channel. Returns count."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHAT_HISTORY_SERVICE_URL}/delete_user_images/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return int(data.get("deleted", 0))
                logger.error(f"delete_user_images: {r.status}")
    except Exception as e:
        logger.error(f"delete_user_images failed: {e}")
    return 0


# --- Opt-out HTTP clients ----------------------------------------------------
# These wrap /optout, /optin, /optout_summary, /check_opt_out on the history
# service. The local in-memory cache below makes record_user_message's gate
# cheap (no HTTP round-trip per message).

# Per-channel set of opted-out user_ids. Populated lazily as we observe
# opt-outs / opt-ins; on cold start, we fall back to the HTTP check on first
# write attempt for an unknown user. Keys are str(channel_id), values are sets
# of bare-id strings.
_optout_cache = {}  # {channel_id_str: set[user_id_str]}
# Marker so we know whether we've ever queried this channel; lets us avoid
# repeatedly re-fetching for users we've never seen.
_optout_cache_loaded = set()  # set of channel_id_str


async def fetch_optout_summary(channel_id, user_id):
    """GET /optout_summary — count what /optout would delete. Returns dict
    or None on error."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/optout_summary/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    return await r.json()
                logger.error(f"optout_summary: {r.status}")
    except Exception as e:
        logger.error(f"fetch_optout_summary failed: {e}")
    return None


async def perform_optout(channel_id, user_id):
    """POST /optout — destructive. Returns dict with deletion counts or None
    on error. After success, updates the local cache so subsequent
    record_user_message calls drop quickly without an HTTP round-trip."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHAT_HISTORY_SERVICE_URL}/optout/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    _optout_cache.setdefault(str(channel_id), set()).add(str(user_id))
                    return data
                logger.error(f"perform_optout: {r.status}")
    except Exception as e:
        logger.error(f"perform_optout failed: {e}")
    return None


async def perform_optin(channel_id, user_id):
    """POST /optin — undoes a previous /optout. Updates the local cache."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHAT_HISTORY_SERVICE_URL}/optin/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    s = _optout_cache.get(str(channel_id))
                    if s is not None:
                        s.discard(str(user_id))
                    return data
                logger.error(f"perform_optin: {r.status}")
    except Exception as e:
        logger.error(f"perform_optin failed: {e}")
    return None


async def check_opt_out(channel_id, user_id):
    """
    Returns True if (channel_id, user_id) is opted out. Cache-first to keep
    the per-message overhead near-zero. On first observation of a channel,
    we treat the cache as authoritative — if we've never seen the user
    opt-out, we assume they haven't. The opt-out write path is the only
    way an entry gets into the cache, so this is consistent within a single
    bot session. On bot restart, the cache is empty and we trust the
    history service's check on subsequent writes (which is implicit because
    /add_message itself silently drops opted-out users).
    """
    cache = _optout_cache.get(str(channel_id))
    if cache is None:
        return False
    return str(user_id) in cache


def is_user_opted_out_local(channel_id, user_id):
    """Sync version of check_opt_out for use in non-async paths."""
    cache = _optout_cache.get(str(channel_id))
    return cache is not None and str(user_id) in cache


async def get_llm_response(prompt, retry=True, model=None, enable_thinking=None,
                           num_predict=None, temperature=0.7, on_partial=None):
    """
    Talks to Open WebUI's Ollama passthrough at /ollama/api/chat.

    The passthrough forwards requests to the underlying Ollama instance with no
    transformation, so the request and response shapes are identical to native
    Ollama. This is what lets us keep:
      - the `think` parameter (chain-of-thought toggle)
      - the separate `thinking` / `content` response fields
      - our _extract_from_thinking() rescue path

    Streaming: this function ALWAYS uses stream=True under the hood. That's
    what keeps us under Cloudflare's ~100s idle-timeout (CF only kills idle
    connections; bytes flowing back keep it alive). Whether the caller cares
    about partial output is controlled by `on_partial`.

    Args:
        model:           override which model to call. Defaults to MODEL_ID_LLM.
                         Used for summaries.
        enable_thinking: override whether to enable chain-of-thought.
                         Defaults to the global ENABLE_THINKING flag.
        num_predict:     override the token budget. Defaults pick THINK_NUM_PREDICT
                         or FAST_NUM_PREDICT based on whether thinking is on.
        temperature:     sampling temperature.
        on_partial:      optional async callback `await on_partial(state)` invoked
                         as content tokens arrive. `state` is a dict:
                            {"phase": "thinking" | "content",
                             "content": str,    # accumulated content so far
                             "thinking": str,   # accumulated thinking so far
                             "done": bool}      # True only on the final call
                         The callback is responsible for its own throttling
                         (we call it on every chunk that brings new content).

    Returns the final cleaned `content` string, or None on hard failure.
    """
    use_model = model or MODEL_ID_LLM
    use_think = ENABLE_THINKING if enable_thinking is None else enable_thinking
    if num_predict is None:
        num_predict = THINK_NUM_PREDICT if use_think else FAST_NUM_PREDICT

    try:
        # --- Gemma compatibility: fold system into first user turn ---
        gemma_safe = _fold_system_into_user(prompt)

        # --- Ollama native format: content must be string, images go in separate field ---
        ollama_messages = _to_ollama_format(gemma_safe)

        print(f"\n--- DEBUG: PAYLOAD TO LLM ({use_model}) ---repr({json.dumps(_redact_images(ollama_messages), indent=2, ensure_ascii=False)})\n-----------------------------\n")

        # Wide HTTP timeouts: the model can take a while to produce the FIRST
        # token (especially with thinking on and a cold weight cache), but as
        # long as bytes keep arriving we're fine. sock_read = max gap BETWEEN
        # chunks; total = max wall-clock for the whole call.
        timeout = aiohttp.ClientTimeout(
            total=STREAM_TIMEOUT_TOTAL_SEC,
            sock_read=STREAM_TIMEOUT_SOCK_READ_SEC,
        )

        accumulated_content = ""
        accumulated_thinking = ""
        done_reason = "unknown"
        # Has the user-visible "content" phase started yet? Used for the phase
        # tag we pass back to on_partial.
        content_started = False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # OWUI Ollama passthrough — request shape identical to Ollama's /api/chat.
            url = f"{LM_STUDIO_URL.rstrip('/')}/ollama/api/chat"
            payload = {
                "model": use_model,
                "messages": ollama_messages,
                "stream": True,
                "think": use_think,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            }
            async with session.post(url, headers=HEADERS, json=payload) as response:
                if response.status != 200:
                    err_text = await response.text()
                    logger.error(f"OWUI/Ollama returned status {response.status}: {err_text[:500]}")
                    print(f"DEBUG: OWUI/Ollama error body (truncated): {err_text[:500]}")
                    # 401/403 are auth issues — surface them clearly so they're easy to spot in logs.
                    if response.status in (401, 403):
                        logger.error("Auth failure — check OPENWEBUI_API_KEY_<botname> env var. "
                                     "Token may be missing, expired, or revoked.")
                    elif response.status in (502, 503, 504, 524):
                        # Cloudflare/origin issues. The streaming approach SHOULD
                        # avoid 524 but if it still happens we want to know.
                        logger.error(f"Upstream timeout/unavailable ({response.status}) — "
                                     f"if 524, check that streaming is actually flowing through.")
                    return None

                # Stream parsing: Ollama emits one JSON object per line (NDJSON).
                # Each line has a `message.content` and/or `message.thinking`
                # delta and a `done` boolean. The final line has done=true and
                # may also carry done_reason and final stats.
                async for raw_line in response.content:
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Occasionally the stream may include partial bytes if a
                        # chunk straddles a newline boundary; aiohttp's content
                        # iterator should give us complete lines, but be safe.
                        logger.debug(f"Could not parse stream line: {line[:200]!r}")
                        continue

                    msg = chunk.get("message", {}) or {}
                    delta_content = msg.get("content", "") or ""
                    delta_thinking = msg.get("thinking", "") or ""

                    if delta_thinking:
                        accumulated_thinking += delta_thinking
                    if delta_content:
                        accumulated_content += delta_content
                        content_started = True

                    # Notify caller — but only when something user-visible
                    # changed (or thinking changed and content hasn't started
                    # yet, so they can keep showing a "thinking..." indicator).
                    if on_partial is not None and (delta_content or (delta_thinking and not content_started)):
                        try:
                            await on_partial({
                                "phase": "content" if content_started else "thinking",
                                "content": accumulated_content,
                                "thinking": accumulated_thinking,
                                "done": False,
                            })
                        except Exception as cb_err:
                            # A failing callback shouldn't kill the stream.
                            logger.warning(f"on_partial callback raised: {cb_err}")

                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason", "unknown")
                        break

        print(f"DEBUG: stream finished, done_reason={done_reason}, "
              f"content_len={len(accumulated_content)}, thinking_len={len(accumulated_thinking)}")

        raw_content = accumulated_content
        thinking = accumulated_thinking

        # Fallback: if content is empty but thinking has text, try to rescue
        # the last drafted line (Gemma often writes multiple *Draft N:* lines).
        if not raw_content.strip() and thinking:
            print("DEBUG: Empty content but thinking present — attempting rescue from thinking trace")
            raw_content = _extract_from_thinking(thinking)
            print(f"DEBUG: Rescued from thinking: {repr(raw_content)}")
            # If the rescue produced something, hand it to on_partial as a
            # final synthetic update so the Discord message reflects it.
            if raw_content and on_partial is not None:
                try:
                    await on_partial({
                        "phase": "content",
                        "content": raw_content,
                        "thinking": thinking,
                        "done": False,
                    })
                except Exception as cb_err:
                    logger.warning(f"on_partial (rescue) raised: {cb_err}")

        full_response = raw_content.strip()

        if not full_response:
            print("DEBUG: The model generated a completely empty response.")
            if retry:
                print("DEBUG: Retrying once with temperature=0.9 and a nudge...")
                nudged = list(prompt) + [{
                    "role": "user",
                    "content": "(answer in one short sentence, don't be silent)"
                }]
                return await _retry_with_nudge(nudged, model=use_model, on_partial=on_partial)
            # Final on_partial so the caller knows we're done (even if empty)
            if on_partial is not None:
                try:
                    await on_partial({"phase": "content", "content": "",
                                      "thinking": thinking, "done": True})
                except Exception:
                    pass
            return None

        # Strip any "Message:" prefix the model might parrot from the RAG format
        content_start = full_response.find("Message:")
        if content_start != -1 and content_start < 20:
            # only strip if it's at the very start, not mid-response
            actual_content = full_response[content_start + 8:].strip()
        else:
            actual_content = full_response

        print(f"DEBUG: Final parsed content to send: {repr(actual_content[:200])}")

        # Final done=True notification so the caller can do its end-of-stream
        # work (final edit, history record, etc.). Pass the cleaned content.
        if on_partial is not None:
            try:
                await on_partial({
                    "phase": "content",
                    "content": actual_content,
                    "thinking": thinking,
                    "done": True,
                })
            except Exception as cb_err:
                logger.warning(f"on_partial (done) raised: {cb_err}")

        return actual_content

    except asyncio.TimeoutError:
        logger.error(f"Streaming request timed out (sock_read={STREAM_TIMEOUT_SOCK_READ_SEC}s, "
                     f"total={STREAM_TIMEOUT_TOTAL_SEC}s) — model may be stuck or unloaded")
        return None
    except Exception as e:
        print(f"DEBUG: Python Exception in get_llm_response: {repr(e)}")
        logger.error(f"Failed to get LLM response: {str(e)}", exc_info=True)
        return None


async def _retry_with_nudge(prompt, model=None, on_partial=None):
    """Second attempt with higher temp, no further recursion. Also streams so
    the in-progress Discord message keeps updating during the retry."""
    use_model = model or MODEL_ID_LLM
    try:
        gemma_safe = _fold_system_into_user(prompt)
        ollama_messages = _to_ollama_format(gemma_safe)

        timeout = aiohttp.ClientTimeout(
            total=STREAM_TIMEOUT_TOTAL_SEC,
            sock_read=STREAM_TIMEOUT_SOCK_READ_SEC,
        )

        accumulated_content = ""
        accumulated_thinking = ""

        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{LM_STUDIO_URL.rstrip('/')}/ollama/api/chat"
            async with session.post(url, headers=HEADERS, json={
                "model": use_model,
                "messages": ollama_messages,
                "stream": True,
                "think": False,
                "options": {
                    "temperature": 0.95,
                    "num_predict": 400,
                }
            }) as response:
                if response.status != 200:
                    return None
                async for raw_line in response.content:
                    line = raw_line.strip() if raw_line else b""
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message", {}) or {}
                    delta_content = msg.get("content", "") or ""
                    delta_thinking = msg.get("thinking", "") or ""
                    if delta_thinking:
                        accumulated_thinking += delta_thinking
                    if delta_content:
                        accumulated_content += delta_content
                        if on_partial is not None:
                            try:
                                await on_partial({
                                    "phase": "content",
                                    "content": accumulated_content,
                                    "thinking": accumulated_thinking,
                                    "done": False,
                                })
                            except Exception:
                                pass
                    if chunk.get("done"):
                        break

        raw = accumulated_content.strip()
        if not raw and accumulated_thinking:
            # Last-resort rescue from the thinking trace
            raw = _extract_from_thinking(accumulated_thinking)

        # Last-ditch fallback so the bot doesn't silently fail in chat
        final = raw if raw else "Hmph! I'm the strongest! ⑨"

        # Final on_partial so the Discord message reflects the retry result
        if on_partial is not None:
            try:
                await on_partial({
                    "phase": "content",
                    "content": final,
                    "thinking": accumulated_thinking,
                    "done": True,
                })
            except Exception:
                pass

        print(f"DEBUG: Retry produced: {repr(final[:200])}")
        return final
    except Exception as e:
        logger.error(f"Retry failed: {e}")
        if on_partial is not None:
            try:
                await on_partial({"phase": "content", "content": "Hmph! I'm the strongest! ⑨",
                                  "thinking": "", "done": True})
            except Exception:
                pass
        return "Hmph! I'm the strongest! ⑨"


def _extract_from_thinking(thinking_text):
    """
    When Gemma's reasoning model runs out of tokens mid-thought, content is empty
    but 'thinking' contains the chain-of-thought. Heuristically pull the most
    likely final answer from the trace.

    Strategy (in order of priority):
    1. Lines explicitly marked as decisions ("Let's go with:", "Final:", "Response:")
    2. The LAST quoted string that looks like a complete reply
    3. Lines starting with a drafted response pattern

    We skip any candidate that's clearly meta-commentary ("something simple...",
    "I should...", "the answer should...").
    """
    import re
    if not thinking_text:
        return ""

    META_PREFIXES = (
        "something ", "I should", "the answer", "let me", "the user",
        "this is", "that's", "it's a", "maybe I", "perhaps ", "probably ",
        "response:", "tone:", "response should", "answer should",
    )

    def is_meta(s):
        low = s.lower().strip()
        return any(low.startswith(p) for p in META_PREFIXES)

    # Strategy 1: explicit decision markers
    decision_patterns = [
        r"Let'?s\s+(?:go\s+with|try)[:\s]+[\"']?([^\"'\n]{3,200})[\"']?",
        r"Final(?:\s+answer)?[:\s]+[\"']?([^\"'\n]{3,200})[\"']?",
        r"(?:Final\s+)?Response[:\s]+[\"']?([^\"'\n]{3,200})[\"']?",
        r"(?:I'?ll\s+say|I'?ll\s+go\s+with)[:\s]+[\"']?([^\"'\n]{3,200})[\"']?",
    ]
    for pat in decision_patterns:
        matches = re.findall(pat, thinking_text, re.IGNORECASE)
        # Walk backwards — most recent decision wins
        for m in reversed(matches):
            m = m.strip(' "*.')
            if m and not is_meta(m) and len(m.split()) >= 2:
                return m

    # Strategy 2: last quoted string that isn't meta-commentary
    quotes = re.findall(r'"([^"\n]{3,200})"', thinking_text)
    for q in reversed(quotes):
        q = q.strip()
        if q and not is_meta(q) and len(q.split()) >= 2:
            # Also skip things that look like rephrased instructions
            if not q.lower().startswith(("rarely ", "never ", "always ", "only ")):
                return q

    # Strategy 3: "Draft N: ..." lines, last complete one
    drafts = re.findall(r'\*?Draft\s*\d+[:\*]?\s*\*?\s*(.+?)(?=\n|$)', thinking_text)
    for d in reversed(drafts):
        d = d.strip(' "*')
        if d and len(d.split()) >= 2 and not d.endswith(('...', 'Probably', 'Maybe')):
            return d

    return ""


def _to_ollama_format(messages):
    """
    Convert OpenAI-style messages (where content can be a list of text/image_url
    parts) into Ollama's native format (content is a string, images go in a
    separate 'images' list of raw base64 strings).

    See: https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-images
    """
    result = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            # Multi-part content (OpenAI vision format). Flatten to string + images.
            text_parts = []
            images = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    # Strip data URI prefix if present — Ollama wants raw base64
                    if url.startswith("data:"):
                        comma = url.find(",")
                        if comma != -1:
                            images.append(url[comma + 1:])
                    elif url:
                        # Could be a plain base64 string already, or an http URL.
                        # Ollama's native endpoint accepts base64; http URLs would
                        # need to be fetched upstream. Passing as-is for base64 case.
                        images.append(url)

            entry = {"role": role, "content": "\n".join(text_parts)}
            if images:
                entry["images"] = images
            result.append(entry)
            continue

        # Fallback: coerce whatever it is to string
        result.append({"role": role, "content": str(content)})

    return result


def _redact_images(messages):
    """For debug printing only — replace base64 image blobs with a placeholder
    so the console log stays readable."""
    redacted = []
    for msg in messages:
        m = dict(msg)
        if "images" in m and m["images"]:
            m["images"] = [f"<base64 image, {len(img)} chars>" for img in m["images"]]
        redacted.append(m)
    return redacted


def _fold_system_into_user(messages):
    """
    Gemma's chat template only supports user/assistant roles.
    Merge all system-role messages into the first user message as a preamble.
    """
    system_chunks = []
    other = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                system_chunks.append(c)
            elif isinstance(c, list):
                # extract text parts from vision-format content
                for part in c:
                    if part.get("type") == "text":
                        system_chunks.append(part.get("text", ""))
        else:
            other.append(m)

    if not system_chunks:
        return other

    preamble = "\n\n".join(s.strip() for s in system_chunks if s.strip())

    # Find first user message and prepend preamble to it
    result = []
    injected = False
    for m in other:
        if not injected and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                new_content = f"[System instructions]\n{preamble}\n[End system instructions]\n\n{c}"
                result.append({"role": "user", "content": new_content})
            elif isinstance(c, list):
                # vision format: prepend to the first text part
                new_list = list(c)
                for i, part in enumerate(new_list):
                    if part.get("type") == "text":
                        new_list[i] = {
                            "type": "text",
                            "text": f"[System instructions]\n{preamble}\n[End system instructions]\n\n{part.get('text', '')}"
                        }
                        break
                else:
                    new_list.insert(0, {"type": "text", "text": f"[System instructions]\n{preamble}\n[End system instructions]"})
                result.append({"role": "user", "content": new_list})
            injected = True
        else:
            result.append(m)

    # Edge case: no user message existed — create one
    if not injected:
        result.insert(0, {"role": "user", "content": preamble})

    return result


def _parse_msg_timestamp(ts):
    """Parse an ISO timestamp string from the history service into a tz-aware datetime.
    Returns None on failure."""
    if not ts:
        return None
    try:
        # Replace trailing 'Z' with '+00:00' for fromisoformat
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # If naive, assume UTC
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _is_before_cutoff(channel_id, msg):
    """
    True if `msg` was sent at or before the ctxbreak cutoff for this channel
    (and so should be hidden). False if no cutoff is set or the message is newer.
    """
    cutoff = _ctx_cutoff.get(str(channel_id))
    if cutoff is None:
        return False
    msg_ts = _parse_msg_timestamp(msg.get('timestamp'))
    if msg_ts is None:
        # Can't tell — be conservative and hide it (better to lose context than
        # leak post-cutoff context across the break).
        return True
    return msg_ts <= cutoff


def split_message(content):
    parts = []
    while len(content) > MAX_MESSAGE_LENGTH:
        split_index = content.rfind(' ', 0, MAX_MESSAGE_LENGTH)
        if split_index == -1:
            split_index = MAX_MESSAGE_LENGTH
        parts.append(content[:split_index])
        content = content[split_index:].lstrip()
    parts.append(content)
    return parts


async def send_split_message(channel, content, reply_to=None):
    """
    Send `content` to the channel, splitting if it exceeds Discord's length cap.
    If `reply_to` is a Message, the FIRST chunk is sent as a Discord reply
    (creating the threaded reply UI in Discord); subsequent chunks fall back
    to plain channel sends.

    `mention_author=False` keeps the reply visible without sending an extra
    ping on top of any @-mentions that may already be in the content.

    Returns the FIRST sent Message object (so callers can record its
    discord_message_id), or None if everything failed.
    """
    parts = split_message(content)
    first_sent = None
    for i, part in enumerate(parts):
        try:
            if i == 0 and reply_to is not None:
                sent = await reply_to.reply(part, mention_author=False)
            else:
                sent = await channel.send(part)
            if first_sent is None:
                first_sent = sent
        except discord.HTTPException as e:
            # Reply can fail if the original message was deleted — fall back
            # to a plain channel send rather than dropping the message.
            logger.warning(f"reply send failed ({e}), falling back to plain send")
            try:
                sent = await channel.send(part)
                if first_sent is None:
                    first_sent = sent
            except discord.HTTPException as e2:
                logger.error(f"plain send also failed ({e2})")
    return first_sent

def _relative_time(iso_timestamp: str) -> str:
    """
    Convert an ISO 8601 timestamp string to a human-readable relative time
    like '3 hours ago', '2 days ago', 'just now', etc.
    Returns the original string if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo)
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            m = seconds // 60
            return f"{m} minute{'s' if m != 1 else ''} ago"
        elif seconds < 86400:
            h = seconds // 3600
            return f"{h} hour{'s' if h != 1 else ''} ago"
        elif seconds < 604800:
            d = seconds // 86400
            return f"{d} day{'s' if d != 1 else ''} ago"
        elif seconds < 2592000:
            w = seconds // 604800
            return f"{w} week{'s' if w != 1 else ''} ago"
        else:
            m = seconds // 2592000
            return f"{m} month{'s' if m != 1 else ''} ago"
    except (ValueError, TypeError):
        return iso_timestamp  # graceful fallback


def _filter_rag_results(rag_results, current_message, history_ids):
    """
    Apply client-side filters to retrieved RAG results:
    - drop the current message itself (if the service indexed it before query)
    - drop messages already in the sliding history window (avoid duplication)
    - drop low-similarity hits if scores are present
    - drop assistant messages and other-bot messages (safety net; server should
      already exclude these but we don't trust it)
    - drop messages whose author looks like a bot we know about
    """
    bot_mention = f"<@{bot.user.id}>" if bot.user else None
    other_bot_mentions_lower = {b.lower() for b in OTHER_BOTS}

    current_id = str(current_message.id)
    current_content = current_message.content
    current_author_mention = f"<@{current_message.author.id}>"

    filtered = []
    for r in rag_results:
        # similarity threshold (only enforced if service returns a score)
        score = r.get('score') or r.get('similarity')
        if score is not None:
            try:
                if float(score) < RAG_MIN_SIMILARITY:
                    continue
            except (TypeError, ValueError):
                pass

        # role filter (server-side filter is best-effort; double-check here)
        role = r.get('role', 'human')
        if role == 'assistant' and not RAG_INCLUDE_ASSISTANT:
            continue
        if role == 'user':
            # other bots are stored as role="user"; humans as role="human"
            continue

        # author filter — drop other bots by name. Don't drop self here when
        # RAG_INCLUDE_ASSISTANT is on; the role check above already gates that.
        author = r.get('user', '') or ''
        if not RAG_INCLUDE_ASSISTANT and bot_mention and author == bot_mention:
            continue
        author_plain = author.lstrip('@').strip()
        if author_plain.lower() in other_bot_mentions_lower:
            continue

        # drop the current message if it sneaked into results.
        # Same layered match as in get_structured_input.
        if str(r.get('discord_message_id', '')) == current_id:
            continue
        if str(r.get('id', '')) == current_id:
            continue
        if r.get('content') == current_content and author == current_author_mention:
            continue

        # dedupe against sliding-window history
        rid = str(r.get('id', ''))
        if rid and rid in history_ids:
            continue

        filtered.append(r)

    return filtered


# Time-extraction setup. dateparser handles "last week", "yesterday",
# "3 days ago", "in March 2024", etc. — much more robust than rolling our own.
# If it's not installed, we fall back to a small regex-based parser that
# covers the common cases.
try:
    import dateparser  # type: ignore
    _HAS_DATEPARSER = True
except ImportError:
    dateparser = None  # type: ignore
    _HAS_DATEPARSER = False
    logger.info("dateparser not installed — using simple regex fallback for time queries. "
                "Install with `pip install dateparser` for better coverage.")

# Patterns we look for to detect time-expressions in queries.
# Order matters: longer/more-specific patterns first so they win.
# Each entry is (regex, callable(match) -> (start_dt, end_dt, label)).
import re as _re
from datetime import timedelta, timezone as _tz

def _now_utc():
    return datetime.now(_tz.utc)

def _start_of_day(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _regex_time_fallback(query):
    """
    Lightweight time extractor used when `dateparser` isn't installed.
    Returns (start_dt, end_dt, matched_substring) or None.

    Handles: yesterday, today, last week, last month, last N days/weeks/months,
    N days/weeks/months ago, this week, this month.
    """
    q = query.lower()
    now = _now_utc()
    today_start = _start_of_day(now)

    patterns = [
        # "yesterday"
        (r'\byesterday\b',
         lambda m: (today_start - timedelta(days=1), today_start, m.group(0))),
        # "today"
        (r'\btoday\b',
         lambda m: (today_start, now, m.group(0))),
        # "last week" — previous 7 days
        (r'\blast\s+week\b',
         lambda m: (today_start - timedelta(days=7), today_start, m.group(0))),
        # "this week" — last 7 days including today
        (r'\bthis\s+week\b',
         lambda m: (today_start - timedelta(days=7), now, m.group(0))),
        # "last month" — previous 30 days
        (r'\blast\s+month\b',
         lambda m: (today_start - timedelta(days=30), today_start, m.group(0))),
        # "this month"
        (r'\bthis\s+month\b',
         lambda m: (today_start - timedelta(days=30), now, m.group(0))),
        # "last N days/weeks/months/hours"
        (r'\blast\s+(\d+)\s+(hour|day|week|month)s?\b',
         lambda m: (
             now - timedelta(**{
                 'hour': lambda n: {'hours': n},
                 'day': lambda n: {'days': n},
                 'week': lambda n: {'days': 7 * n},
                 'month': lambda n: {'days': 30 * n},
             }[m.group(2)](int(m.group(1)))),
             now,
             m.group(0),
         )),
        # "N days/weeks/months ago" — point in past, give it a 24h window
        (r'\b(\d+)\s+(hour|day|week|month)s?\s+ago\b',
         lambda m: (
             (now - timedelta(**{
                 'hour': lambda n: {'hours': n + 1},
                 'day': lambda n: {'days': n, 'hours': 12},
                 'week': lambda n: {'days': 7 * n + 3},
                 'month': lambda n: {'days': 30 * n + 5},
             }[m.group(2)](int(m.group(1))))),
             (now - timedelta(**{
                 'hour': lambda n: {'hours': max(0, n - 1)},
                 'day': lambda n: {'days': max(0, n - 1)},
                 'week': lambda n: {'days': max(0, 7 * n - 3)},
                 'month': lambda n: {'days': max(0, 30 * n - 5)},
             }[m.group(2)](int(m.group(1))))),
             m.group(0),
         )),
        # "earlier today"
        (r'\bearlier\s+today\b',
         lambda m: (today_start, now, m.group(0))),
        # "an hour ago" / "a day ago" / "a week ago"
        (r'\ban?\s+(hour|day|week|month)\s+ago\b',
         lambda m: (
             now - timedelta(**{
                 'hour': {'hours': 2},
                 'day': {'days': 1, 'hours': 12},
                 'week': {'days': 8},
                 'month': {'days': 35},
             }[m.group(1)]),
             now - timedelta(**{
                 'hour': {'minutes': 30},
                 'day': {'hours': 12},
                 'week': {'days': 6},
                 'month': {'days': 25},
             }[m.group(1)]),
             m.group(0),
         )),
    ]

    for pattern, builder in patterns:
        m = _re.search(pattern, q, _re.IGNORECASE)
        if m:
            try:
                start, end, label = builder(m)
                return (start, end, label)
            except Exception as e:
                logger.debug(f"time fallback builder failed for {pattern!r}: {e}")
                continue
    return None


def _extract_time_range(query):
    """
    Try to find a time expression in the query. Returns:
      (start_dt, end_dt, matched_substring) or None.

    Both datetimes are tz-aware UTC. The matched_substring can be stripped
    from the query before embedding so it doesn't pollute the semantic match.
    """
    if not query:
        return None

    if _HAS_DATEPARSER:
        # dateparser's search_dates is the most flexible — finds time mentions
        # anywhere in the string. settings ensure tz-aware UTC output.
        try:
            from dateparser.search import search_dates  # type: ignore
            results = search_dates(
                query,
                settings={
                    'PREFER_DATES_FROM': 'past',
                    'RETURN_AS_TIMEZONE_AWARE': True,
                    'TIMEZONE': 'UTC',
                    'TO_TIMEZONE': 'UTC',
                },
            )
            if results:
                # Take the first hit — usually the most relevant. Multiple hits
                # ("between Monday and Friday") aren't worth the complexity.
                matched_text, dt = results[0]
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)

                # Heuristic window:
                #   - vague phrases ("last week") → use regex fallback's wider range
                #   - specific date/time → ±12h
                fallback = _regex_time_fallback(query)
                if fallback is not None:
                    return fallback

                start = dt - timedelta(hours=12)
                end = dt + timedelta(hours=12)
                # Don't let "end" go into the future
                now = _now_utc()
                if end > now:
                    end = now
                return (start, end, matched_text)
        except Exception as e:
            logger.debug(f"dateparser failed: {e} — falling back to regex")

    return _regex_time_fallback(query)


async def _expand_query_for_rag(query, channel_id, guild=None):
    """
    Expand mentions in a query so semantic search matches both how messages
    are stored (with <@id> mentions) and how users phrase questions
    (with plain "@name" or just bare names — with or without the @).

    Three passes:
      1. Existing <@id> mentions → "@DisplayName <@id>" so the embedding
         contains both readable name and ID.
      2. Plain '@name' or bare known display names from this channel's
         participants → suffixed with "<@id>".
      3. Bare names that match GUILD members (not just recent speakers) →
         suffixed with "<@id>". Lets users ask about people who haven't
         spoken in this specific channel yet.

    The query is used only for retrieval; the prompt sees the original message.
    """
    if not query:
        return query

    import re

    expanded = query
    already_added_ids = set()  # uids already injected, so we don't double-add

    # Pass 1: <@id> -> "@Name <@id>"
    for uid in set(_MENTION_RE.findall(expanded)):
        name = await _resolve_user_name(uid, guild=guild)
        if name:
            for raw in (f"<@{uid}>", f"<@!{uid}>"):
                expanded = expanded.replace(raw, f"@{name} <@{uid}>")
            already_added_ids.add(uid)

    def _try_inject(name_lc, uid):
        """Try to inject <@uid> into expanded if `name_lc` appears as @name or bare.
        Returns True if injected."""
        nonlocal expanded
        if uid in already_added_ids:
            return False
        mention_token = f"<@{uid}>"
        if mention_token in expanded:
            already_added_ids.add(uid)
            return False

        # "@name" form first (more specific)
        pattern_at = r'@' + re.escape(name_lc) + r'(?=\W|$)'
        if re.search(pattern_at, expanded, flags=re.IGNORECASE):
            expanded = re.sub(
                pattern_at,
                lambda m, _t=mention_token: f"{m.group(0)} {_t}",
                expanded,
                flags=re.IGNORECASE,
            )
            already_added_ids.add(uid)
            return True

        # Bare name as a whole word. Require >=3 chars to avoid matching short
        # noise words but be lenient enough for short usernames like "joe".
        if len(name_lc) >= 3:
            pattern_bare = r'(?:^|\W)' + re.escape(name_lc) + r'(?=\W|$)'
            if re.search(pattern_bare, expanded, flags=re.IGNORECASE):
                # Append the mention rather than splicing inline, to keep
                # the user's original phrasing intact.
                expanded = f"{expanded} {mention_token}"
                already_added_ids.add(uid)
                return True

        return False

    # Pass 2: channel participants (people who've actually spoken here)
    participants = _channel_participants.get(str(channel_id), {})
    if participants:
        # Sort longest-first so "@JohnDoe" wins over "@John"
        for lname in sorted(participants.keys(), key=len, reverse=True):
            _try_inject(lname, participants[lname])

    # Pass 3: guild members (anyone in the server, even if they haven't spoken).
    # Iterating all members can be expensive on huge servers, so we only do this
    # when the query plausibly mentions someone — a heuristic check for an "@"
    # or for a capitalized word that didn't already match a participant.
    if guild is not None:
        looks_like_query_about_person = (
            "@" in query
            or re.search(r'\bwhat\s+(did|does|is)\b', query, re.IGNORECASE)
            or re.search(r'\bwho\s+is\b', query, re.IGNORECASE)
        )
        if looks_like_query_about_person:
            # Build a {lowered_name: id} view of guild members to mirror the
            # participant lookup. Capped to avoid pathological costs on huge guilds.
            seen = set()
            count = 0
            for member in guild.members:
                if count > 2000:
                    break
                count += 1
                for nm in (member.display_name, member.name):
                    if not nm:
                        continue
                    nl = nm.lower()
                    if nl in seen or nl in participants:
                        continue
                    seen.add(nl)
                    if _try_inject(nl, str(member.id)):
                        break  # one match per member is enough

    return expanded


def _truncate(text, n=120):
    """Shorten a string to n chars with ellipsis. Safe on None/empty."""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n - 1].rstrip() + "…"


async def _build_reply_prefix(current_message, guild):
    """
    If the current Discord message is a reply to another message, build a short
    parenthetical reply marker so the LLM has explicit context. Uses
    "(↳ to @user: "...")" rather than square brackets — the latter looks like
    instruction syntax and the model tends to copy it.

    Returns "" if not a reply, or if the referenced message is unresolvable.
    """
    ref = getattr(current_message, 'reference', None)
    if not ref or not ref.resolved:
        return ""

    target = ref.resolved
    # ref.resolved can be a DeletedReferencedMessage in rare cases — guard for that
    target_author = getattr(target, 'author', None)
    if target_author is None:
        return ""

    author_name = target_author.display_name or target_author.name
    target_content = getattr(target, 'content', '') or ''
    target_content = await _rewrite_mentions(target_content, guild=guild)
    if not target_content and getattr(target, 'attachments', None):
        target_content = "(image/attachment)"
    snippet = _truncate(target_content, 140)
    if not snippet:
        return f"(↳ to @{author_name})"
    return f"(↳ to @{author_name}: \"{snippet}\")"


def _format_history_reply_prefix(msg, history_by_id_or_discord):
    """
    For a history entry that has `replying_to`, try to find the referenced
    message in the same history slice and build a short prefix.

    `history_by_id_or_discord` is a dict keyed by both service id and
    discord_message_id so we can look up by either. Returns "" if the
    referenced message isn't in the slice.
    """
    rt = msg.get('replying_to')
    if not rt:
        return ""
    # `replying_to` is currently stored as a user mention like "<@1234>" — that's
    # the AUTHOR of the reply target, not the message id. We can't resolve a
    # specific message from that alone. Best we can do is name the target user
    # so the LLM at least knows who's being replied to.
    return f"[reply to {rt}]"


async def get_structured_input(channel_id, current_message):
    # Pull a wider slice from the service when a ctxbreak cutoff is active —
    # we'll filter the older entries out below, and want the post-cutoff window
    # to still have HISTORY_WINDOW worth of messages.
    cid = str(channel_id)
    cutoff_active = cid in _ctx_cutoff
    history_pull = HISTORY_WINDOW * 4 if cutoff_active else HISTORY_WINDOW

    # Guild context for resolving user mentions to display names — needed
    # for the RAG query expansion below.
    guild = current_message.guild

    history = await get_conversation_history(channel_id, limit=history_pull)

    # ---- RAG query construction ----
    # 1. Extract any time expression ("last week", "yesterday", etc.) so we can
    #    pass it to the service as a date filter instead of polluting the
    #    semantic match.
    raw_query = current_message.content
    time_range = _extract_time_range(raw_query)
    after_dt = before_dt = None
    if time_range:
        after_dt, before_dt, matched = time_range
        # Strip the matched time phrase from the query so embedding focuses on
        # what was said rather than when. Whitespace cleanup keeps it readable.
        cleaned = raw_query.replace(matched, " ").replace("  ", " ").strip()
        # If stripping leaves the query empty, keep the original — better to
        # have a noisy semantic match than no query at all.
        query_for_embedding = cleaned if cleaned else raw_query
        logger.debug(f"RAG time filter: matched={matched!r} "
                     f"after={after_dt.isoformat()} before={before_dt.isoformat()}")
    else:
        query_for_embedding = raw_query

    # 2. Expand mentions / bare names so they match stored <@id> form.
    rag_query = await _expand_query_for_rag(query_for_embedding, channel_id, guild=guild)
    if rag_query != raw_query:
        logger.debug(f"RAG query expanded: {raw_query!r} -> {rag_query!r}")

    # 3. Apply ctxbreak cutoff to the time filter. If user said "last week" but
    #    a ctxbreak happened 2 days ago, we should not return anything older
    #    than the cutoff.
    if cutoff_active:
        cutoff_ts = _ctx_cutoff[cid]
        if after_dt is None or after_dt < cutoff_ts:
            after_dt = cutoff_ts

    relevant_context = await get_relevant_context(
        channel_id, rag_query, after=after_dt, before=before_dt,
    )

    # ctxbreak enforcement — drop everything older than the cutoff from BOTH
    # the sliding window and RAG. The service-side `after` filter (above) does
    # the same job for RAG, but we keep this client-side filter as a safety net
    # in case the service doesn't honor `after` yet.
    if cutoff_active:
        history = [m for m in history if not _is_before_cutoff(channel_id, m)]
        # Trim to window after filtering — the wider pull was just to ensure
        # we still have enough recent context if many messages were cut.
        history = history[-HISTORY_WINDOW:] if len(history) > HISTORY_WINDOW else history
        relevant_context = [m for m in relevant_context if not _is_before_cutoff(channel_id, m)]

    # If the user asked a time-bounded question but the service didn't honor
    # the filter, enforce it client-side too. This is a safety net.
    if after_dt is not None or before_dt is not None:
        def _in_range(m):
            ts = _parse_msg_timestamp(m.get('timestamp'))
            if ts is None:
                return False
            if after_dt is not None and ts < after_dt:
                return False
            if before_dt is not None and ts > before_dt:
                return False
            return True
        relevant_context = [m for m in relevant_context if _in_range(m)]

    async def rewrite(text, msg_id=None):
        return await _rewrite_mentions_cached(text, msg_id, guild=guild)

    # Build set of message IDs already in the sliding window so RAG can dedupe
    history_ids = {str(m.get('id', '')) for m in history if m.get('id') is not None}

    # Filter RAG results: drop bot/assistant noise, dupes, and low-similarity hits
    relevant_context = _filter_rag_results(relevant_context, current_message, history_ids)

    system_content = CIRNO_PERSONA + "\n\n"
    system_content += "You are in a conversation with multiple humans on Discord. Respond naturally.\n\n"

    # If we have a stored personality summary for the user who just spoke,
    # include it as factual context. This is generated by a separate analyst
    # prompt (not Cirno's voice) — see summarize_user(). Treat it as
    # background knowledge: things she "remembers" about this person.
    if ENABLE_PERSONALITY_SUMMARIES:
        try:
            current_author_name = current_message.author.display_name or current_message.author.name
            personality = await get_personality(str(channel_id), str(current_message.author.id))
            if personality and (personality.get('summary') or '').strip():
                system_content += (
                    f"What you remember about @{current_author_name} from past conversations: "
                    f"{personality['summary'].strip()}\n\n"
                )
        except Exception as e:
            logger.debug(f"Could not load personality for current speaker: {e}")

    structured_input = [
        {"role": "system", "content": system_content.strip()}
    ]

    # If we have RAG hits, inject them as a separate, clearly-labeled user turn
    # rather than gluing them onto the persona. The model can then treat them
    # as background context without confusing them for live conversation.
    if relevant_context:
        excerpt_lines = ["[Background — earlier messages from this channel that may be relevant: (make a summary to know the personality of the person you are replying to in ur thinking process) "
                         "These are NOT part of the current conversation and should not be replied to directly.]"]
        for msg in relevant_context:
            safe_content = await rewrite(msg.get('content', ''), msg.get('id'))
            safe_user = await rewrite(msg.get('user', ''), None)
            ts = msg.get('timestamp', '')
            excerpt_lines.append(f"  - {safe_user} ({_relative_time(ts)}): {safe_content}")
        excerpt_lines.append("[End background.]")
        structured_input.append({
            "role": "user",
            "content": "\n".join(excerpt_lines),
        })
        # Add a brief assistant ack so the next user turn (live conversation) sits
        # against an assistant turn — helps Gemma's strict alternation expectations.
        structured_input.append({
            "role": "assistant",
            "content": "(noted)",
        })

    current_msg_id = str(current_message.id)
    current_author_mention = f"<@{current_message.author.id}>"
    bot_mention = f"<@{bot.user.id}>"

    def _is_current_message(msg):
        """
        Decide whether a history entry IS the message we're currently responding
        to (and therefore should be excluded — we'll add it explicitly below).

        Layered checks:
          1. Discord snowflake match (preferred — only works if the service
             stores and returns `discord_message_id`).
          2. Service's own `id` matching the snowflake (works if the service
             happens to use the snowflake as its primary key).
          3. Fallback: same author + same content + within 60s of now.
             Bounded by recency so legitimate repeats ("gm cirno" said again
             tomorrow) aren't clobbered.
        """
        # 1. Service stores discord_message_id explicitly
        if str(msg.get('discord_message_id', '')) == current_msg_id:
            return True
        # 2. Service used the snowflake as its primary id
        if str(msg.get('id', '')) == current_msg_id:
            return True
        # 3. Recency-bounded content+author fallback
        if (msg.get('content') == current_message.content
                and msg.get('user') == current_author_mention):
            ts = _parse_msg_timestamp(msg.get('timestamp'))
            if ts is None:
                # No timestamp — assume it's the current one (safer than duplicating)
                return True
            from datetime import timezone
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age < 60:
                return True
        return False

    # Build a lookup map from any message id (service id OR discord snowflake)
    # to the message dict, so we can resolve "this is a reply to <id>" by
    # looking up the actual content of the target. Quoting the target in the
    # prompt is what makes reply chains comprehensible to the LLM.
    msg_lookup = {}
    for hmsg in history:
        sid = str(hmsg.get('id') or '')
        dmid = str(hmsg.get('discord_message_id') or '')
        if sid:
            msg_lookup[sid] = hmsg
        if dmid:
            msg_lookup[dmid] = hmsg

    async def _quote_of(target_msg, max_chars=140):
        """Render a one-line blockquote of a history message: '> @user: content'.
        Truncates long content. Returns '' if target_msg is None/empty."""
        if not target_msg:
            return ""
        content = (target_msg.get('content') or '').strip()
        if not content:
            return ""
        content_rewritten = await rewrite(content, target_msg.get('id'))
        # Collapse to one line so the blockquote stays compact
        content_rewritten = content_rewritten.replace('\n', ' ').strip()
        if len(content_rewritten) > max_chars:
            content_rewritten = content_rewritten[:max_chars - 1].rstrip() + "…"
        # Identify the speaker
        is_assistant = (target_msg.get('role') == 'assistant'
                        or target_msg.get('user') == bot_mention)
        if is_assistant:
            speaker = f"@{BOT_NAME}"
        else:
            speaker_raw = target_msg.get('user', '') or ''
            speaker = await rewrite(speaker_raw, None) if speaker_raw else "someone"
        return f"> {speaker}: {content_rewritten}"

    for msg in history:
        if _is_current_message(msg):
            continue

        # Look up the message being replied to (if any) and quote it inline.
        # This is the key fix: showing the LLM WHAT was replied to, not just
        # WHO it was directed at.
        reply_target_id = msg.get('replying_to_message_id')
        quote_line = ""
        if reply_target_id:
            target = msg_lookup.get(str(reply_target_id))
            if target:
                quote_line = await _quote_of(target)

        if msg.get('role') == 'assistant' or msg.get('user') == bot_mention:
            content_rewritten = await rewrite(msg.get('content', ''), msg.get('id'))
            # Format: blockquote of target above, then Cirno's response below
            if quote_line:
                content_block = f"{quote_line}\n{content_rewritten}".strip()
            else:
                content_block = content_rewritten.strip()
            structured_input.append({
                "role": "assistant",
                "content": content_block,
            })
        else:
            speaker_user = await rewrite(msg.get('user', ''), None)
            speaker_content = await rewrite(msg.get('content', ''), msg.get('id'))
            if quote_line:
                # Indent the quote with the speaker so the structure stays clear.
                content_block = f"{speaker_user}:\n{quote_line}\n{speaker_content}".strip()
            else:
                content_block = f"{speaker_user}: {speaker_content}".strip()
            structured_input.append({
                "role": "user",
                "content": content_block,
            })

    # ---- Current message ----
    author_name = current_message.author.display_name or current_message.author.name
    rewritten_content = await _rewrite_mentions(current_message.content, guild=guild)

    # Build a blockquote of the message being replied to, if any. We try the
    # live Discord reference first (it's authoritative and includes deleted-but-
    # cached messages), then fall back to the lookup map (in case the reference
    # wasn't resolved by Discord but we have it in our history slice).
    current_quote = ""
    ref = getattr(current_message, 'reference', None)
    if ref and ref.resolved and getattr(ref.resolved, 'author', None):
        target_author = ref.resolved.author
        target_content = (getattr(ref.resolved, 'content', '') or '').strip()
        target_content = await _rewrite_mentions(target_content, guild=guild)
        if not target_content and getattr(ref.resolved, 'attachments', None):
            target_content = "(image/attachment)"
        target_content = target_content.replace('\n', ' ').strip()
        if len(target_content) > 140:
            target_content = target_content[:139].rstrip() + "…"
        # Identify the target speaker — Cirno or someone else
        if target_author == bot.user:
            target_speaker = f"@{BOT_NAME}"
        else:
            target_speaker = "@" + (target_author.display_name or target_author.name)
        if target_content:
            current_quote = f"> {target_speaker}: {target_content}"
        else:
            current_quote = f"> {target_speaker}: (no text)"
    elif ref and ref.message_id:
        # Live ref didn't resolve but we might have the target in history
        target = msg_lookup.get(str(ref.message_id))
        if target:
            current_quote = await _quote_of(target)

    if current_quote:
        text_content = f"@{author_name}:\n{current_quote}\n{rewritten_content}".strip()
    else:
        text_content = f"@{author_name}: {rewritten_content}"

    # --- Gather all visual content into one labeled bundle ---
    # Why labels: when we attach 3 images (an emoji + a sticker + an attachment),
    # the model needs to know which image is which. We number them and add
    # text annotations so it can refer to them precisely.

    visual_items = []  # each: {"b64": <str>, "media_type": <str>, "label": <str>}

    # 1. Real image attachments (the bot was already doing this — keep it)
    if current_message.attachments:
        for attachment in current_message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                try:
                    image_bytes = await attachment.read()
                    visual_items.append({
                        "b64": base64.b64encode(image_bytes).decode('utf-8'),
                        "media_type": attachment.content_type,
                        "label": f"attachment ({attachment.filename})",
                    })
                except Exception as e:
                    logger.debug(f"Could not read attachment {attachment.filename}: {e}")

    # 2. Stickers — annotate text always, attach bytes if vision-readable.
    if VISION_INCLUDE_STICKERS:
        sticker_visuals = await _gather_sticker_visuals(current_message)
        for text_annotation, b64, name in sticker_visuals:
            # Always append the text annotation so the model knows a sticker
            # was sent even if we couldn't fetch (e.g. Lottie sticker).
            text_content = (text_content + "\n" + text_annotation).strip()
            if b64:
                visual_items.append({
                    "b64": b64,
                    "media_type": "image/png",  # Discord's sticker URL serves PNG
                    "label": f"sticker :{name}:",
                })

    # 3. Custom emoji images — relaxed gate. Previously we only fetched these
    #    when the message was almost entirely emojis ("mostly emoji" check).
    #    Now: if any custom emojis are present and we have room in the budget,
    #    show them to vision so the meaning of an emoji-laced message comes
    #    through. The :name: text substitution alone is enough only when the
    #    name itself is descriptive, which is rare.
    if VISION_EMOJIS:
        emoji_info = _extract_emoji_urls(current_message.content,
                                          max_emojis=VISION_MAX_EMOJIS_INLINE)
        for name, url, is_animated in emoji_info:
            b64 = await _fetch_emoji_as_base64(url)
            if b64:
                visual_items.append({
                    "b64": b64,
                    "media_type": "image/gif" if is_animated else "image/png",
                    "label": f"emoji :{name}:",
                })

    # 4. Image RAG — search the bot's image memory for captions that
    #    semantically match the current message. Use the same expanded
    #    query we built for text RAG; the captioner's English captions
    #    will match natural-language questions much better than emoji
    #    spam or mention syntax.
    image_hits = []
    if ENABLE_IMAGE_MEMORY:
        try:
            raw_image_hits = await get_relevant_images(channel_id, rag_query,
                                                       limit=IMAGE_RAG_LIMIT)
            # Apply our minimum-score threshold client-side (server returns scores).
            for h in raw_image_hits:
                score = h.get("score")
                if score is None or float(score) >= IMAGE_RAG_MIN_SCORE:
                    image_hits.append(h)
            # Don't re-suggest images from the current message itself
            current_dmid = str(current_message.id)
            image_hits = [h for h in image_hits
                          if str(h.get("discord_message_id", "")) != current_dmid]
        except Exception as e:
            logger.debug(f"image RAG failed (non-fatal): {e}")

    # If we have image hits, build a system note listing them with stable IDs.
    # The bot is told it can include `[recall_image: <id>]` in its reply to
    # re-attach the actual image to the response. We strip the marker from
    # text and attach the bytes after streaming finishes.
    if image_hits:
        recall_lines = ["[Visual memory — past images this channel has shared "
                        "that might be relevant. Each has a stable id; include "
                        "[recall_image: <id>] in your reply to re-attach the "
                        "image. Do NOT make up ids — only the ones listed below "
                        "are valid.]"]
        for h in image_hits:
            cap = (h.get("caption") or "").strip()
            saved_at_rel = _relative_time(h.get("saved_at", ""))
            uploader = h.get("user_id") or ""
            uploader_name = await _rewrite_mentions(uploader, guild=guild)
            recall_lines.append(
                f"  - id={h['id']} (uploaded by {uploader_name}, {saved_at_rel}): "
                f"\"{cap}\""
            )
        recall_lines.append("[End visual memory.]")
        # Inject as another labeled user turn (we'll merge consecutive same-role
        # turns at the end of this function). Adding before the live message.
        structured_input.append({
            "role": "user",
            "content": "\n".join(recall_lines),
        })
        structured_input.append({
            "role": "assistant",
            "content": "(noted)",
        })

    # Poll-tool primer — once every POLL_PRIME_EVERY_N user messages in this
    # channel, remind the model that the [create_poll: ...] tool exists. We
    # don't put this in the persona because the persona is sticky across all
    # turns; we want this prompt to be a periodic, faint nudge — there one
    # turn, gone the next.
    primer = _maybe_poll_primer_for_channel(str(channel_id))
    if primer:
        structured_input.append({
            "role": "user",
            "content": primer,
        })
        structured_input.append({
            "role": "assistant",
            "content": "(noted)",
        })

    # 5. Combine the text content with all visual items into a vision-format
    #    user turn. Each image gets a "[image N: <label>]" marker in the text
    #    so the model can address them precisely.
    if visual_items:
        # Add labels for each image
        if len(visual_items) == 1:
            text_content = text_content.rstrip() + f"\n[image: {visual_items[0]['label']}]"
        else:
            for i, item in enumerate(visual_items, start=1):
                text_content = text_content.rstrip() + f"\n[image {i}: {item['label']}]"

        content_list = [{"type": "text", "text": text_content}]
        for item in visual_items:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{item['media_type']};base64,{item['b64']}"}
            })
        structured_input.append({
            "role": "user",
            "content": content_list
        })
    else:
        structured_input.append({
            "role": "user",
            "content": text_content
        })

    # Merge consecutive same-role messages (many chat templates require strict alternation)
    merged_input = []
    for msg in structured_input:
        if not merged_input:
            merged_input.append(msg)
        elif merged_input[-1]["role"] == msg["role"]:
            prev_content = merged_input[-1]["content"]
            curr_content = msg["content"]

            if isinstance(prev_content, str) and isinstance(curr_content, str):
                merged_input[-1]["content"] = prev_content + "\n" + curr_content
            elif isinstance(prev_content, str) and isinstance(curr_content, list):
                curr_content[0]["text"] = prev_content + "\n" + curr_content[0]["text"]
                merged_input[-1]["content"] = curr_content
            elif isinstance(prev_content, list) and isinstance(curr_content, str):
                prev_content[0]["text"] += "\n" + curr_content
            elif isinstance(prev_content, list) and isinstance(curr_content, list):
                prev_content[0]["text"] += "\n" + curr_content[0]["text"]
                # append any images from curr
                prev_content.extend([c for c in curr_content if c.get("type") == "image_url"])
        else:
            merged_input.append(msg)

    # Ensure first non-system turn is 'user' (Gemma/most chat templates require this)
    if len(merged_input) > 1 and merged_input[1]["role"] == "assistant":
        merged_input.pop(1)

    # Return the offered image ids alongside the prompt so the caller can
    # validate any [recall_image: N] tokens the model emits in its reply.
    offered_image_ids = {int(h["id"]) for h in image_hits if h.get("id") is not None}
    return merged_input, offered_image_ids


async def _stream_to_discord(channel, prompt, *, reply_to=None, channel_id_str,
                             retry=True, model=None, enable_thinking=None,
                             num_predict=None, temperature=0.7,
                             placeholder_text=None):
    """
    End-to-end streaming: send a placeholder message, stream tokens from the
    LLM, edit the placeholder as content arrives (throttled to respect
    Discord's edit rate limit), and split into overflow messages when the
    reply exceeds Discord's 2000-char cap.

    Cleanup (strip leaked reply markers, rewrite @name → <@id>) is applied to
    the accumulator on every edit, so the user sees clean text the entire
    time — no in-character marker artifacts flashing up briefly.

    Args:
        placeholder_text: text shown until the first content token arrives.
                          Defaults to STREAM_PLACEHOLDER. Pass STREAM_PLACEHOLDER_WAKING
                          when /api/ps says the model isn't loaded.

    Returns (final_cleaned_text, first_sent_message_or_None).
      - final_cleaned_text: the post-cleanup content the bot actually said.
        Empty string if generation produced nothing usable AND the retry
        was either disabled or also empty.
      - first_sent_message: the placeholder message (now edited to the final
        text). Used by the caller to capture the discord_message_id for
        recording in chat history.
    """
    # 1. Send the placeholder. As a Discord reply if `reply_to` is given,
    #    otherwise a plain channel message. We use mention_author=False on
    #    the reply because @-pings inside the eventual content are enough.
    initial_placeholder = placeholder_text or STREAM_PLACEHOLDER
    try:
        if reply_to is not None:
            placeholder_msg = await reply_to.reply(initial_placeholder, mention_author=False)
        else:
            placeholder_msg = await channel.send(initial_placeholder)
    except discord.HTTPException as e:
        logger.warning(f"Could not send placeholder via reply ({e}); falling back to plain send")
        try:
            placeholder_msg = await channel.send(initial_placeholder)
        except discord.HTTPException as e2:
            logger.error(f"Plain placeholder send also failed: {e2}")
            return "", None

    # State for the on_partial callback. Encapsulated in a dict so the closure
    # can mutate it. (Python closures can read enclosing-scope vars freely but
    # writing requires `nonlocal` for each — a dict is simpler.)
    state = {
        "last_edit_at": 0.0,                # monotonic seconds of last edit
        "last_edited_text": initial_placeholder,  # what's currently shown in Discord
        "overflow_msgs": [],                # additional messages for content > 2000 chars
        "first_chunk_seen": False,          # flips True when the first content delta arrives
        "any_chunk_seen": False,            # flips True on ANY chunk (thinking or content)
    }

    def _clean(text):
        """Apply the same post-processing the non-streaming path used to do at the end:
        strip leaked reply markers, then rewrite plain @name → <@id> for known
        channel participants. Also strip [recall_image: N] and [create_poll: ...]
        markers so users never see them in chat — the marker validation/handling
        happens post-stream in the caller. Cheap regex; safe to run on every edit."""
        if not text:
            return ""
        cleaned = _strip_leaked_reply_markers(text)
        cleaned = _rewrite_output_mentions(cleaned, channel_id_str)
        # Strip recall + poll markers regardless of validity — they're metadata,
        # not user-facing text.
        cleaned = RECALL_IMAGE_RE.sub("", cleaned)
        cleaned = CREATE_POLL_RE.sub("", cleaned)
        # Tidy whitespace runs left by the strips
        cleaned = _re_module.sub(r'[ \t]+', ' ', cleaned).strip()
        return cleaned

    async def _render(full_text, *, force=False):
        """
        Render `full_text` across the placeholder + any overflow messages.

        `force=True` skips the throttle (used for the final edit after the
        stream completes). Otherwise we only edit if the text changed AND
        either (a) STREAM_EDIT_INTERVAL_SEC has elapsed since the last edit,
        or (b) we've accumulated at least STREAM_EDIT_MIN_CHARS of new text.
        """
        if not full_text:
            full_text = STREAM_PLACEHOLDER  # never let the message go empty mid-stream

        now = time.monotonic()
        new_chars = len(full_text) - len(state["last_edited_text"])
        elapsed = now - state["last_edit_at"]

        if not force:
            # Throttle: skip if text is unchanged or too soon AND too short
            if full_text == state["last_edited_text"]:
                return
            if elapsed < STREAM_EDIT_INTERVAL_SEC and new_chars < STREAM_EDIT_MIN_CHARS:
                return

        # Slice the text into Discord-sized chunks. The first chunk goes into
        # the placeholder; subsequent chunks need overflow messages.
        chunks = split_message(full_text)

        # Edit the placeholder with chunk[0]
        try:
            await placeholder_msg.edit(content=chunks[0] or STREAM_PLACEHOLDER)
        except discord.HTTPException as e:
            # Most likely a rate-limit (429) or message-deleted (404). Both are
            # recoverable: skip this edit and try again on the next one.
            logger.debug(f"placeholder edit skipped ({e})")
            return

        # Handle overflow: if there are more chunks than overflow messages,
        # send new ones. If the existing overflow messages are out of sync
        # with the chunks (text changed), edit them.
        for i in range(1, len(chunks)):
            if i - 1 < len(state["overflow_msgs"]):
                # Edit existing overflow message
                ov = state["overflow_msgs"][i - 1]
                try:
                    await ov.edit(content=chunks[i])
                except discord.HTTPException as e:
                    logger.debug(f"overflow[{i-1}] edit skipped ({e})")
            else:
                # Send a new overflow message
                try:
                    sent = await channel.send(chunks[i])
                    state["overflow_msgs"].append(sent)
                except discord.HTTPException as e:
                    logger.warning(f"overflow send failed ({e}); truncating reply at this point")
                    break

        state["last_edited_text"] = full_text
        state["last_edit_at"] = now

    async def on_partial(s):
        """Callback handed to get_llm_response. Rebuilds clean text from the
        accumulator and asks _render to maybe-edit the Discord message."""
        # First chunk seen: if we started with the "waking up" placeholder,
        # swap it to the regular "thinking" placeholder now that the model is
        # actually responding. We do this even for thinking-phase chunks since
        # what they signal — "the model is alive and working" — matters more
        # than which kind of token it is.
        if not state["any_chunk_seen"]:
            state["any_chunk_seen"] = True
            if state["last_edited_text"] == STREAM_PLACEHOLDER_WAKING:
                try:
                    await placeholder_msg.edit(content=STREAM_PLACEHOLDER)
                    state["last_edited_text"] = STREAM_PLACEHOLDER
                except discord.HTTPException:
                    pass

        if s.get("phase") == "thinking" and not state["first_chunk_seen"]:
            # Still in chain-of-thought phase; don't show anything yet, just
            # leave the placeholder visible.
            return

        # Once we have any content, we're past thinking phase
        state["first_chunk_seen"] = True

        cleaned = _clean(s.get("content", "") or "")
        if not cleaned:
            # Cleaning stripped everything — keep showing the placeholder.
            return

        await _render(cleaned, force=bool(s.get("done")))

    # 2. Run the streaming LLM call. The on_partial callback drives the edits.
    final_raw = await get_llm_response(
        prompt,
        retry=retry,
        model=model,
        enable_thinking=enable_thinking,
        num_predict=num_predict,
        temperature=temperature,
        on_partial=on_partial,
    )

    if not final_raw:
        # Total failure (and retry already happened inside get_llm_response if enabled).
        # Edit the placeholder away so the user doesn't see "thinking..." forever.
        # We pick a quiet fallback rather than a verbose error to stay in character.
        try:
            await placeholder_msg.edit(content="...")
        except discord.HTTPException:
            pass
        return "", placeholder_msg, ""

    # 3. Final cleanup pass on the complete response. on_partial(done=True)
    #    already did this, but defensive: ensure the message is in its final
    #    cleaned form.
    final_cleaned = _clean(final_raw)
    if not final_cleaned.strip():
        # Cleanup left nothing useful (rare — only if response was 100% reply markers).
        try:
            await placeholder_msg.edit(content="...")
        except discord.HTTPException:
            pass
        return "", placeholder_msg, final_raw

    # Force one more render to make sure the displayed text matches final_cleaned
    # exactly (the last throttled edit may have been a few tokens behind).
    if final_cleaned != state["last_edited_text"]:
        await _render(final_cleaned, force=True)

    # Return cleaned (for display + history), placeholder (for snowflake), and
    # raw (for [recall_image: N] parsing — the caller does the actual extraction
    # because only it knows which ids were offered to the model this turn).
    return final_cleaned, placeholder_msg, final_raw


async def process_message_queue(channel_id):
    global last_response_time

    if channel_id in message_queue and channel_id not in processing_channels:
        processing_channels.add(channel_id)
        try:
            while channel_id in message_queue:
                messages = message_queue.pop(channel_id)

                # Same trigger logic as on_message — keep them in sync.
                # Image/emoji/random auto-replies were removed; only direct
                # interaction (mention, reply-to-Cirno, other bot) reaches here.
                relevant_messages = [
                    msg for msg in messages
                    if bot.user.mentioned_in(msg)
                    or _message_mentions_bot_by_name(msg.content)
                    or any(msg.author.name.upper() == ob for ob in OTHER_BOTS)
                    or (msg.reference and msg.reference.resolved
                        and msg.reference.resolved.author == bot.user)
                    or getattr(msg, '_cirno_unprompted', False)  # see chime_in_loop
                ]

                if not relevant_messages:
                    logger.debug(f"No relevant messages found for channel {channel_id}")
                    continue

                for msg in relevant_messages:
                    structured_input, offered_image_ids = await get_structured_input(
                        channel_id, msg)

                    logger.debug(f"Constructed prompt for channel {channel_id} "
                                 f"(offered_image_ids={sorted(offered_image_ids)})")

                    # Decide whether to use Discord's reply mechanism.
                    # YES if this was a direct interaction (mention or a
                    # reply to one of Cirno's prior messages). NO if it's
                    # an unprompted chime — that should look like a
                    # spontaneous channel message, not a targeted reply.
                    is_direct_interaction = (
                        bot.user.mentioned_in(msg)
                        or _message_mentions_bot_by_name(msg.content)
                        or (msg.reference and msg.reference.resolved
                            and msg.reference.resolved.author == bot.user)
                    )
                    unprompted = getattr(msg, '_cirno_unprompted', False)
                    reply_to_msg = msg if (is_direct_interaction and not unprompted) else None

                    # Pre-flight: ask Ollama whether the model is currently
                    # loaded. If not, the first chat call will block on a cold
                    # weight load (10–60s); telling the user "waking up..."
                    # rather than "thinking..." sets correct expectations.
                    is_loaded = await _check_model_loaded(MODEL_ID_LLM)
                    if is_loaded is False:
                        ph = STREAM_PLACEHOLDER_WAKING
                    else:
                        ph = STREAM_PLACEHOLDER

                    # Show the typing indicator only briefly — we'll send the
                    # placeholder message right after, which serves as a more
                    # informative "I'm working on it" signal than the indicator.
                    async with msg.channel.typing():
                        # Stream the response into Discord. Returns:
                        #   final_text: cleaned text shown to the user
                        #   sent_msg: the placeholder, now edited
                        #   final_raw: uncleaned text — still has [recall_image: N]
                        #              markers so we can extract requested ids
                        final_text, sent_msg, final_raw = await _stream_to_discord(
                            msg.channel,
                            structured_input,
                            reply_to=reply_to_msg,
                            channel_id_str=str(channel_id),
                            placeholder_text=ph,
                        )

                    if final_text:
                        # If the model asked for image recalls, validate the ids
                        # against what we offered, fetch bytes, and attach as a
                        # follow-up message. We don't try to attach to the
                        # placeholder itself because Discord doesn't allow
                        # adding attachments via edit() — only the initial send.
                        recall_ids = []
                        if offered_image_ids:
                            _cleaned_again, recall_ids = _extract_recall_image_ids(
                                final_raw, offered_image_ids)

                        if recall_ids:
                            # Cap to a sane number to avoid spamming
                            recall_ids = recall_ids[:3]
                            files_to_send = []
                            for img_id in recall_ids:
                                data = await fetch_image_bytes(img_id)
                                if not data:
                                    continue
                                try:
                                    raw = base64.b64decode(data["image_b64"])
                                except Exception as e:
                                    logger.debug(f"recall: bad b64 for id={img_id}: {e}")
                                    continue
                                # Pick a reasonable filename based on media_type
                                ext = _ext_from_media_type(data.get("media_type", ""))
                                fname = f"recall_{img_id}.{ext}"
                                import io
                                files_to_send.append(
                                    discord.File(io.BytesIO(raw), filename=fname))
                            if files_to_send:
                                try:
                                    await msg.channel.send(files=files_to_send)
                                    logger.info(f"recall: attached {len(files_to_send)} "
                                                f"image(s) ids={recall_ids}")
                                except discord.HTTPException as e:
                                    logger.warning(f"recall: attach send failed: {e}")

                        # Poll creation — parse [create_poll: ...] markers from
                        # the raw response. The display version (final_text)
                        # already had them stripped via _clean(); we use the
                        # raw text so the parser can find them.
                        if ENABLE_POLLS:
                            poll_specs, _ = _parse_create_poll_markers(final_raw)
                            if poll_specs:
                                # Cap to 1 poll per reply so the bot can't spam
                                # multiple polls in one turn. If the model
                                # emitted several, take the first valid one.
                                poll_specs = poll_specs[:1]
                                created = await _create_polls_from_specs(
                                    msg.channel, poll_specs)
                                # Record each created poll into chat history so
                                # RAG can match later "what did we vote about X"
                                # queries.
                                for poll_msg, spec in created:
                                    try:
                                        await _record_poll_creation(
                                            channel_id,
                                            f"<@{bot.user.id}>",
                                            spec["question"],
                                            spec["options"],
                                            poll_msg.id,
                                        )
                                    except Exception as e:
                                        logger.debug(f"poll record failed (non-fatal): {e}")

                        # Record AFTER streaming completes so we have the real Discord ID.
                        # `replying_to_message_id` links Cirno's response to the
                        # message that triggered it — the prompt builder uses
                        # this to quote the target back to the model.
                        bot_message = {
                            "user_id": f"<@{bot.user.id}>",
                            "content": final_text,
                            "replying_to": f"<@{msg.author.id}>",
                            "role": "assistant",
                            "discord_message_id": str(sent_msg.id) if sent_msg else None,
                            "replying_to_message_id": str(msg.id) if reply_to_msg else None,
                        }
                        await record_chat_history(channel_id, bot_message)
                        last_response_time[channel_id] = time.time()

                        # Background save for any images on the triggering message.
                        # Only fires when Cirno was actually pinged (which is the
                        # condition for this code path), so the saved set is
                        # biased toward "interesting" images.
                        if (ENABLE_IMAGE_MEMORY and msg.attachments
                                and not unprompted and not msg.author.bot):
                            asyncio.create_task(_save_pinged_images(msg))
                    else:
                        logger.error(f"Failed to get a valid LLM response for channel {channel_id}")
                        # Don't spam an error reply — _stream_to_discord already
                        # edited the placeholder to "..." so Cirno just appears
                        # quiet rather than breaking character with an apology.

                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error processing message queue: {str(e)}", exc_info=True)
        finally:
            processing_channels.discard(channel_id)


async def health_check():
    while True:
        logger.info(f"{BOT_NAME} health check: Bot is running")
        await asyncio.sleep(300)


# --- Unprompted chime-in system ----------------------------------------------
# Tracks the last time Cirno chimed in unprompted per-channel, so we can enforce
# a cooldown across runs of the loop (within a single bot session).
_last_chime_at = {}  # {channel_id: timestamp}


async def _maybe_chime_in_channel(channel):
    """
    Decide whether to chime in unprompted in this channel. Several gates must
    all pass for a chime to actually fire — this is meant to be RARE.

    Gates (in order, all must pass):
      1. Random dice roll passes CHIME_PROBABILITY
      2. Cooldown: haven't chimed here in the last CHIME_COOLDOWN_SEC seconds
      3. There's been recent activity (within CHIME_RECENT_WINDOW_SEC)
      4. The most recent activity is older than CHIME_QUIET_AFTER_SEC
         (don't interrupt an active back-and-forth)
      5. At least CHIME_MIN_RECENT_MSGS messages have been sent since Cirno's
         last reply in this channel
    """
    if bot.user is None:
        return  # not connected yet
    cid = str(channel.id)
    now = time.time()

    # Gate 1: dice roll
    if random.random() >= CHIME_PROBABILITY:
        return

    # Gate 2: cooldown
    if cid in _last_chime_at and (now - _last_chime_at[cid]) < CHIME_COOLDOWN_SEC:
        return

    # Gate 3 & 4: peek recent history to find last activity & message count
    history = await get_conversation_history(cid, limit=20)
    if not history:
        return

    last_msg = history[-1]
    last_ts_dt = _parse_msg_timestamp(last_msg.get('timestamp'))
    if last_ts_dt is None:
        return
    from datetime import timezone as _tz
    age_sec = (datetime.now(_tz.utc) - last_ts_dt).total_seconds()

    if age_sec > CHIME_RECENT_WINDOW_SEC:
        # Channel is too quiet — chiming in cold would be weird.
        return
    if age_sec < CHIME_QUIET_AFTER_SEC:
        # Conversation is still active — let humans talk.
        return

    # Gate 5: messages since Cirno's last reply
    bot_mention = f"<@{bot.user.id}>"
    msgs_since_last_reply = 0
    for msg in reversed(history):
        if msg.get('user') == bot_mention or msg.get('role') == 'assistant':
            break
        msgs_since_last_reply += 1
    if msgs_since_last_reply < CHIME_MIN_RECENT_MSGS:
        return

    # All gates passed — synthesize a "fake user message" pointing at the
    # most recent message so the existing pipeline can produce a reply.
    # We don't want the bot to just monologue; she should react to context.
    try:
        # Fetch the actual Discord message object so the existing flow has
        # everything it needs (author, attachments, etc.).
        last_discord_id = last_msg.get('discord_message_id')
        anchor_msg = None
        if last_discord_id:
            try:
                anchor_msg = await channel.fetch_message(int(last_discord_id))
            except (discord.NotFound, discord.HTTPException, ValueError):
                pass
        if anchor_msg is None:
            # Fall back to scanning recent channel history directly
            async for m in channel.history(limit=10):
                if m.author != bot.user:
                    anchor_msg = m
                    break
        if anchor_msg is None:
            return

        # Mark this message so process_message_queue knows to handle it
        # despite none of the normal triggers firing.
        anchor_msg._cirno_unprompted = True

        _last_chime_at[cid] = now
        logger.info(f"Unprompted chime triggered in channel {cid} "
                    f"(quiet for {age_sec:.0f}s, {msgs_since_last_reply} msgs since last reply)")

        if cid not in message_queue:
            message_queue[cid] = []
            asyncio.create_task(process_message_queue(cid))
        message_queue[cid].append(anchor_msg)
    except Exception as e:
        logger.error(f"chime_in error in channel {cid}: {e}", exc_info=True)


async def chime_in_loop():
    """Periodically considers chiming in across all channels the bot can see."""
    if not ENABLE_UNPROMPTED_CHIME:
        return
    # Wait a bit after startup before considering any chime — we don't want
    # the first thing on a fresh restart to be an unprompted message.
    await asyncio.sleep(CHIME_CHECK_INTERVAL_SEC)
    while True:
        try:
            for guild in bot.guilds:
                for channel in guild.text_channels:
                    # Only consider channels Cirno can actually read AND send in
                    perms = channel.permissions_for(guild.me) if guild.me else None
                    if not perms or not perms.send_messages or not perms.read_messages:
                        continue
                    await _maybe_chime_in_channel(channel)
                    # Small breather between channels so we don't burst the
                    # history service on big servers.
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"chime_in_loop iteration failed: {e}", exc_info=True)
        await asyncio.sleep(CHIME_CHECK_INTERVAL_SEC)


# --- Personality summary system ----------------------------------------------
# Cirno builds a short note about each user she's spoken with, refreshed
# periodically. The note is generated with a NEUTRAL prompt (not Cirno's
# persona) to avoid a feedback loop where her own voice contaminates her
# observations of others.

# In-flight set so we don't summarize the same user twice in parallel
_summarizing_now = set()  # {(channel_id, user_id)}


SUMMARIZER_SYSTEM_PROMPT = (
    "You are NOT roleplaying as any character. Ignore any persona, name, or "
    "identity instructions you may have seen earlier. For this task only, you "
    "are a neutral analyst writing a concise character note about a Discord "
    "user based on their recent messages in one channel.\n\n"
    "What to cover:\n"
    "- their general communication style and tone\n"
    "- recurring topics or interests\n"
    "- attitude toward others in the channel (friendly, hostile, teasing, indifferent, etc.)\n"
    "- noteworthy patterns (e.g. asks for help often, jokes constantly, lurks then replies in bursts)\n\n"
    "Strict rules:\n"
    "- Write in plain English, third person, 3 to 6 short sentences.\n"
    "- No headers, no bullet lists, no markdown.\n"
    "- Be factual and observational. Do not be flattering or judgmental.\n"
    "- Do not invent details that aren't visible in the messages.\n"
    "- Do not address the user directly (no \"you\").\n"
    "- Do not include the user's ID, @mention, or username in the summary.\n"
    "- Do not use first-person voice (\"I\", \"my\"). You are an analyst, not a participant.\n"
    "- Do not refer to a Touhou character, an ice fairy, or any roleplay identity. "
    "If you find yourself wanting to, stop and write a plain analyst note instead.\n"
)


async def _fetch_user_messages(channel_id, user_id, limit=SUMMARY_MAX_USER_MESSAGES):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/get_user_messages/{channel_id}/{user_id}",
                params={"limit": limit},
            ) as r:
                if r.status == 200:
                    return await r.json()
                logger.error(f"get_user_messages: {r.status}")
    except Exception as e:
        logger.error(f"get_user_messages failed: {e}")
    return []


async def _fetch_user_message_count(channel_id, user_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/count_user_messages/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return int(data.get("count", 0))
    except Exception as e:
        logger.error(f"count_user_messages failed: {e}")
    return 0


async def get_personality(channel_id, user_id):
    """Return stored personality dict for this user/channel, or None."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CHAT_HISTORY_SERVICE_URL}/get_personality/{channel_id}/{user_id}"
            ) as r:
                if r.status == 200:
                    text = await r.text()
                    # Service returns "null" (literal) when no row exists
                    if text.strip() == "null":
                        return None
                    return json.loads(text)
    except Exception as e:
        logger.error(f"get_personality failed: {e}")
    return None


async def set_personality(channel_id, user_id, summary):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHAT_HISTORY_SERVICE_URL}/set_personality",
                json={"channel_id": channel_id, "user_id": user_id, "summary": summary},
            ) as r:
                if r.status != 200:
                    logger.error(f"set_personality: {r.status}")
                    return False
                return True
    except Exception as e:
        logger.error(f"set_personality failed: {e}")
        return False


async def summarize_user(channel_id, user_id, force=False):
    """
    Generate (or refresh) a personality summary for one user in one channel.
    Skips if (a) we don't have enough new messages since the last summary
    (unless `force=True`) or (b) we're already summarizing this user.
    """
    if not ENABLE_PERSONALITY_SUMMARIES:
        return None

    key = (str(channel_id), str(user_id))
    if key in _summarizing_now:
        return None
    _summarizing_now.add(key)
    try:
        existing = await get_personality(channel_id, user_id)
        total_msgs = await _fetch_user_message_count(channel_id, user_id)
        if not force and existing:
            new_msgs = total_msgs - int(existing.get('messages_at_last_summary', 0) or 0)
            if new_msgs < SUMMARY_MIN_NEW_MESSAGES:
                return existing  # still fresh enough

        if total_msgs == 0:
            return None  # nothing to summarize

        msgs = await _fetch_user_messages(channel_id, user_id)
        if not msgs:
            return None

        # Build a compact transcript. Strip mentions for cleanliness.
        transcript_lines = []
        for m in msgs:
            ts = _relative_time(m.get('timestamp', ''))
            content = (m.get('content') or '').strip()
            if not content:
                continue
            content = _EMOJI_RE.sub(r':\2:', content)  # emoji syntax → :name:
            transcript_lines.append(f"({ts}) {content}")
        if not transcript_lines:
            return None
        transcript = "\n".join(transcript_lines[-SUMMARY_MAX_USER_MESSAGES:])

        # IMPORTANT: this prompt does NOT use Cirno's persona — we want a
        # plain analyst voice so the summary is observation, not roleplay.
        # We also call get_llm_response with explicit overrides:
        #   - model:  MODEL_ID_SUMMARIZER (defaults to the chat model, but you
        #             can point it at a separate plain instruct model in Ollama
        #             via the MODEL_ID_SUMMARIZER_<BOT_NAME> env var)
        #   - thinking off: summaries are one-shot, not a chat turn
        prompt = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user",
             "content": (f"Recent messages from this user in this channel:\n\n{transcript}\n\n"
                         f"Write the character note now. Remember: plain analyst voice, "
                         f"third person, no roleplay.")},
        ]

        summary_text = await get_llm_response(
            prompt,
            retry=False,
            model=MODEL_ID_SUMMARIZER,
            enable_thinking=False,
            num_predict=SUMMARY_MAX_TOKENS,
            temperature=0.4,  # lower temp = more consistent observational output
        )

        if not summary_text or not summary_text.strip():
            logger.warning(f"summarize_user: empty summary for {user_id} in {channel_id}")
            return None

        summary_text = summary_text.strip()
        # Hard cap length so a runaway model doesn't blow the storage.
        if len(summary_text) > 2000:
            summary_text = summary_text[:1997].rstrip() + "..."

        ok = await set_personality(channel_id, user_id, summary_text)
        if ok:
            logger.info(f"Updated personality summary for {user_id} in {channel_id} "
                        f"({len(summary_text)} chars, {total_msgs} total msgs)")
            return {"summary": summary_text, "messages_at_last_summary": total_msgs}
        return None
    finally:
        _summarizing_now.discard(key)


async def maybe_trigger_user_summary(channel_id, user_id):
    """
    Fire-and-forget summarizer trigger. Called from on_message after each user
    message; will only actually run if there are enough new messages.
    Runs in a background task so it never blocks the user-facing reply.

    Wraps summarize_user in a logged exception handler — without this wrapper,
    any exception in the background task disappears silently (asyncio.create_task
    swallows exceptions unless you await the resulting Task or attach a done
    callback). That silent-failure mode is a common reason summaries appear
    to "never run" when they're actually crashing every time.
    """
    if not ENABLE_PERSONALITY_SUMMARIES:
        return

    async def _safe_summarize():
        try:
            await summarize_user(channel_id, user_id)
        except Exception as e:
            logger.error(f"summarize_user task crashed for {user_id} in "
                         f"{channel_id}: {e}", exc_info=True)

    asyncio.create_task(_safe_summarize())


# --- Image captioning and save flow -----------------------------------------

CAPTIONER_SYSTEM_PROMPT = (
    "You are an image captioner. Look at the image and write a single concise "
    "factual caption (one or two sentences) describing what is visibly in it. "
    "If the image is a meme, screenshot, or has readable text, include the "
    "key text or punchline. Use plain English. Do not use first person, do not "
    "speak as a character, do not address anyone. Just describe what's there."
)


async def _caption_image(image_b64, media_type):
    """
    Run the vision model with a neutral prompt to produce a caption.
    Returns the caption string or None on failure.

    Uses the chat model (Gemma) since we agreed to keep things simple — same
    model handles chat and captioning. Streaming is on under the hood (per
    get_llm_response) so this is Cloudflare-safe even if the model is cold.
    """
    prompt = [
        {"role": "system", "content": CAPTIONER_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text",
             "text": "Caption this image. Plain factual description, one or two sentences."},
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{image_b64}"}}
        ]}
    ]
    text = await get_llm_response(
        prompt,
        retry=False,
        enable_thinking=False,
        num_predict=IMAGE_CAPTION_MAX_TOKENS,
        temperature=0.3,  # captions should be stable, not creative
    )
    if not text:
        return None
    text = text.strip()
    # Clip in case the model went long despite num_predict
    if len(text) > 500:
        text = text[:497].rstrip() + "..."
    return text


def _ext_from_media_type(mt):
    """Map a Content-Type to a file extension. Falls back to png."""
    if not mt:
        return "png"
    mt = mt.lower().split(";")[0].strip()
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
    }.get(mt, "png")


async def _save_pinged_images(message):
    """
    Background task: caption and save every image attachment on this message
    to the bot's image memory. Called from process_message_queue ONLY when
    Cirno was actually pinged on the message (not on every image upload),
    so saved images are biased toward "interesting".

    This function runs after the user-facing reply has been sent — it's
    pure background work and never blocks chat.
    """
    if not ENABLE_IMAGE_MEMORY:
        return
    if not message.attachments:
        return
    # Opt-out gate — same as record_user_message. The history service is the
    # source of truth, but checking the cache locally avoids spending the
    # captioning model time on bytes we'd just throw away.
    if is_user_opted_out_local(str(message.channel.id), str(message.author.id)):
        return

    saved_count = 0
    for attachment in message.attachments:
        ct = attachment.content_type or ""
        if not ct.startswith("image/"):
            continue
        if attachment.size and attachment.size > IMAGE_MAX_FILE_SIZE_BYTES:
            logger.debug(f"skip save: attachment {attachment.filename} too large "
                         f"({attachment.size} > {IMAGE_MAX_FILE_SIZE_BYTES})")
            continue

        try:
            raw = await attachment.read()
        except discord.HTTPException as e:
            logger.debug(f"skip save: could not read {attachment.filename}: {e}")
            continue

        b64 = base64.b64encode(raw).decode("utf-8")

        # Caption first — if this fails we don't bother saving (a captionless
        # image is unretrievable, no point storing it).
        caption = await _caption_image(b64, ct)
        if not caption:
            logger.debug(f"skip save: captioner returned nothing for {attachment.filename}")
            continue

        new_id = await save_image_to_memory(
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            discord_message_id=str(message.id),
            media_type=ct,
            extension=_ext_from_media_type(ct),
            image_b64=b64,
            caption=caption,
        )
        if new_id is not None:
            saved_count += 1
            logger.info(f"image-memory: saved id={new_id} from {message.author} "
                        f"in {message.channel.id}: {caption[:80]}")

    if saved_count:
        logger.debug(f"_save_pinged_images: saved {saved_count} image(s)")


def _extract_recall_image_ids(text, allowed_ids):
    """
    Find [recall_image: N] tokens in `text`, validate each N against
    `allowed_ids` (the set of ids we offered to the model this turn), and
    return (cleaned_text, list_of_valid_ids_in_order).

    We strip the marker from the text in all cases (valid or not) so the
    user never sees `[recall_image: 47]` in the bot's output.

    `allowed_ids` is enforced to prevent the model from hallucinating ids
    that don't exist or don't belong to this channel.
    """
    if not text:
        return text, []
    valid_in_order = []
    invalid_seen = []

    def _replace(m):
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            return ""
        if n in allowed_ids and n not in valid_in_order:
            valid_in_order.append(n)
        else:
            invalid_seen.append(n)
        return ""  # always strip from visible text

    cleaned = RECALL_IMAGE_RE.sub(_replace, text)
    # Tidy whitespace left by the strip
    cleaned = _re_module.sub(r'[ \t]+', ' ', cleaned)
    cleaned = _re_module.sub(r'\n{3,}', '\n\n', cleaned).strip()

    if invalid_seen:
        logger.debug(f"_extract_recall_image_ids: dropped invalid ids "
                     f"{invalid_seen} (allowed: {sorted(allowed_ids)})")

    return cleaned, valid_in_order


# --- Poll creation + recording ----------------------------------------------

POLL_TOOL_PRIMER_NOTE = (
    "[Tool available — use SPARINGLY, only when actually useful: you can create "
    "a Discord poll by including this exact marker anywhere in your reply: "
    "[create_poll: question=\"...\" options=\"A|B|C\" duration_h=24 multi=false] "
    "Rules: 2-10 options, options pipe-separated, each option ≤55 chars, "
    "question ≤300 chars, duration_h is hours (1 to 768). multi=true allows "
    "voting for multiple options. The marker is stripped from your reply, so "
    "write naturally around it. Use ONLY when there is a real question to vote "
    "on (e.g. group decisions, picking a game, settling a debate) — do NOT use "
    "as an icebreaker, do NOT use to be cute, and never make a poll about a "
    "single user. If unsure, don't use it.]"
)


def _maybe_poll_primer_for_channel(channel_id_str):
    """
    Returns the priming note string if this turn should include it, else None.
    Increments the per-channel counter. Triggers every POLL_PRIME_EVERY_N
    user messages (1st, 21st, 41st, ...). Counter is in-memory only.
    """
    if not ENABLE_POLLS:
        return None
    cur = _poll_prime_counter.get(channel_id_str, 0)
    _poll_prime_counter[channel_id_str] = cur + 1
    # Prime when count is 0 (first ever in this session) or at every Nth.
    # `cur` is the COUNT BEFORE incrementing — using cur means we prime on
    # turns 0, N, 2N, ... which gives one priming per Nth turn.
    if cur % POLL_PRIME_EVERY_N == 0:
        return POLL_TOOL_PRIMER_NOTE
    return None


async def _create_polls_from_specs(channel, specs):
    """
    Given a list of poll specs (parsed from [create_poll: ...] markers in the
    bot's reply), create native Discord polls. Returns a list of (poll_message,
    spec) tuples for the polls successfully sent.

    We send each poll as its own message rather than trying to attach to the
    placeholder, because Discord doesn't let us add a poll to an existing
    message via edit() — polls must be set on the initial send.
    """
    if not specs:
        return []
    sent = []
    for spec in specs:
        try:
            # discord.Poll signature varies slightly across forks/versions:
            #   - Pycord 2.6+: Poll(question, *, duration=24 (int hours),
            #                       allow_multiselect=False)
            #   - discord.py 2.4+: similar; duration accepts int hours OR a
            #     timedelta in 2.5+. We pass int hours to be portable.
            poll = discord.Poll(
                question=spec["question"],
                duration=spec["duration_h"],  # integer hours
                allow_multiselect=spec["multi"],
            )
            for opt in spec["options"]:
                poll.add_answer(text=opt)
            poll_msg = await channel.send(poll=poll)
            sent.append((poll_msg, spec))
            logger.info(f"poll: created in {channel.id}: q={spec['question']!r} "
                        f"opts={spec['options']} duration_h={spec['duration_h']} "
                        f"multi={spec['multi']}")
        except (AttributeError, TypeError) as e:
            # Likely discord.py too old, or kwarg names differ on this fork.
            # Try the older "multiple" kwarg + timedelta as a fallback.
            try:
                poll = discord.Poll(
                    question=spec["question"],
                    duration=datetime_timedelta(hours=spec["duration_h"]),
                    multiple=spec["multi"],
                )
                for opt in spec["options"]:
                    poll.add_answer(text=opt)
                poll_msg = await channel.send(poll=poll)
                sent.append((poll_msg, spec))
                logger.info(f"poll: created (fallback signature) in {channel.id}: "
                            f"q={spec['question']!r}")
            except Exception as e2:
                logger.error(f"poll: discord.py version doesn't support polls "
                             f"(or API mismatch): primary={e}, fallback={e2}")
                return sent
        except discord.HTTPException as e:
            # Discord rejected — bot may lack send_polls permission, or the
            # channel doesn't allow polls (threads + announcement channels
            # have varying support). Log and skip.
            logger.warning(f"poll: send failed in {channel.id}: {e}")
        except Exception as e:
            logger.error(f"poll: unexpected error: {e}", exc_info=True)
    return sent


async def _record_poll_creation(channel_id, author_user_mention,
                                question, options, discord_message_id):
    """Add a synthetic chat-history entry describing a created poll, so RAG
    can retrieve 'what did we vote about cookies' style queries later. Stored
    as a regular content row that gets embedded by the history service."""
    options_str = ", ".join(options)
    summary_text = f"[poll started] «{question}» options: {options_str}"
    # role for human-created polls = "human" so it shows in regular history;
    # for Cirno-created polls we tag as "assistant" (caller decides).
    # Both go through normal record_chat_history.
    return await record_chat_history(channel_id, {
        "user_id": author_user_mention,
        "content": summary_text,
        "replying_to": None,
        "role": "human",
        "discord_message_id": str(discord_message_id) if discord_message_id else None,
    })


async def _record_poll_vote(channel_id, voter_mention, voter_id,
                            question, option_text, poll_message_id):
    """Record a single poll vote as a low-priority chat-history row. Stored
    as a 'human' role message so it gets embedded normally. Skip if disabled
    or the embedding service is overloaded — votes are nice-to-have data."""
    summary_text = f"[poll vote] voted for «{option_text}» on poll «{question}»"
    return await record_chat_history(channel_id, {
        "user_id": voter_mention,
        "content": summary_text,
        "replying_to": None,
        "role": "human",
        "discord_message_id": None,  # votes don't have their own message ids
    })


@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    bot.loop.create_task(health_check())
    if ENABLE_UNPROMPTED_CHIME:
        bot.loop.create_task(chime_in_loop())

    # Slash command sync. We do this inside on_ready (rather than on the
    # bot's startup hook) so it runs once we know who we are and have the
    # guild list available. Failures are logged but not fatal — the bot can
    # still operate; only the slash UI will be missing or stale.
    try:
        if GUILD_IDS_FOR_SLASH_SYNC:
            # Per-guild sync: instant propagation, scoped to the listed guilds.
            # Best for development where you want command changes to appear
            # in the slash menu immediately.
            total = 0
            for gid in GUILD_IDS_FOR_SLASH_SYNC:
                guild_obj = discord.Object(id=gid)
                # Copy the global tree to this guild and sync — this avoids
                # needing decorators that pre-bind to a specific guild.
                bot.tree.copy_global_to(guild=guild_obj)
                synced = await bot.tree.sync(guild=guild_obj)
                total += len(synced)
                logger.info(f"slash: synced {len(synced)} commands to guild {gid}")
            logger.info(f"slash: total {total} command(s) synced across "
                        f"{len(GUILD_IDS_FOR_SLASH_SYNC)} guild(s)")
        else:
            # Global sync: works everywhere the bot is, but takes up to 1h
            # to propagate the first time. Subsequent syncs are usually fast.
            synced = await bot.tree.sync()
            logger.info(f"slash: synced {len(synced)} command(s) globally "
                        f"(may take up to 1h to appear)")
    except Exception as e:
        logger.error(f"slash sync failed: {e}", exc_info=True)


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Track this user as a participant in this channel so the bot can
    # mention them by name later if the LLM references them.
    _register_participant(message.channel.id, message.author)
    # Also register anyone mentioned in the message — they're part of the context.
    for mentioned in message.mentions:
        _register_participant(message.channel.id, mentioned)

    await record_user_message(message)

    # Fire off a background check to see if it's time to refresh this user's
    # personality summary. Runs only if enough new messages have accumulated;
    # doesn't block the reply.
    if not message.author.bot:
        await maybe_trigger_user_summary(str(message.channel.id), str(message.author.id))

    # Reasons to process a message:
    #  - Direct ping (mention or alias by name)
    #  - Reply to one of Cirno's messages
    #  - Another known bot speaking (so Cirno can respond to them)
    # Removed (intentional, per user request):
    #  - attachments / images: vision still works when the message also pings
    #    her, but a bare image upload no longer auto-triggers a reply.
    #  - mostly-emoji messages: same reasoning.
    #  - 10% random reply: the new "rare unprompted" system below replaces this.
    should_process = (
        bot.user.mentioned_in(message)
        or _message_mentions_bot_by_name(message.content)
        or message.author.bot
        or (message.reference and message.reference.resolved
            and message.reference.resolved.author == bot.user)
    )

    if should_process:
        channel_id = str(message.channel.id)
        if channel_id not in message_queue:
            message_queue[channel_id] = []
            asyncio.create_task(process_message_queue(channel_id))
        message_queue[channel_id].append(message)

    await bot.process_commands(message)


@bot.event
async def on_poll_vote_add(user, answer):
    """
    Fired when a user votes on a poll. We log the vote into chat history
    (under the voter's mention, role=human) so RAG can later answer queries
    like "did anyone vote for X" or "what did we decide about Y".

    This also fires for votes on polls that humans created via the Discord
    UI, not just Cirno-created polls — gives us a complete record either way.
    """
    if not ENABLE_POLLS:
        return
    if user.bot:
        return  # bots can't actually vote, but defensive

    try:
        # Each PollAnswer has either `.text` directly or `.media.text`.
        opt_text = getattr(answer, "text", None)
        if opt_text is None and hasattr(answer, "media"):
            opt_text = getattr(answer.media, "text", None)
        if not opt_text:
            return  # can't usefully record without an option label

        # Question text from the parent poll
        poll = getattr(answer, "poll", None) or getattr(answer, "_poll", None)
        question_text = ""
        channel_id = None
        if poll is not None:
            q = getattr(poll, "question", None)
            question_text = q if isinstance(q, str) else (getattr(q, "text", "") or "")
            poll_msg = getattr(poll, "message", None)
            if poll_msg is not None:
                channel_id = poll_msg.channel.id

        if channel_id is None:
            # Without a channel we can't store the vote — skip
            return

        await _record_poll_vote(
            str(channel_id),
            f"<@{user.id}>",
            user.id,
            question_text or "(unknown poll)",
            opt_text,
            poll_msg.id if poll_msg else None,
        )
    except Exception as e:
        logger.debug(f"on_poll_vote_add failed (non-fatal): {e}")


async def record_user_message(message):
    channel_id = str(message.channel.id)

    # Opt-out gate — if this user opted out of this channel, drop the write
    # before any work. The history service also enforces this server-side
    # (defense in depth), but the local cache lets us skip the HTTP round-trip
    # entirely once an opt-out has been recorded in this bot session.
    if is_user_opted_out_local(channel_id, str(message.author.id)):
        return

    # Tag other bots distinctly so RAG can filter them out cleanly.
    # human  = real user
    # bot    = another LLM bot in the channel (HERMES/NEMO/etc)
    # (Cirno's own messages are tagged "assistant" elsewhere.)
    role = "bot" if message.author.bot else "human"

    content_to_save = message.content
    if message.attachments:
        content_to_save += " [User attached an image]"
    # Stickers — annotate with name (and description if provided) so RAG can
    # match later "remember when X sent that <sticker>" queries.
    if getattr(message, "stickers", None):
        for s in message.stickers:
            name = getattr(s, "name", "") or "sticker"
            desc = (getattr(s, "description", "") or "").strip()
            if desc:
                content_to_save += f" [sticker: {name} — {desc}]"
            else:
                content_to_save += f" [sticker: {name}]"
    # Native Discord polls — when a human creates one via Discord's UI, the
    # message has a `poll` attribute. Annotate with question + options so RAG
    # can recall later "what did we vote about cookies" queries.
    poll = getattr(message, "poll", None)
    if poll is not None:
        try:
            q_text = ""
            if hasattr(poll, "question"):
                q = poll.question
                q_text = q if isinstance(q, str) else (getattr(q, "text", "") or "")
            opts = []
            for ans in (getattr(poll, "answers", None) or []):
                t = getattr(ans, "text", None)
                if t is None and hasattr(ans, "media"):
                    t = getattr(ans.media, "text", None)
                if t:
                    opts.append(t)
            if q_text and opts:
                content_to_save += f" [poll started] «{q_text}» options: {', '.join(opts)}"
        except Exception as e:
            logger.debug(f"could not extract poll info: {e}")

    # Extract reply target info. We capture both the author's mention (legacy
    # `replying_to` field) and the target message's snowflake (`replying_to_message_id`).
    # The latter is the key to the new prompt design — it lets the bot look up
    # the actual content of what's being replied to and quote it back to the LLM.
    replying_to_user = None
    replying_to_msg_id = None
    if message.reference and message.reference.resolved:
        target = message.reference.resolved
        target_author = getattr(target, 'author', None)
        if target_author is not None:
            replying_to_user = f"<@{target_author.id}>"
        # ref.message_id is the snowflake even when ref.resolved is None,
        # but we use the resolved object when available for consistency.
        replying_to_msg_id = str(getattr(target, 'id', '') or message.reference.message_id or '')
        if not replying_to_msg_id:
            replying_to_msg_id = None
    elif message.reference and message.reference.message_id:
        # ref.resolved is None (message was deleted or wasn't fetched), but the
        # reference itself still carries the target's id.
        replying_to_msg_id = str(message.reference.message_id)

    user_message = {
        "user_id": f"<@{message.author.id}>",
        "content": content_to_save,
        "replying_to": replying_to_user,
        "role": role,
        "discord_message_id": str(message.id),
        "replying_to_message_id": replying_to_msg_id,
    }
    await record_chat_history(channel_id, user_message)


# ============================================================================
# Slash commands
# ============================================================================
# All commands are slash-only — the previous text-based `?cmd` triggers no
# longer exist. Slash uses Discord's interaction model:
#   - First response within 3 seconds, or call interaction.response.defer()
#   - Ephemeral responses (visible only to the invoker) are the right choice
#     for these admin/diagnostic commands so they don't clutter the channel
#   - There's no "trigger message" to delete — interactions are inherently
#     non-cluttering when ephemeral
#
# Sync happens in on_ready(); see GUILD_IDS_FOR_SLASH_SYNC for dev vs prod.
# ============================================================================


@bot.tree.command(name='ctxbreak',
                  description='Hard-reset Cirno\'s conversation context in this channel')
async def ctxbreak_slash(interaction: discord.Interaction):
    """Slash version of ctxbreak. Sets the in-memory cutoff so previous
    messages are hidden from sliding-window history and RAG. The history
    service still has the rows; they're just filtered out from this point on."""
    from datetime import timezone
    channel_id = str(interaction.channel_id)

    _ctx_cutoff[channel_id] = datetime.now(timezone.utc)
    message_queue.pop(channel_id, None)
    last_response_time.pop(channel_id, None)
    _channel_participants.pop(channel_id, None)
    _rewritten_content_cache.clear()

    # Sentinel marker (informational; not what enforces the break)
    await record_chat_history(channel_id, {
        "user_id": f"<@{bot.user.id}>",
        "content": "[CONTEXT RESET — treat this as the start of a new conversation. "
                   "Forget prior exchanges above this line.]",
        "replying_to": None,
        "role": "assistant"
    })

    await interaction.response.send_message(
        "context cleared for this channel. older messages are hidden from "
        "Cirno until `/ctxrestore`.",
        ephemeral=True,
    )
    logger.info(f"/ctxbreak in channel {channel_id} by {interaction.user} "
                f"(cutoff={_ctx_cutoff[channel_id].isoformat()})")


@bot.tree.command(name='ctxrestore',
                  description='Undo the most recent /ctxbreak — older messages become visible again')
async def ctxrestore_slash(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    removed = _ctx_cutoff.pop(channel_id, None)
    if removed:
        await interaction.response.send_message(
            "context restored. older messages are visible to Cirno again.",
            ephemeral=True,
        )
        logger.info(f"/ctxrestore in channel {channel_id} by {interaction.user} "
                    f"(was cutoff={removed.isoformat()})")
    else:
        await interaction.response.send_message(
            "no context break was active in this channel.",
            ephemeral=True,
        )


@bot.tree.command(name='personality',
                  description='Show what Cirno remembers about a user')
@app_commands.describe(target='The user to look up. Defaults to yourself.')
async def personality_slash(interaction: discord.Interaction,
                            target: discord.Member = None):
    """Slash version. Result is ephemeral so the channel stays clean."""
    target = target or interaction.user
    channel_id = str(interaction.channel_id)
    user_id = str(target.id)

    record = await get_personality(channel_id, user_id)
    if not record or not (record.get("summary") or "").strip():
        await interaction.response.send_message(
            f"no personality stored for {target.display_name} yet — need at "
            f"least {SUMMARY_MIN_NEW_MESSAGES} of their messages to build "
            f"one. try `/reflect target:{target.display_name}` to force one.",
            ephemeral=True,
        )
        logger.info(f"/personality by {interaction.user} for {target} -> none stored")
        return

    summary = record["summary"].strip()

    # Redact any user mentions / IDs that snuck into the summary.
    # The summarizer prompt forbids them, but Gemma sometimes leaks them.
    summary = _re_module.sub(r'<@!?\d+>', '[user]', summary)
    summary = _re_module.sub(r'\b\d{17,20}\b', '[id]', summary)

    last_summarized = record.get("last_summarized_at") or ""
    msg_count_at = record.get("messages_at_last_summary") or 0
    when = _relative_time(last_summarized) if last_summarized else "unknown"

    body = (f"**personality note for {target.display_name}** "
            f"_(last updated {when}, {msg_count_at} messages observed)_\n\n"
            f"{summary}")

    if len(body) > 1900:
        body = body[:1897].rstrip() + "..."

    await interaction.response.send_message(body, ephemeral=True)
    logger.info(f"/personality by {interaction.user} for {target} -> {len(summary)} chars")


@bot.tree.command(name='reflect',
                  description='Force a fresh personality summary for a user')
@app_commands.describe(target='The user to summarize. Defaults to yourself.')
async def reflect_slash(interaction: discord.Interaction,
                        target: discord.Member = None):
    target = target or interaction.user
    channel_id = str(interaction.channel_id)
    user_id = str(target.id)

    # Defer because the LLM call will take longer than 3s. Ephemeral so the
    # channel doesn't see an empty "Cirno is thinking" placeholder.
    await interaction.response.defer(ephemeral=True, thinking=True)

    result = await summarize_user(channel_id, user_id, force=True)
    if result and result.get('summary'):
        await interaction.followup.send(
            f"updated personality for {target.display_name} "
            f"({len(result['summary'])} chars). use `/personality target:{target.display_name}` to see it.",
            ephemeral=True,
        )
        logger.info(f"/reflect by {interaction.user} for {target} -> "
                    f"{len(result['summary'])} chars")
    else:
        await interaction.followup.send(
            f"could not summarize {target.display_name} — likely no messages "
            f"recorded yet, or the LLM produced nothing. run `/diagnose` to investigate.",
            ephemeral=True,
        )
        logger.info(f"/reflect by {interaction.user} for {target} -> no summary produced")


@bot.tree.command(name='forget',
                  description='Erase Cirno\'s stored personality summary for a user')
@app_commands.describe(target='Whose summary to erase. Defaults to yourself.')
async def forget_slash(interaction: discord.Interaction,
                       target: discord.Member = None):
    target = target or interaction.user
    channel_id = str(interaction.channel_id)
    user_id = str(target.id)

    ok = await set_personality(channel_id, user_id, "")
    await interaction.response.send_message(
        f"forgot personality for {target.display_name}." if ok
        else f"could not forget personality for {target.display_name}.",
        ephemeral=True,
    )
    logger.info(f"/forget by {interaction.user} for {target} -> {'ok' if ok else 'failed'}")


@bot.tree.command(name='forget_images',
                  description='Delete saved images uploaded by a user in this channel')
@app_commands.describe(target='Whose saved images to delete. Defaults to yourself.')
async def forget_images_slash(interaction: discord.Interaction,
                              target: discord.Member = None):
    target = target or interaction.user
    channel_id = str(interaction.channel_id)
    user_id = str(target.id)

    deleted = await delete_user_images(channel_id, user_id)
    if deleted > 0:
        msg = (f"forgot {deleted} image{'s' if deleted != 1 else ''} "
               f"from {target.display_name}.")
    else:
        msg = f"no saved images for {target.display_name}."
    await interaction.response.send_message(msg, ephemeral=True)
    logger.info(f"/forget_images by {interaction.user} for {target} -> deleted {deleted}")


# Pending opt-out confirmations: maps (channel_id, user_id) -> timestamp of
# the first /optout call. A second call within OPTOUT_CONFIRM_WINDOW_SEC is
# treated as the confirmation. After that window, a fresh first call is
# required again. In-memory only; bot restart wipes pending confirmations,
# which is the safe default (no destructive action without recent intent).
_pending_optout = {}  # {(channel_id, user_id): float (monotonic)}
OPTOUT_CONFIRM_WINDOW_SEC = 60


@bot.tree.command(name='optout',
                  description='Opt out of Cirno data collection in this channel — DESTRUCTIVE')
async def optout_slash(interaction: discord.Interaction):
    """
    Two-step destructive command. Per-channel only.
      1st call:  shows what would be deleted, primes confirmation
      2nd call:  if within 60s, actually deletes everything

    What gets deleted:
      - All your messages stored in this channel's history
      - All saved images you uploaded in this channel
      - Your personality summary in this channel
      - The replying_to field on other people's messages that referenced you
        (their messages stay; only the link to you is severed)

    After opt-out, future messages and images from you in this channel are
    silently dropped at the history-service boundary — Cirno effectively
    stops noticing you in this channel.

    Use `/optin` to undo (re-enables recording; deleted data stays gone).
    """
    channel_id = str(interaction.channel_id)
    user_id = str(interaction.user.id)
    key = (channel_id, user_id)

    # Pre-flight: how much would we delete? Also tells us if they're already opted out.
    summary = await fetch_optout_summary(channel_id, user_id)
    if summary is None:
        await interaction.response.send_message(
            "could not reach the history service. try again in a moment.",
            ephemeral=True,
        )
        return

    if summary.get("already_opted_out"):
        await interaction.response.send_message(
            "you're already opted out of this channel. use `/optin` to re-enable.",
            ephemeral=True,
        )
        return

    msgs = summary.get("messages", 0)
    imgs = summary.get("saved_images", 0)
    personalities = summary.get("personality_summaries", 0)

    now = time.monotonic()
    pending_at = _pending_optout.get(key)

    if pending_at is None or (now - pending_at) > OPTOUT_CONFIRM_WINDOW_SEC:
        # First call (or expired): show warning + count, prime confirmation
        _pending_optout[key] = now
        body = (
            "**⚠ this is destructive and per-channel.**\n"
            f"if you confirm, the following will be **permanently deleted** "
            f"from Cirno's memory in this channel:\n"
            f"• **{msgs}** message{'s' if msgs != 1 else ''}\n"
            f"• **{imgs}** saved image{'s' if imgs != 1 else ''}\n"
            f"• **{personalities}** personality summary\n"
            f"plus any future messages/images you send in this channel will be ignored "
            f"until you run `/optin`.\n\n"
            f"**run `/optout` again within 60s to confirm.**"
        )
        await interaction.response.send_message(body, ephemeral=True)
        logger.info(f"/optout (warn) by {interaction.user} in {channel_id}: "
                    f"msgs={msgs} imgs={imgs} personalities={personalities}")
        return

    # Second call within window — actually do it
    _pending_optout.pop(key, None)
    await interaction.response.defer(ephemeral=True, thinking=True)
    result = await perform_optout(channel_id, user_id)
    if result is None:
        await interaction.followup.send(
            "opt-out failed — could not reach the history service. nothing was deleted.",
            ephemeral=True,
        )
        return

    body = (
        "✓ opted out.\n"
        f"deleted: **{result.get('messages_deleted', 0)}** messages, "
        f"**{result.get('images_deleted', 0)}** images, "
        f"**{result.get('personalities_deleted', 0)}** personality summary.\n"
        f"unlinked **{result.get('replying_to_nulled', 0)}** reply references.\n"
        f"Cirno will ignore your messages in this channel until `/optin`."
    )
    await interaction.followup.send(body, ephemeral=True)
    logger.info(f"/optout (confirmed) by {interaction.user} in {channel_id}: "
                f"deleted msgs={result.get('messages_deleted', 0)} "
                f"imgs={result.get('images_deleted', 0)} "
                f"personalities={result.get('personalities_deleted', 0)}")


@bot.tree.command(name='optin',
                  description='Re-enable Cirno data collection in this channel after a previous /optout')
async def optin_slash(interaction: discord.Interaction):
    """
    Reverses an opt-out. Does NOT restore previously deleted data — that's
    gone forever. Future messages/images will be recorded normally.
    """
    channel_id = str(interaction.channel_id)
    user_id = str(interaction.user.id)

    result = await perform_optin(channel_id, user_id)
    if result is None:
        await interaction.response.send_message(
            "could not reach the history service. try again in a moment.",
            ephemeral=True,
        )
        return

    if result.get("was_opted_out"):
        await interaction.response.send_message(
            "✓ opted back in. Cirno will record your messages in this channel "
            "again. previously deleted data is gone.",
            ephemeral=True,
        )
        logger.info(f"/optin by {interaction.user} in {channel_id}")
    else:
        await interaction.response.send_message(
            "you weren't opted out of this channel — nothing to do.",
            ephemeral=True,
        )


@bot.tree.command(name='diagnose',
                  description='Walk the personality-summary pipeline and report what works / what fails')
@app_commands.describe(target='The user to test against. Defaults to yourself.')
async def diagnose_slash(interaction: discord.Interaction,
                         target: discord.Member = None):
    """
    Slash version of diagnose. Defers immediately because step 5 hits the
    LLM and that can take a while.
    """
    target = target or interaction.user
    channel_id = str(interaction.channel_id)
    user_id = str(target.id)

    await interaction.response.defer(ephemeral=True, thinking=True)

    lines = [f"diagnose for @{target.display_name} in this channel:", ""]

    # Step 1: feature flag
    lines.append(f"1. ENABLE_PERSONALITY_SUMMARIES = {ENABLE_PERSONALITY_SUMMARIES}")
    if not ENABLE_PERSONALITY_SUMMARIES:
        lines.append("   ✗ feature is disabled — set ENABLE_PERSONALITY_SUMMARIES = True")
        await _send_diag_slash(interaction, lines)
        return
    lines.append("   ✓ ok")

    # Step 2: count endpoint
    lines.append("")
    lines.append(f"2. GET /count_user_messages/{channel_id}/{user_id}")
    try:
        total = await _fetch_user_message_count(channel_id, user_id)
        lines.append(f"   returned: {total}")
        if total == 0:
            lines.append("   ✗ count is 0 — either the history service isn't recording")
            lines.append("     this user's messages, or the user has opted out of this")
            lines.append("     channel, or the endpoint doesn't exist on the running service.")
            await _send_diag_slash(interaction, lines)
            return
        lines.append("   ✓ ok")
    except Exception as e:
        lines.append(f"   ✗ exception: {e}")
        await _send_diag_slash(interaction, lines)
        return

    # Step 3: messages endpoint
    lines.append("")
    lines.append(f"3. GET /get_user_messages/{channel_id}/{user_id}")
    try:
        msgs = await _fetch_user_messages(channel_id, user_id)
        lines.append(f"   returned: {len(msgs)} messages")
        if not msgs:
            lines.append("   ✗ no messages returned")
            await _send_diag_slash(interaction, lines)
            return
        sample = (msgs[-1].get('content') or '')[:60]
        lines.append(f"   sample (most recent): {sample!r}")
        lines.append("   ✓ ok")
    except Exception as e:
        lines.append(f"   ✗ exception: {e}")
        await _send_diag_slash(interaction, lines)
        return

    # Step 4: existing summary
    lines.append("")
    lines.append(f"4. GET /get_personality/{channel_id}/{user_id}")
    try:
        existing = await get_personality(channel_id, user_id)
        if existing is None:
            lines.append("   no row stored yet (this is normal for first run)")
        else:
            stored_count = existing.get('messages_at_last_summary', 0)
            summary_len = len(existing.get('summary') or '')
            lines.append(f"   row exists: messages_at_last_summary={stored_count}, "
                         f"summary length={summary_len}")
            if summary_len == 0:
                lines.append("   ⚠ row exists but summary is empty (was forgotten or LLM produced nothing)")
        lines.append("   ✓ ok (endpoint reachable)")
    except Exception as e:
        lines.append(f"   ✗ exception: {e}")
        await _send_diag_slash(interaction, lines)
        return

    # Step 5: LLM call
    lines.append("")
    lines.append(f"5. LLM summarizer call (model={MODEL_ID_SUMMARIZER})")
    try:
        transcript_lines = []
        for m in msgs:
            ts = _relative_time(m.get('timestamp', ''))
            content = (m.get('content') or '').strip()
            if not content:
                continue
            content = _EMOJI_RE.sub(r':\2:', content)
            transcript_lines.append(f"({ts}) {content}")
        if not transcript_lines:
            lines.append("   ✗ no non-empty messages to feed to the LLM")
            await _send_diag_slash(interaction, lines)
            return
        transcript = "\n".join(transcript_lines[-SUMMARY_MAX_USER_MESSAGES:])
        prompt = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user",
             "content": (f"Recent messages from this user in this channel:\n\n{transcript}\n\n"
                         f"Write the character note now. Remember: plain analyst voice, "
                         f"third person, no roleplay.")},
        ]
        text = await get_llm_response(
            prompt,
            retry=False,
            model=MODEL_ID_SUMMARIZER,
            enable_thinking=False,
            num_predict=SUMMARY_MAX_TOKENS,
            temperature=0.4,
        )
        if not text or not text.strip():
            lines.append("   ✗ LLM returned empty/None — this is likely the actual failure")
            lines.append("     possible causes: model not loaded, OWUI auth, model errored")
            lines.append("     check the bot's MetaLLM.log for stack traces")
            await _send_diag_slash(interaction, lines)
            return
        text = text.strip()
        lines.append(f"   returned {len(text)} chars")
        lines.append(f"   preview: {text[:120]!r}{'...' if len(text) > 120 else ''}")
        lines.append("   ✓ ok")
    except Exception as e:
        lines.append(f"   ✗ exception: {e}")
        await _send_diag_slash(interaction, lines)
        return

    # Step 6: set_personality
    lines.append("")
    lines.append(f"6. POST /set_personality")
    try:
        ok = await set_personality(channel_id, user_id, text)
        if not ok:
            lines.append("   ✗ /set_personality returned non-200 — check chat_history_service logs")
            await _send_diag_slash(interaction, lines)
            return
        lines.append("   ✓ ok")
    except Exception as e:
        lines.append(f"   ✗ exception: {e}")
        await _send_diag_slash(interaction, lines)
        return

    lines.append("")
    lines.append("ALL STEPS PASSED — summary is now stored.")
    lines.append(f"run `/personality target:{target.display_name}` to see it.")
    await _send_diag_slash(interaction, lines)
    logger.info(f"/diagnose by {interaction.user} for {target}: all steps passed")


async def _send_diag_slash(interaction, lines):
    """Helper: post a code-block diagnostic, splitting if it exceeds 1900 chars.
    Always uses interaction.followup since we deferred at the top of /diagnose."""
    body = "```\n" + "\n".join(lines) + "\n```"
    if len(body) <= 1900:
        try:
            await interaction.followup.send(body, ephemeral=True)
        except discord.HTTPException as e:
            logger.warning(f"diagnose: send failed: {e}")
        return
    # Too long — chunk it
    chunks = []
    cur = []
    cur_len = 0
    for ln in lines:
        if cur_len + len(ln) + 1 > 1700:
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append(cur)
    for ch in chunks:
        try:
            await interaction.followup.send(
                "```\n" + "\n".join(ch) + "\n```", ephemeral=True
            )
        except discord.HTTPException as e:
            logger.warning(f"diagnose: chunk send failed: {e}")



if __name__ == "__main__":
    # Token must come from environment. Never hardcode — rotate the previous one if it was committed.
    token = os.getenv(f'DISCORD_TOKEN_{BOT_NAME}')
    if not token:
        logger.error(f"DISCORD_TOKEN_{BOT_NAME} not found in environment")
    else:
        bot.run(token)