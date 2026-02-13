// index.js
// FINAL BACKEND (CLEAN + ROBUST)
// - Audio upload
// - Whisper transcription
// - AI insights generation with schema validation
// - Graceful handling of empty transcripts
// - All required endpoints for Flutter
// - In-memory DB (dev-safe)

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const OpenAI = require("openai");
const { InsightsSchema } = require("./schemas/insights.schema");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const { z } = require("zod");

const app = express();
const PORT = process.env.PORT || 8080;

// --------------------
// App setup
// --------------------

app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --------------------
// In-memory DB
// --------------------

const DB = {
  sessions: {}, // id -> session
};

// --------------------
// Uploads
// --------------------

const UPLOAD_DIR = path.resolve("./uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) =>
    cb(null, `${Date.now()}-${file.originalname}`),
});

const upload = multer({ storage });

// --------------------
// ROUTES
// --------------------

// Health check
app.get("/", (_, res) => {
  res.send("Backend is running");
});

// Create session + upload audio
app.post("/api/sessions", upload.single("audio"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file received" });
  }

  const id = `sess-${Date.now()}`;

  const session = {
    id,
    eventId: req.body.eventId || null,
    title: req.body.title || "Capture",

    audioPath: path.resolve(req.file.path),
    audioFilename: req.file.filename,

    transcript: "",
    insights: null,

    status: "processing", // processing | ready | failed
    progress: 0,
    error: null,

    createdAt: new Date().toISOString(),
    processedAt: null,
  };

  DB.sessions[id] = session;

  processSession(id).catch((err) => {
    console.error("Processing failed:", err.message);
    session.status = "failed";
    session.progress = 0;
    session.error = err.message || "Unknown error";
  });

  res.status(201).json({
    id: session.id,
    status: session.status,
    eventId: session.eventId,
    title: session.title,
  });
});

// List all sessions
app.get("/api/sessions", (req, res) => {
  const sessions = Object.values(DB.sessions);

  sessions.sort((a, b) => {
    const da = new Date(a.createdAt).getTime();
    const db = new Date(b.createdAt).getTime();
    return db - da;
  });

  res.json(sessions);
});

// Sessions count (Explore screen)
app.get("/api/sessions/count", (req, res) => {
  res.json({ count: Object.keys(DB.sessions).length });
});

// Session status only
app.get("/api/sessions/:id/status", (req, res) => {
  const s = DB.sessions[req.params.id];
  if (!s) return res.status(404).json({ error: "Not found" });

  res.json({
    status: s.status,
    progress: s.progress,
    error: s.error,
  });
});

// Reprocess a session
app.post("/api/sessions/:id/reprocess", (req, res) => {
  const s = DB.sessions[req.params.id];
  if (!s) return res.status(404).json({ error: "Not found" });

  s.status = "processing";
  s.progress = 0;
  s.error = null;
  s.transcript = "";
  s.insights = null;
  s.processedAt = null;

  processSession(s.id).catch((err) => {
    console.error("Reprocess failed:", err.message);
    s.status = "failed";
    s.progress = 0;
    s.error = err.message || "Unknown error";
  });

  res.json({ id: s.id, status: s.status });
});

// Full session (detail screen)
app.get("/api/sessions/:id", (req, res) => {
  const s = DB.sessions[req.params.id];
  if (!s) return res.status(404).json({ error: "Not found" });
  res.json(s);
});

// Syntra chat (AI assistant)
app.post("/api/syntra/chat", async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Missing OPENAI_API_KEY" });
    }

    const message = String(req.body?.message || "").trim();
    const sessionIds = Array.isArray(req.body?.sessionIds)
      ? req.body.sessionIds
      : [];
    const stream = Boolean(req.body?.stream);

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const sessions = Object.values(DB.sessions)
      .filter((s) => (sessionIds.length ? sessionIds.includes(s.id) : true))
      .slice(0, 10)
      .map((s) => ({
        id: s.id,
        title: s.title,
        status: s.status,
        createdAt: s.createdAt,
        transcript: s.transcript,
        insights: s.insights,
      }));

    const systemPrompt = `
You are Syntra, a study copilot for students.
Use the provided session summaries, insights, and transcripts first.
If the answer is not in the sessions, say so and then answer briefly from general knowledge.
Be concise, accurate, and avoid speculation.
When referencing sessions, cite the session title or date.
Prefer bullet points for multi-part answers.
`;

    const payload = {
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: systemPrompt.trim() },
        {
          role: "user",
          content: JSON.stringify({
            message,
            sessions,
          }),
        },
      ],
      temperature: 0.3,
      stream,
    };

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders?.();

      const streamResp = await openai.chat.completions.create(payload);
      for await (const chunk of streamResp) {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        if (delta) {
          res.write(`data: ${delta}\n\n`);
        }
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const response = await openai.chat.completions.create(payload);
      const reply = response.choices[0]?.message?.content || "";
      res.json({ reply });
    }
  } catch (err) {
    console.error("Syntra chat error:", err.message);
    res.status(500).json({ error: "Syntra chat failed" });
  }
});

// Outline extraction (AI)
app.post("/api/outline/extract", upload.single("file"), async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Missing OPENAI_API_KEY" });
    }

    if (!req.file) {
      return res.status(400).json({ error: "No outline file received" });
    }

    const ext = path.extname(req.file.originalname || "").toLowerCase();
    const text = await extractTextFromFile(req.file.path, ext);
    if (!text || text.trim().length < 20) {
      return res.status(400).json({ error: "Outline text too short" });
    }

    const extracted = await extractOutlineFromText(text);
    return res.json(extracted);
  } catch (err) {
    console.error("Outline extract error:", err.message);
    return res.status(500).json({ error: "Outline extraction failed" });
  }
});

// --------------------
// PROCESSING PIPELINE
// --------------------

async function processSession(sessionId) {
  const session = DB.sessions[sessionId];
  if (!session) throw new Error("Session not found");

  // STEP 1: Whisper transcription
  session.progress = 10;
  session.status = "processing";

  const transcript = await transcribeAudioWithWhisper(session.audioPath);

  session.transcript = transcript;
  session.progress = 60;
  session.status = "transcribed";

  // GUARD: transcript too short or empty
  if (!transcript || transcript.trim().length < 20) {
    console.warn("Transcript too short. Skipping AI insight generation.");

    session.insights = {
      summary:
        "The recording did not contain enough audible content to generate insights.",
      keyConcepts: [],
      flashcards: [],
      actionItems: [
        "Re-record the session with clearer speech.",
        "Ensure the microphone is close to the speaker.",
        "Avoid long silences during recording.",
      ],
    };

    session.status = "ready";
    session.progress = 100;
    session.processedAt = new Date().toISOString();

    console.log(`Session ${sessionId} completed with empty transcript`);
    return;
  }

  // STEP 2: AI insights (async after transcript is available)
  const insights = await generateInsightsFromTranscript(transcript);

  session.insights = insights;
  session.progress = 100;
  session.status = "ready";
  session.processedAt = new Date().toISOString();

  console.log(`Session ${sessionId} processed successfully`);
}

// --------------------
// WHISPER STT
// --------------------

async function transcribeAudioWithWhisper(audioPath) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY in .env");
  }

  if (!fs.existsSync(audioPath)) {
    throw new Error(`Audio file not found: ${audioPath}`);
  }

  const audioStream = fs.createReadStream(audioPath);

  const result = await openai.audio.transcriptions.create({
    file: audioStream,
    model: "whisper-1",
    response_format: "text",
  });

  const text = String(result || "").trim();

  console.log("WHISPER TRANSCRIPT:");
  console.log(text);

  return text;
}

// --------------------
// INSIGHTS + VALIDATION
// --------------------

const COURSE_OUTLINE_PROMPT = `
You are an academic course-outline extractor.
Given a course outline/syllabus, extract only what is explicitly stated.
Return JSON only (no markdown).

Return this exact structure:
{
  "highlights": [{ "text": string, "category": string, "reason": string }],
  "exams": [{ "text": string, "date": "YYYY-MM-DD" | null, "priority": "high" | "med" | "low" | null }],
  "assignments": [{ "text": string, "date": "YYYY-MM-DD" | null, "priority": "high" | "med" | "low" | null }]
}

Rules:
- highlights: 3-6 short phrases (textbook, grading, attendance, policies, prerequisites)
- For each highlight: category (e.g., Grading, Attendance, Materials, Policies, Objectives) and a brief reason.
- exams/assignments: include only if explicitly mentioned as deliverables or tests
- dates: use YYYY-MM-DD when present; null if missing
- priority: high if within 7 days of start term; med within 21 days; low otherwise; null if no date
- do not invent dates or add extra commentary
`;

const OutlineSchema = z.object({
  highlights: z.array(
    z.object({
      text: z.string(),
      category: z.string(),
      reason: z.string(),
    })
  ),
  exams: z.array(
    z.object({
      text: z.string(),
      date: z.string().nullable().optional(),
      priority: z.enum(["high", "med", "low"]).nullable().optional(),
    })
  ),
  assignments: z.array(
    z.object({
      text: z.string(),
      date: z.string().nullable().optional(),
      priority: z.enum(["high", "med", "low"]).nullable().optional(),
    })
  ),
});

async function extractOutlineFromText(text) {
  const response = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: COURSE_OUTLINE_PROMPT.trim() },
      { role: "user", content: text.slice(0, 20000) },
    ],
    temperature: 0.2,
    response_format: { type: "json_object" },
  });

  let raw = response.choices[0]?.message?.content || "";
  raw = raw.trim();

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    console.error("Outline JSON parse error:", e.message);
    throw new Error("Invalid outline JSON");
  }

  const validation = OutlineSchema.safeParse(parsed);
  if (!validation.success) {
    console.error("Outline schema errors:", validation.error.errors);
    throw new Error("Outline schema validation failed");
  }
  return parsed;
}

async function extractTextFromFile(filePath, ext) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Outline file not found: ${filePath}`);
  }

  if (ext === ".pdf") {
    const buffer = fs.readFileSync(filePath);
    const data = await pdfParse(buffer);
    return data.text || "";
  }

  if (ext === ".docx") {
    const result = await mammoth.extractRawText({ path: filePath });
    return result.value || "";
  }

  throw new Error(`Unsupported outline file type: ${ext || "unknown"}`);
}

async function generateInsightsFromTranscript(transcript) {
  const systemPrompt = `
You are an academic study assistant for university students.
Extract concise, high-signal learning insights from lecture transcripts.
Be grounded in the transcript and avoid guessing.
You MUST return valid JSON only.
`;

  const userPrompt = `
Analyze the following lecture transcript and extract learning insights.

Transcript:
"""
${transcript}
"""

Return a JSON object with exactly this structure:
{
  "summary": string,
  "keyConcepts": string[],
  "flashcards": [
    { "question": string, "answer": string }
  ],
  "actionItems": string[]
}

Rules:
- summary: 2-4 sentences, only key takeaways
- keyConcepts: max 5, noun phrases only
- flashcards: max 4, Q/A must be grounded in transcript
- actionItems: 1-3 concrete study tasks tied to content
- Do NOT include text outside JSON
`;

  const response = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.3,
  });

  let raw = response.choices[0].message.content;

  console.log("RAW LLM OUTPUT:");
  console.log(raw);

  // 🔧 Strip ```json ``` wrappers if present
  raw = raw.trim();

  if (raw.startsWith("```")) {
    raw = raw.replace(/^```(?:json)?\s*/i, "");
    raw = raw.replace(/\s*```$/, "");
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    console.error("JSON PARSE ERROR:", e.message);
    throw new Error("LLM returned invalid JSON");
  }

  const validation = InsightsSchema.safeParse(parsed);
  if (!validation.success) {
    console.error("SCHEMA ERRORS:", validation.error.errors);
    throw new Error("LLM output failed schema validation");
  }

  return parsed;
}

// --------------------
// START SERVER
// --------------------

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Backend running on port ${PORT}`);
});

