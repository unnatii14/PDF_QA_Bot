const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const rateLimit = require("express-rate-limit");
const session = require("express-session");

const app = express();
app.use(cors());
app.set('trust proxy', 1); // Fix ERR_ERL_UNEXPECTED_X_FORWARDED_FOR
app.use(express.json());

// Session middleware for per-user chat history
app.use(session({
  secret: "pdf-qa-bot-secret-key",
  resave: false,
  saveUninitialized: true,
  cookie: { 
    secure: false,
    maxAge: 1000 * 60 * 60 * 24 // 24 hours
  }
}));

// Rate limiting middleware
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: "Too many PDF uploads from this IP, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  message: "Too many questions asked, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: "Too many summarization requests, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

// Storage for uploaded PDFs
const upload = multer({ dest: "uploads/" });

// Route: Upload PDF
app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Use form field name 'file'." });
    }

    const filePath = path.join(__dirname, req.file.path);

    // **CRITICAL**: Clear session and reset backend state before processing new PDF
    // This prevents cross-document context leakage
    if (req.session) {
      req.session.chatHistory = [];
      req.session.currentPdfSessionId = null;
    }

    // Reset backend state through the /reset endpoint
    try {
      await axios.post("http://localhost:5000/reset");
    } catch (resetError) {
      console.warn("Warning: Could not reset backend state:", resetError.message);
      // Continue with PDF upload even if reset fails
    }

    // Send PDF to Python service for processing
    const uploadResponse = await axios.post("http://localhost:5000/process-pdf", {
      filePath: filePath,
    });

    // Store the new PDF session ID for future validation
    if (uploadResponse.data.session_id && req.session) {
      req.session.currentPdfSessionId = uploadResponse.data.session_id;
    }

    res.json({
      message: "PDF uploaded & processed successfully!",
      session_id: uploadResponse.data.session_id,
      details: uploadResponse.data
    });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Upload processing failed:", details);
    res.status(500).json({ error: "PDF processing failed", details });
  }
});

// Route: Ask Question
app.post("/ask", askLimiter, async (req, res) => {
  try {
    const question = req.body.question;
    
    // Initialize session chat history if it doesn't exist
    if (!req.session.chatHistory) {
      req.session.chatHistory = [];
    }

    // Add user message to session history
    req.session.chatHistory.push({
      role: "user",
      content: question
    });

    // Send question + history to FastAPI
    const response = await axios.post(
      "http://localhost:5000/ask",
      {
        question: question,
        history: req.session.chatHistory
      }
    );

    // Add assistant response to session history
    req.session.chatHistory.push({
      role: "assistant",
      content: response.data.answer
    });

    res.json(response.data);

  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Error asking question" });
  }
});

app.post("/clear-history", (req, res) => {
  // Clear only this user's session history
  if (req.session) {
    req.session.chatHistory = [];
    req.session.currentPdfSessionId = null;
  }
  res.json({ message: "Chat history cleared" });
});

app.get("/pdf-status", async (req, res) => {
  try {
    // Check backend PDF status
    const statusResponse = await axios.get("http://localhost:5000/status");
    
    // Include frontend session status
    const frontendStatus = {
      hasSession: !!req.session,
      hasHistory: req.session?.chatHistory?.length > 0 || false,
      historyLength: req.session?.chatHistory?.length || 0,
      currentSessionId: req.session?.currentPdfSessionId || null
    };

    res.json({
      backend: statusResponse.data,
      frontend: frontendStatus
    });
  } catch (err) {
    console.error("Error fetching PDF status:", err.message);
    res.status(500).json({ error: "Could not fetch PDF status" });
  }
});

app.post("/summarize", summarizeLimiter, async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/summarize", req.body || {});
    res.json({ summary: response.data.summary });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Summarization failed:", details);
    res.status(500).json({ error: "Error summarizing PDF", details });
  }
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));
