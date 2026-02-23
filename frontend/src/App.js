import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Document, Page, pdfjs } from "react-pdf";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Row,
  Col,
  Button,
  Form,
  Card,
  Spinner,
  Navbar,
} from "react-bootstrap";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const API_BASE = process.env.REACT_APP_API_URL || "";
const THEME_STORAGE_KEY = "pdf-qa-bot-theme";

function App() {
  // -----------------------------
  // Core State
  // -----------------------------
  const [file, setFile] = useState(null);
  const [pdfs, setPdfs] = useState([]); // { name, doc_id, url }
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [question, setQuestion] = useState("");

  // -----------------------------
  // UI State
  // -----------------------------
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);

  // -----------------------------
  // Theme
  // -----------------------------
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return saved ? JSON.parse(saved) : false;
  });

  // -----------------------------
  // Session Isolation
  // -----------------------------
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    setSessionId(
      crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2, 15)
    );
  }, []);

  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
    document.body.classList.toggle("dark-mode", darkMode);
  }, [darkMode]);

  // -----------------------------
  // Upload PDF (with timeout)
  // -----------------------------
  const uploadPDF = async () => {
    if (!file) return;

    setUploading(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("sessionId", sessionId);

      const res = await axios.post(`${API_BASE}/upload`, formData, {
        signal: controller.signal,
      });

      const url = URL.createObjectURL(file);
      setPdfs((prev) => [
        ...prev,
        { name: file.name, doc_id: res.data.doc_id, url },
      ]);

      setFile(null);
      alert("PDF uploaded!");
    } catch (e) {
      let msg = "Upload failed.";
      if (e.name === "AbortError" || e.code === "ECONNABORTED") {
        msg = "Upload timed out. Try a smaller PDF.";
      } else if (e.response?.status === 504) {
        msg = "Gateway timeout.";
      }
      alert(msg);
    } finally {
      clearTimeout(timeoutId);
      setUploading(false);
    }
  };

  // -----------------------------
  // Toggle PDF Selection
  // -----------------------------
  const toggleDocSelection = (docId) => {
    setComparisonResult(null);
    setSelectedDocs((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  };

  // -----------------------------
  // Ask Question (with timeout)
  // -----------------------------
  const askQuestion = async () => {
    if (!question.trim() || selectedDocs.length === 0) return;
    if (question.length > 2000) {
      alert("Question too long (max 2000 chars)");
      return;
    }

    setAsking(true);
    setChatHistory((prev) => [...prev, { role: "user", text: question }]);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    try {
      const res = await axios.post(
        `${API_BASE}/ask`,
        {
          question,
          sessionId,
          doc_ids: selectedDocs,
        },
        { signal: controller.signal }
      );

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.answer },
      ]);
    } catch (e) {
      let msg = "Error getting answer.";
      if (e.name === "AbortError" || e.code === "ECONNABORTED") {
        msg = "Request timed out.";
      }
      setChatHistory((prev) => [...prev, { role: "bot", text: msg }]);
    } finally {
      clearTimeout(timeoutId);
      setQuestion("");
      setAsking(false);
    }
  };

  // -----------------------------
  // Summarize
  // -----------------------------
  const summarizePDF = async () => {
    if (selectedDocs.length === 0) return;

    setSummarizing(true);

    try {
      const res = await axios.post(`${API_BASE}/summarize`, {
        sessionId,
        doc_ids: selectedDocs,
      });

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.summary },
      ]);
    } catch {
      alert("Error summarizing.");
    }

    setSummarizing(false);
  };

  // -----------------------------
  // Compare
  // -----------------------------
  const compareDocuments = async () => {
    if (selectedDocs.length < 2) return;

    setComparing(true);
    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        sessionId,
        doc_ids: selectedDocs,
      });

      if (selectedDocs.length === 2) {
        setComparisonResult(res.data.comparison);
      } else {
        setChatHistory((prev) => [
          ...prev,
          { role: "bot", text: res.data.comparison },
        ]);
      }
    } catch {
      alert("Error comparing documents.");
    }
    setComparing(false);
  };

  const selectedPdfs = pdfs.filter((p) =>
    selectedDocs.includes(p.doc_id)
  );

  const pageBg = darkMode ? "bg-dark text-light" : "bg-light text-dark";
  const cardClass = darkMode
    ? "text-white border-secondary shadow"
    : "bg-white text-dark border-0 shadow-sm";

  // -----------------------------
  // Render
  // -----------------------------
  return (
    <div className={pageBg} style={{ minHeight: "100vh" }}>
      <Navbar bg={darkMode ? "dark" : "primary"} variant="dark">
        <Container className="d-flex justify-content-between">
          <Navbar.Brand>🤖 PDF Q&A Bot</Navbar.Brand>
          <Button
            variant="outline-light"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? "Light" : "Dark"}
          </Button>
        </Container>
      </Navbar>

      <Container className="mt-4">
        {/* Upload */}
        <Card className={`mb-4 ${cardClass}`}>
          <Card.Body>
            <Form>
              <Form.Control
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
              />
              <Button
                className="mt-2"
                onClick={uploadPDF}
                disabled={!file || uploading}
              >
                {uploading ? <Spinner size="sm" /> : "Upload"}
              </Button>
            </Form>
          </Card.Body>
        </Card>

        {/* Selection */}
        {pdfs.length > 0 && (
          <Card className={`mb-4 ${cardClass}`}>
            <Card.Body>
              <h5>Select Documents</h5>
              {pdfs.map((pdf) => (
                <Form.Check
                  key={pdf.doc_id}
                  label={pdf.name}
                  checked={selectedDocs.includes(pdf.doc_id)}
                  onChange={() => toggleDocSelection(pdf.doc_id)}
                />
              ))}
            </Card.Body>
          </Card>
        )}

        {/* Side-by-side view */}
        {selectedPdfs.length === 2 && (
          <Row className="mb-4">
            {selectedPdfs.map((pdf) => (
              <Col md={6} key={pdf.doc_id}>
                <Card className={cardClass}>
                  <Card.Body>
                    <h6>{pdf.name}</h6>
                    <Document file={pdf.url}>
                      <Page pageNumber={1} />
                    </Document>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        )}

        {/* Chat */}
        <Card className={cardClass}>
          <Card.Body>
            <div style={{ maxHeight: 300, overflowY: "auto" }}>
              {chatHistory.map((msg, i) => (
                <div key={i} className="mb-2">
                  <strong>{msg.role === "user" ? "You" : "Bot"}:</strong>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ))}
            </div>

            <Form className="d-flex gap-2 mt-3">
              <Form.Control
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question..."
              />
              <Button onClick={askQuestion} disabled={asking}>
                {asking ? <Spinner size="sm" /> : "Ask"}
              </Button>
            </Form>

            <div className="mt-3">
              <Button
                variant="warning"
                className="me-2"
                onClick={summarizePDF}
                disabled={summarizing}
              >
                Summarize
              </Button>

              <Button
                variant="info"
                onClick={compareDocuments}
                disabled={selectedDocs.length < 2 || comparing}
              >
                Compare
              </Button>
            </div>
          </Card.Body>
        </Card>
      </Container>
    </div>
  );
}

export default App;