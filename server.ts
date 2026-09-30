import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { spawn, ChildProcess } from 'child_process';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const isProduction = process.env.NODE_ENV === 'production';
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8001';

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Spawn Python FastAPI backend process if running locally
let pythonProcess: ChildProcess | null = null;

function startPythonBackend() {
  if (process.env.BACKEND_URL && !process.env.BACKEND_URL.includes('127.0.0.1') && !process.env.BACKEND_URL.includes('localhost')) {
    console.log(`[EVOX] Using external Python backend at: ${process.env.BACKEND_URL}`);
    return;
  }

  const backendDir = path.resolve(__dirname, 'backend');
  console.log(`[EVOX] Spawning Python FastAPI backend in ${backendDir} on port 8001...`);

  pythonProcess = spawn(
    'python3',
    ['-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8001', '--reload'],
    {
      cwd: backendDir,
      env: { ...process.env, PYTHONPATH: backendDir, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  pythonProcess.stdout?.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[FastAPI] ${msg}`);
  });

  pythonProcess.stderr?.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) console.error(`[FastAPI] ${msg}`);
  });

  pythonProcess.on('exit', (code) => {
    console.log(`[EVOX] Python backend exited with code ${code}.`);
  });
}

// Proxy all /api requests to the Python FastAPI backend
app.all('/api/*', async (req: Request, res: Response) => {
  const targetUrl = `${BACKEND_URL}${req.originalUrl}`;
  try {
    const headers: Record<string, string> = {
      'Content-Type': req.headers['content-type'] || 'application/json',
    };

    const fetchOptions: RequestInit = {
      method: req.method,
      headers,
    };

    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const response = await fetch(targetUrl, fetchOptions);
    const contentType = response.headers.get('content-type') || '';

    res.status(response.status);

    if (contentType.includes('application/json')) {
      const json = await response.json();
      return res.json(json);
    } else {
      const text = await response.text();
      return res.send(text);
    }
  } catch (err: any) {
    console.error(`[Proxy Error] Failed to reach Python backend at ${targetUrl}:`, err.message);
    res.status(503).json({
      success: false,
      error: 'Python ML Engine is starting or unreachable.',
      details: err.message,
      suggestion: 'Ensure the Python backend is running via: uvicorn app.main:app --port 8000',
    });
  }
});

/* -------------------------------------------------------------
 * STATIC ASSETS & VITE INTEGRATION
 * -----------------------------------------------------------*/

async function startServer() {
  startPythonBackend();

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`[EVOX] Frontend server running on http://localhost:${PORT}`);
    console.log(`[EVOX] Proxying /api/* -> ${BACKEND_URL}`);
  });
}

// Cleanup child process on exit
process.on('SIGINT', () => {
  if (pythonProcess) pythonProcess.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (pythonProcess) pythonProcess.kill();
  process.exit(0);
});

startServer();
