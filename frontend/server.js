const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { Pool } = require('pg');

const app = express();
const upload = multer({ dest: 'uploads/' });

const DB_URL = process.env.DATABASE_URL || 'postgresql://neondb_owner:npg_zFYjt0hdEXN9@ep-raspy-snow-anhp06i7-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=verify-full';
const pool = new Pool({
  connectionString: DB_URL,
  max: 1,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 30000,
  allowExitOnIdle: true
});

pool.on('error', (err) => {
  console.error('Postgres pool error:', err);
});

const PORT = process.env.PORT || 3000;
const DASHBOARD_PORT = process.env.DASHBOARD_PORT || 7860;
let dashboardProcess = null;

app.use(cors({
  origin: ['http://localhost:5173', 'http://127.0.0.1:5173'],
  credentials: true
}));
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));
app.use(express.static(path.join(__dirname, '.')));
app.use('/outputs', express.static(path.join(__dirname, '..', 'outputs')));

if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

async function ensureTables() {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        email VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await client.query(`
      ALTER TABLE users
      ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT false NOT NULL;
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS analysis_history (
        id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(id),
        shot_type VARCHAR(50),
        accuracy_score FLOAT,
        entry_text TEXT,
        video_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await client.query(`
      ALTER TABLE analysis_history
      ADD COLUMN IF NOT EXISTS accuracy_score FLOAT,
      ADD COLUMN IF NOT EXISTS video_path TEXT;
    `);

    await client.query(
      `INSERT INTO users (username, password, email, is_admin)
       VALUES ($1, $2, $3, true)
       ON CONFLICT (username) DO UPDATE SET password = EXCLUDED.password, is_admin = true`,
      ['admin', '0000', 'admin@athletiq.local']
    );
  } finally {
    client.release();
  }
}

ensureTables().catch((err) => {
  console.error('Database initialization failed:', err);
});

app.post('/run-analysis', upload.single('video'), async (req, res) => {
  const { shot_type = 'None', click_x = 320, click_y = 240 } = req.body;
  const userId = req.body.user_id ? parseInt(req.body.user_id, 10) : null;
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'Video file is required.' });
  }

  const videoPath = path.resolve(req.file.path);
  const runnerPath = path.join(__dirname, 'analysis_runner.py');
  
  // Prioritize virtual environment python
  const venvPath = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
  const pythonExecutable = fs.existsSync(venvPath) ? venvPath : (process.env.PYTHON || 'python');
  const args = [runnerPath, '--video', videoPath, '--shot', shot_type, '--clickx', click_x.toString(), '--clicky', click_y.toString()];
  if (userId) args.push('--user_id', userId.toString());

  const python = spawn(pythonExecutable, args, { cwd: __dirname });
  let output = '';
  let errorOutput = '';

  python.stdout.on('data', (data) => {
    output += data.toString();
  });
  python.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  python.on('close', async (code) => {
    let result = null;
    let entryText = '';
    if (code !== 0) {
      console.error('Analysis runner error:', errorOutput);
      entryText = `Video: ${req.file.originalname} | Shot: ${shot_type} | Error: ${errorOutput || 'Analysis process failed.'}`;
    } else {
      try {
        // Ultra-robust JSON extraction: try parsing from every '{' in reverse order
        const braceIndices = [];
        for (let i = 0; i < output.length; i++) {
          if (output[i] === '{') braceIndices.push(i);
        }
        
        for (let i = braceIndices.length - 1; i >= 0; i--) {
          try {
            const potentialJson = output.substring(braceIndices[i]).trim();
            // Find the corresponding closing brace or just try parsing to the end
            // JSON.parse is smart enough to ignore trailing whitespace
            result = JSON.parse(potentialJson);
            if (result && typeof result === 'object' && result.success !== undefined) {
              break; 
            }
          } catch (e) { /* ignore and try next brace */ }
        }

        if (!result) throw new Error('No valid JSON output found in runner response');

        if (result.success) {
          // Robustly extract score from summary (matches "Overall Score: 85" or "Score: 85.4%")
          const scoreMatch = result.summary ? result.summary.match(/(?:Overall\s+)?Score:\s*(\d+(?:\.\d+)?)/i) : null;
          const score = scoreMatch ? parseFloat(scoreMatch[1]) : 0;
          entryText = `Shot: ${result.shot_type || shot_type} | Accuracy: ${score}%`;
        } else {
          entryText = `Video: ${req.file.originalname} | Shot: ${shot_type} | Error: ${result.error || 'Analysis failed'}`;
        }
      } catch (e) {
        console.error('Failed to parse analysis runner output. Raw output:', output);
        console.error('Parsing error:', e);
        entryText = `Video: ${req.file.originalname} | Shot: ${shot_type} | Error: Could not parse analysis result.`;
      }
    }

    let history = [];
    if (userId) {
      try {
        const parsedShot = result ? (result.shot_type || shot_type) : shot_type;
        const scoreMatch = result && result.summary ? result.summary.match(/(?:Overall\s+)?Score:\s*(\d+(?:\.\d+)?)/i) : null;
        const parsedScore = scoreMatch ? parseFloat(scoreMatch[1]) : 0;

        await pool.query(
          'INSERT INTO analysis_history (user_id, entry_text, shot_type, accuracy_score) VALUES ($1, $2, $3, $4)',
          [userId, entryText, parsedShot, parsedScore]
        );
        const historyResult = await pool.query(
          'SELECT id, entry_text, shot_type, accuracy_score, created_at FROM analysis_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT 20',
          [userId]
        );
        history = historyResult.rows;
      } catch (saveErr) {
        console.error('Failed to save history entry:', saveErr);
      }
    }

    if (code !== 0) {
      return res.json({ success: false, error: errorOutput || 'Analysis process failed.', history });
    }
    if (!result) {
      return res.json({ success: false, error: 'Could not parse analysis result.', history });
    }
    return res.json({ ...result, history, video_path: videoPath });
  });
});

app.post('/api/upload', upload.single('video'), (req, res) => {
  console.log('Received upload request for:', req.file?.originalname);
  if (!req.file) return res.status(400).json({ success: false, error: 'No file uploaded.' });
  const videoPath = path.resolve(req.file.path);
  res.json({ success: true, video_path: videoPath });
});

app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.json({ success: false, message: 'Username and password are required.' });
  }

  try {
    const result = await pool.query(
      'SELECT id, username, email, is_admin FROM users WHERE username = $1 AND password = $2',
      [username, password]
    );
    if (result.rowCount === 0) {
      return res.json({ success: false, message: 'Invalid credentials.' });
    }

    // WARM START: Launch dashboard in background immediately on login
    // This pre-loads models so the user doesn't wait 15s later.
    http.get(`http://127.0.0.1:${PORT}/launch-dashboard`, () => {
       console.log('Dashboard background warm-start triggered.');
    }).on('error', () => {});

    return res.json({ success: true, user: result.rows[0] });
  } catch (err) {
    console.error('Login error:', err);
    return res.json({ success: false, message: 'Login failed.' });
  }
});

app.post('/api/register', async (req, res) => {
  const { username, password, email } = req.body;
  if (!username || !password) {
    return res.json({ success: false, message: 'Username and password are required.' });
  }

  try {
    const result = await pool.query(
      'INSERT INTO users (username, password, email) VALUES ($1, $2, $3) RETURNING id, username, email',
      [username, password, email || null]
    );
    return res.json({ success: true, user: result.rows[0] });
  } catch (err) {
    console.error('Register error:', err);
    if (err.code === '23505') {
      return res.json({ success: false, message: 'Username already exists.' });
    }
    return res.json({ success: false, message: 'Registration failed.' });
  }
});

app.post('/api/admin/users', async (req, res) => {
  const { username, password } = req.body;
  if (username !== 'admin' || password !== '0000') {
    return res.json({ success: false, message: 'Admin credentials required.' });
  }

  try {
    const result = await pool.query(
      'SELECT id, username, password, email, is_admin, created_at FROM users ORDER BY id'
    );
    return res.json({ success: true, users: result.rows });
  } catch (err) {
    console.error('Admin users error:', err);
    return res.json({ success: false, message: 'Could not load users.' });
  }
});

app.delete('/api/admin/users/:user_id', async (req, res) => {
  const { username, password } = req.body;
  const userId = parseInt(req.params.user_id, 10);

  if (username !== 'admin' || password !== '0000') {
    return res.json({ success: false, message: 'Admin credentials required.' });
  }
  if (!userId) {
    return res.json({ success: false, message: 'User id is required.' });
  }

  try {
    const target = await pool.query('SELECT username, is_admin FROM users WHERE id = $1', [userId]);
    if (target.rowCount === 0) {
      return res.json({ success: false, message: 'User not found.' });
    }
    if (target.rows[0].is_admin) {
      return res.json({ success: false, message: 'Cannot delete admin.' });
    }

    await pool.query('DELETE FROM analysis_history WHERE user_id = $1', [userId]);
    await pool.query('DELETE FROM users WHERE id = $1', [userId]);
    return res.json({ success: true });
  } catch (err) {
    console.error('Admin delete error:', err);
    return res.json({ success: false, message: 'Could not delete user.' });
  }
});

app.post('/api/save-analysis', async (req, res) => {
  const { user_id, shot_type, score, video_path } = req.body;
  if (!user_id) return res.status(400).json({ success: false, error: 'User ID is required' });

  try {
    const entryText = `Shot: ${shot_type} | Accuracy: ${score}%`;
    await pool.query(
      'INSERT INTO analysis_history (user_id, shot_type, accuracy_score, entry_text, video_path) VALUES ($1, $2, $3, $4, $5)',
      [user_id, shot_type, score, entryText, video_path]
    );
    res.json({ success: true });
  } catch (err) {
    console.error('Save analysis error:', err);
    res.status(500).json({ success: false, error: 'Database save failed' });
  }
});

app.get('/api/history/:user_id', async (req, res) => {
  const userId = parseInt(req.params.user_id, 10);
  if (!userId) {
    return res.json({ success: false, history: [] });
  }

  try {
    const result = await pool.query(
      'SELECT id, entry_text, shot_type, accuracy_score, created_at FROM analysis_history WHERE user_id = $1 ORDER BY created_at DESC',
      [userId]
    );
    return res.json({ success: true, history: result.rows });
  } catch (err) {
    console.error('History error:', err);
    return res.json({ success: false, history: [] });
  }
});

app.delete('/api/history/:user_id/:entry_id', async (req, res) => {
  const userId = parseInt(req.params.user_id, 10);
  const entryId = parseInt(req.params.entry_id, 10);
  if (!userId || !entryId) {
    return res.status(400).json({ success: false, message: 'Invalid request.' });
  }

  try {
    const result = await pool.query(
      'DELETE FROM analysis_history WHERE id = $1 AND user_id = $2 RETURNING id',
      [entryId, userId]
    );
    if (result.rowCount === 0) {
      return res.status(404).json({ success: false, message: 'History entry not found.' });
    }
    const historyResult = await pool.query(
      'SELECT id, entry_text, created_at FROM analysis_history WHERE user_id = $1 ORDER BY created_at DESC',
      [userId]
    );
    return res.json({ success: true, history: historyResult.rows });
  } catch (err) {
    console.error('Delete history error:', err);
    return res.status(500).json({ success: false, message: 'Could not delete history.' });
  }
});

function waitForDashboard(port, timeoutMs = 15000) {
  const url = `http://127.0.0.1:${port}`;
  const start = Date.now();

  return new Promise((resolve) => {
    const check = () => {
      const req = http.request({ hostname: '127.0.0.1', port, path: '/', method: 'GET', timeout: 2000 }, (res) => {
        res.resume();
        resolve(true);
      });

      req.on('error', () => {
        if (Date.now() - start >= timeoutMs) {
          resolve(false);
        } else {
          setTimeout(check, 500);
        }
      });

      req.on('timeout', () => {
        req.destroy();
        if (Date.now() - start >= timeoutMs) {
          resolve(false);
        } else {
          setTimeout(check, 500);
        }
      });

      req.end();
    };

    check();
  });
}

app.get('/launch-dashboard', async (req, res) => {
  const { video, user_id } = req.query;
  const baseUrl = `http://127.0.0.1:${DASHBOARD_PORT}`;
  const normalizedVideo = video ? path.normalize(path.resolve(video)) : null;
  let urlWithParams = normalizedVideo ? `${baseUrl}/?video=${encodeURIComponent(normalizedVideo)}` : baseUrl;
  if (user_id) {
    urlWithParams += (urlWithParams.includes('?') ? '&' : '/?') + `user_id=${user_id}`;
  }

  if (dashboardProcess && !dashboardProcess.killed) {
    return res.json({ success: true, url: urlWithParams });
  }

  const dashboardScript = path.join(__dirname, '..', 'app', 'main.py');
  // Prioritize virtual environment python
  const venvPath = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
  const pythonExecutable = fs.existsSync(venvPath) ? venvPath : (process.env.PYTHON || 'python');
  
  console.log('Launching Dashboard...');
  console.log('Python:', pythonExecutable);
  console.log('Script:', dashboardScript);
  console.log('CWD:', path.join(__dirname, '..', 'app'));

  dashboardProcess = spawn(pythonExecutable, [dashboardScript], {
    cwd: path.join(__dirname, '..', 'app'),
    env: { ...process.env, GRADIO_SERVER_PORT: DASHBOARD_PORT.toString() },
    stdio: ['pipe', 'pipe', 'pipe'],
    detached: false,
    shell: true
  });

  let launchError = null;
  dashboardProcess.stdout.on('data', (chunk) => {
    const msg = chunk.toString();
    console.log('Dashboard stdout:', msg.trim());
    if (msg.toLowerCase().includes('running on port')) {
      console.log('Dashboard launched.');
    }
  });
  dashboardProcess.stderr.on('data', (chunk) => {
    const msg = chunk.toString();
    console.error('Dashboard error:', msg.trim());
    launchError = msg.trim();
  });
  const ready = await waitForDashboard(DASHBOARD_PORT, 30000);
  if (!ready && launchError) {
    return res.status(500).json({ success: false, message: 'Failed to start dashboard', error: launchError });
  }

  return res.json({ success: true, url: urlWithParams });
});

// --- BACKGROUND WARM START ---
function warmStartDashboard() {
  const dashboardScript = path.join(__dirname, '..', 'app', 'main.py');
  const venvPath = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
  const pythonExecutable = fs.existsSync(venvPath) ? venvPath : (process.env.PYTHON || 'python');
  
  if (dashboardProcess && !dashboardProcess.killed) return;

  console.log('--- Dashboard background warm-start triggered ---');
  dashboardProcess = spawn(pythonExecutable, [dashboardScript], {
    cwd: path.join(__dirname, '..', 'app'),
    env: { ...process.env, GRADIO_SERVER_PORT: DASHBOARD_PORT.toString() },
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: true
  });
  
  dashboardProcess.stdout.on('data', (chunk) => {
    const msg = chunk.toString();
    if (msg.includes('AthletiQ is ready')) console.log('Dashboard is WARM and READY.');
  });
}

// Trigger warm-start on server launch
setTimeout(warmStartDashboard, 1000);

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Frontend server running on http://127.0.0.1:${PORT}`);
});